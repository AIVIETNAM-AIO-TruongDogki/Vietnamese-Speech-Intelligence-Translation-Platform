"""Module VAD: Phát hiện giọng nói (Voice Activity Detection) và phân đoạn âm thanh (Chunking).
Sử dụng Silero VAD hoặc bộ lọc năng lượng RMS fallback khi chạy offline hoàn toàn.
Giúp chia audio dài thành các chunk tự nhiên theo khoảng lặng, chống hiện tượng PhoWhisper
bị lặp từ (hallucination) trong các đoạn im lặng hoặc nhạc nền.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SpeechSegment:
    """Đoạn âm thanh chứa giọng nói đã được phân đoạn."""
    start_sec: float
    end_sec: float
    start_sample: int
    end_sample: int
    audio: np.ndarray

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


class VoiceActivityDetector:
    """Bộ nhận diện khoảng giọng nói và phân đoạn âm thanh."""

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 400,
        max_speech_duration_s: float = 28.0,
        device: str = "cpu",
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.max_speech_duration_s = max_speech_duration_s
        self.device = device
        self._model = None
        self._utils = None

    def _init_silero(self):
        """Khởi tạo Silero VAD từ torch.hub hoặc cục bộ."""
        if self._model is not None:
            return

        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            model.to(self.device)
            model.eval()
            self._model = model
            self._utils = utils
        except Exception as e:
            # Fallback nếu không có internet hoặc torch hub lỗi
            self._model = "fallback"

    def detect_segments(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
    ) -> List[SpeechSegment]:
        """Phát hiện các phân đoạn có tiếng nói và trả về danh sách SpeechSegment.

        Args:
            audio_data: Mảng 1D float32.
            sample_rate: Tần số lấy mẫu (khuyến nghị 16000).

        Returns:
            Danh sách các SpeechSegment kèm timestamp (start_sec, end_sec).
        """
        if len(audio_data) == 0:
            return []

        self._init_silero()

        if self._model != "fallback" and self._model is not None:
            try:
                import torch
                (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = self._utils
                tensor_audio = torch.from_numpy(audio_data).float().to(self.device)
                timestamps = get_speech_timestamps(
                    tensor_audio,
                    self._model,
                    sampling_rate=sample_rate,
                    threshold=self.threshold,
                    min_speech_duration_ms=self.min_speech_duration_ms,
                    min_silence_duration_ms=self.min_silence_duration_ms,
                    max_speech_duration_s=self.max_speech_duration_s,
                )

                segments = []
                for ts in timestamps:
                    s_sample = ts["start"]
                    e_sample = min(ts["end"], len(audio_data))
                    s_sec = s_sample / float(sample_rate)
                    e_sec = e_sample / float(sample_rate)
                    chunk = audio_data[s_sample:e_sample]
                    segments.append(
                        SpeechSegment(
                            start_sec=s_sec,
                            end_sec=e_sec,
                            start_sample=s_sample,
                            end_sample=e_sample,
                            audio=chunk,
                        )
                    )
                if segments:
                    return segments
            except Exception:
                pass

        # Fallback thuật toán phân đoạn dựa trên khung năng lượng (Energy-based VAD)
        return self._energy_based_vad(audio_data, sample_rate)

    def _energy_based_vad(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
    ) -> List[SpeechSegment]:
        """Bộ phân đoạn dự phòng dùng RMS năng lượng và Zero Crossing Rate."""
        frame_len = int(sample_rate * 0.03)  # 30ms
        hop_len = int(sample_rate * 0.01)    # 10ms
        num_frames = (len(audio_data) - frame_len) // hop_len + 1

        if num_frames <= 0:
            return [
                SpeechSegment(
                    start_sec=0.0,
                    end_sec=len(audio_data) / float(sample_rate),
                    start_sample=0,
                    end_sample=len(audio_data),
                    audio=audio_data,
                )
            ]

        # Tính RMS năng lượng từng frame
        frames = np.lib.stride_tricks.sliding_window_view(audio_data, frame_len)[::hop_len]
        energy = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)

        # Ngưỡng thích ứng (Adaptive Threshold)
        silence_level = np.percentile(energy, 20)
        speech_thresh = max(silence_level * 2.5, 0.015)

        is_speech = energy > speech_thresh

        # Gom các frame liên tục
        min_speech_frames = int((self.min_speech_duration_ms / 1000.0) * sample_rate / hop_len)
        min_silence_frames = int((self.min_silence_duration_ms / 1000.0) * sample_rate / hop_len)
        max_speech_samples = int(self.max_speech_duration_s * sample_rate)

        segments: List[SpeechSegment] = []
        in_speech = False
        start_frame = 0
        silence_counter = 0

        for i, speech in enumerate(is_speech):
            if speech:
                if not in_speech:
                    in_speech = True
                    start_frame = i
                silence_counter = 0
            else:
                if in_speech:
                    silence_counter += 1
                    curr_samples = (i - start_frame) * hop_len
                    if silence_counter >= min_silence_frames or curr_samples >= max_speech_samples:
                        # Kết thúc 1 đoạn
                        if (i - silence_counter - start_frame) >= min_speech_frames:
                            s_sample = start_frame * hop_len
                            e_sample = min((i - silence_counter) * hop_len + frame_len, len(audio_data))
                            segments.append(
                                SpeechSegment(
                                    start_sec=s_sample / float(sample_rate),
                                    end_sec=e_sample / float(sample_rate),
                                    start_sample=s_sample,
                                    end_sample=e_sample,
                                    audio=audio_data[s_sample:e_sample],
                                )
                            )
                        in_speech = False
                        silence_counter = 0

        if in_speech and (len(is_speech) - start_frame) >= min_speech_frames:
            s_sample = start_frame * hop_len
            e_sample = len(audio_data)
            segments.append(
                SpeechSegment(
                    start_sec=s_sample / float(sample_rate),
                    end_sec=e_sample / float(sample_rate),
                    start_sample=s_sample,
                    end_sample=e_sample,
                    audio=audio_data[s_sample:e_sample],
                )
            )

        # Nếu không bắt được segment nào (audio quá nhỏ hoặc đều), trả về toàn bộ audio
        if not segments:
            segments = [
                SpeechSegment(
                    start_sec=0.0,
                    end_sec=len(audio_data) / float(sample_rate),
                    start_sample=0,
                    end_sample=len(audio_data),
                    audio=audio_data,
                )
            ]

        return segments
