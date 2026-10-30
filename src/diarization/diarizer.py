"""Module Speaker Diarization: Phân tách người nói (Ai đang nói gì vào lúc nào).
Phân biệt rạch ròi với bài toán Alignment:
- Diarization: Xác định nhãn người nói (Speaker 0, Speaker 1, Speaker 2...) cho từng khoảng thời gian.
- Hỗ trợ pyannote.audio khi có token HF, và tích hợp thuật toán phân cụm âm học (Acoustic Clustering)
  dự phòng offline không phụ thuộc internet.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class SpeakerSegment:
    """Khoảng thời gian một người nói phát biểu."""
    speaker: str
    start_sec: float
    end_sec: float


class SpeakerDiarizer:
    """Bộ nhận diện và phân tách người nói trong hội thoại."""

    def __init__(self, use_auth_token: Optional[str] = None, device: str = "cpu"):
        self.auth_token = use_auth_token
        self.device = device
        self._pipeline = None
        self._has_pyannote = False
        self._init_pipeline()

    def _init_pipeline(self):
        """Khởi tạo pyannote pipeline nếu có thư viện và token."""
        if not self.auth_token:
            return

        try:
            from pyannote.audio import Pipeline
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.auth_token,
            )
            if self._pipeline and self.device == "cuda":
                import torch
                self._pipeline.to(torch.device("cuda"))
            self._has_pyannote = True
        except Exception:
            self._has_pyannote = False

    def diarize(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        num_speakers: Optional[int] = None,
    ) -> List[SpeakerSegment]:
        """Thực hiện phân tách người nói.

        Args:
            audio_data: Mảng 1D float32.
            sample_rate: Tần số lấy mẫu (16000).
            num_speakers: Số lượng người nói định trước (nếu biết trước).

        Returns:
            Danh sách SpeakerSegment kèm nhãn speaker và mốc thời gian.
        """
        duration = len(audio_data) / float(sample_rate)
        if duration <= 0:
            return []

        # 1. Nếu có pyannote đã nạp
        if self._has_pyannote and self._pipeline:
            try:
                import torch
                tensor_audio = torch.from_numpy(audio_data).unsqueeze(0).float()
                diarization_out = self._pipeline(
                    {"waveform": tensor_audio, "sample_rate": sample_rate},
                    num_speakers=num_speakers,
                )
                segments = []
                for turn, _, speaker in diarization_out.itertracks(yield_label=True):
                    segments.append(
                        SpeakerSegment(
                            speaker=speaker,
                            start_sec=float(turn.start),
                            end_sec=float(turn.end),
                        )
                    )
                if segments:
                    return segments
            except Exception:
                pass

        # 2. Phân cụm âm học offline (Acoustic Spectral Clustering Fallback)
        return self._acoustic_clustering_diarize(audio_data, sample_rate, num_speakers=num_speakers or 2)

    def _acoustic_clustering_diarize(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        num_speakers: int = 2,
    ) -> List[SpeakerSegment]:
        """Thuật toán phân tách người nói cục bộ dựa trên đặc trưng phổ MFCC và Energy."""
        win_s = 1.0     # Cửa sổ 1 giây
        hop_s = 0.5     # Bước nhảy 0.5 giây
        win_samples = int(win_s * sample_rate)
        hop_samples = int(hop_s * sample_rate)

        num_chunks = max(1, (len(audio_data) - win_samples) // hop_samples + 1)
        features = []
        time_stamps = []

        for i in range(num_chunks):
            s = i * hop_samples
            e = s + win_samples
            chunk = audio_data[s:e]
            if len(chunk) < win_samples:
                break

            # Đo năng lượng
            rms = np.sqrt(np.mean(chunk ** 2) + 1e-12)
            if rms < 0.01:
                continue

            # Trích xuất đặc trưng phổ FFT đơn giản (tần số cơ bản, trọng tâm phổ)
            fft_mag = np.abs(np.fft.rfft(chunk))
            freqs = np.fft.rfftfreq(len(chunk), 1.0 / sample_rate)
            spectral_centroid = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-12)
            spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_mag) / (np.sum(fft_mag) + 1e-12))
            pitch_proxy = np.argmax(fft_mag[10:300]) + 10  # Đại diện cao độ giọng nói

            features.append([spectral_centroid / 4000.0, spectral_spread / 4000.0, pitch_proxy / 300.0])
            time_stamps.append((s / float(sample_rate), e / float(sample_rate)))

        if len(features) < 2:
            return [
                SpeakerSegment(
                    speaker="SPEAKER_00",
                    start_sec=0.0,
                    end_sec=len(audio_data) / float(sample_rate),
                )
            ]

        # Phân cụm K-Means đơn giản
        feat_arr = np.array(features)
        labels = self._simple_kmeans(feat_arr, k=min(num_speakers, len(feat_arr)))

        # Gom các khoảng thời gian liên tục của cùng 1 speaker
        merged: List[SpeakerSegment] = []
        curr_spk = f"SPEAKER_{labels[0]:02d}"
        s_start, s_end = time_stamps[0]

        for idx in range(1, len(labels)):
            spk = f"SPEAKER_{labels[idx]:02d}"
            t_start, t_end = time_stamps[idx]

            if spk == curr_spk and (t_start - s_end) <= 0.8:
                s_end = t_end
            else:
                merged.append(SpeakerSegment(speaker=curr_spk, start_sec=s_start, end_sec=s_end))
                curr_spk = spk
                s_start, s_end = t_start, t_end

        merged.append(SpeakerSegment(speaker=curr_spk, start_sec=s_start, end_sec=s_end))
        return merged

    def _simple_kmeans(self, x: np.ndarray, k: int, max_iter: int = 10) -> np.ndarray:
        """K-means clustering thuần numpy không cần scikit-learn."""
        np.random.seed(42)
        indices = np.random.choice(len(x), size=k, replace=False)
        centroids = x[indices]

        labels = np.zeros(len(x), dtype=int)
        for _ in range(max_iter):
            dists = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for cluster_idx in range(k):
                members = x[labels == cluster_idx]
                if len(members) > 0:
                    centroids[cluster_idx] = np.mean(members, axis=0)

        return labels
