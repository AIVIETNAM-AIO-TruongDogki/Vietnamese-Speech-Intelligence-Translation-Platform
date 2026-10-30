"""Module Audio Loader: Đọc và tiền xử lý âm thanh đa định dạng (WAV, MP3, MP4, FLAC, M4A).
Kế thừa nguyên tắc từ Zalo_AI: Ưu tiên đọc trực tiếp bằng soundfile (float32, 16kHz mono),
không phụ thuộc vào gọi ffmpeg subprocess dễ lỗi trên môi trường Windows.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


class AudioLoader:
    """Bộ đọc và chuẩn hóa tín hiệu âm thanh chuẩn 16kHz mono."""

    TARGET_SAMPLE_RATE = 16000

    @classmethod
    def load_audio(
        cls,
        audio_source: Union[str, Path, bytes, np.ndarray],
        sample_rate: Optional[int] = None,
        target_sr: int = TARGET_SAMPLE_RATE,
        normalize_volume: bool = True,
    ) -> Tuple[np.ndarray, int, float]:
        """Tải âm thanh từ file path, bytes hoặc ndarray và chuẩn hóa thành float32 mono 16kHz.

        Args:
            audio_source: Đường dẫn file, mảng bytes hoặc numpy array.
            sample_rate: Sample rate gốc (nếu truyền numpy array).
            target_sr: Sample rate đích (mặc định 16000 Hz cho PhoWhisper / VAD).
            normalize_volume: Chuẩn hóa biên độ âm lượng (Peak Normalization).

        Returns:
            Tuple[np.ndarray, int, float]:
                - audio_data: Mảng 1D float32 trong khoảng [-1.0, 1.0].
                - sample_rate: Sample rate đích (16000).
                - duration: Thời lượng âm thanh tính theo giây.
        """
        import soundfile as sf

        # Trường hợp 1: Đã là numpy array
        if isinstance(audio_source, np.ndarray):
            data = audio_source
            sr = sample_rate or target_sr
        # Trường hợp 2: Dữ liệu nhị phân bytes
        elif isinstance(audio_source, bytes):
            buffer = io.BytesIO(audio_source)
            data, sr = sf.read(buffer, dtype="float32")
        # Trường hợp 3: Đường dẫn file
        else:
            file_path = str(Path(audio_source).resolve())
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Không tìm thấy file âm thanh: {file_path}")

            # Thử đọc trực tiếp bằng soundfile
            try:
                data, sr = sf.read(file_path, dtype="float32")
            except Exception as e_sf:
                # Fallback sang librosa cho các file MP4, M4A, MP3 nếu soundfile thiếu plugin
                try:
                    import librosa
                    data, sr = librosa.load(file_path, sr=None, mono=False, dtype=np.float32)
                    if data.ndim > 1:
                        data = data.T  # Đưa về (frames, channels)
                except Exception as e_lib:
                    raise RuntimeError(
                        f"Không thể đọc file âm thanh {file_path}. soundfile error: {e_sf}, librosa error: {e_lib}"
                    )

        # 1. Chuyển đổi sang Mono nếu là Stereo / Multi-channel
        if data.ndim > 1:
            # Trung bình cộng các kênh
            data = np.mean(data, axis=1)

        # 2. Resampling về target_sr nếu cần (16kHz)
        if sr != target_sr:
            try:
                import librosa
                data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            except Exception:
                from scipy import signal
                num_samples = int(len(data) * float(target_sr) / sr)
                data = signal.resample(data, num_samples).astype(np.float32)
                sr = target_sr

        # 3. Chuẩn hóa float32 trong [-1.0, 1.0]
        data = data.astype(np.float32)

        # 4. Peak Normalization nếu bật
        if normalize_volume and len(data) > 0:
            max_val = np.max(np.abs(data))
            if max_val > 1e-6:
                data = data / max_val * 0.95

        duration = float(len(data)) / float(sr) if sr > 0 else 0.0
        return data, sr, duration
