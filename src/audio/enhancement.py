"""Module Audio Enhancement: Giảm nhiễu (Noise Reduction), lọc tạp âm và cân bằng phổ tín hiệu.
Giúp làm sạch âm thanh trước khi nhận diện giọng địa phương, đặc biệt hiệu quả với tạp âm môi trường,
tiếng rè mic hoặc tiếng ồn quạt máy.
"""
from __future__ import annotations

import numpy as np


class AudioEnhancer:
    """Bộ tăng cường chất lượng âm thanh và khử nhiễu nền."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def enhance(
        self,
        audio_data: np.ndarray,
        reduce_noise: bool = True,
        high_pass_filter: bool = True,
    ) -> np.ndarray:
        """Thực hiện khử nhiễu và lọc dải tần giọng nói người.

        Args:
            audio_data: Mảng 1D float32.
            reduce_noise: Bật thuật toán khử nhiễu phổ (spectral gating).
            high_pass_filter: Lọc bỏ tần số siêu trầm (< 70Hz) do rung chấn mic hoặc gió.

        Returns:
            np.ndarray: Tín hiệu âm thanh sau khi tăng cường.
        """
        if len(audio_data) == 0:
            return audio_data

        clean_audio = audio_data.copy()

        # 1. High-pass filter loại bỏ DC offset và rung chấn (< 70Hz)
        if high_pass_filter:
            try:
                from scipy import signal
                sos = signal.butter(4, 70.0, 'hp', fs=self.sample_rate, output='sos')
                clean_audio = signal.sosfilt(sos, clean_audio)
            except Exception:
                pass

        # 2. Khử nhiễu nền Spectral Gating nếu có noisereduce
        if reduce_noise:
            try:
                import noisereduce as nr
                clean_audio = nr.reduce_noise(
                    y=clean_audio,
                    sr=self.sample_rate,
                    stationary=True,
                    prop_decrease=0.75,
                )
            except ImportError:
                # Nếu không có thư viện noisereduce, dùng Wiener filter đơn giản hoặc Spectral Subtraction
                clean_audio = self._simple_spectral_gate(clean_audio)

        return clean_audio.astype(np.float32)

    def _simple_spectral_gate(self, audio_data: np.ndarray) -> np.ndarray:
        """Cổng nhiễu mềm đơn giản dựa trên biến thiên biên độ."""
        threshold = 0.008
        gain = np.where(np.abs(audio_data) < threshold, 0.4, 1.0)
        return audio_data * gain
