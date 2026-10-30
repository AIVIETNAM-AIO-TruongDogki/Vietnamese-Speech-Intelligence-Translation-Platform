"""Module State: Quản lý trạng thái luồng xử lý (Pipeline State) cho VietSpeech AI.
Được thiết kế tương thích với TypedDict của LangGraph, lưu trữ toàn bộ dữ liệu trung gian
từ Audio thô tới kết quả xuất bản phụ đề và tóm tắt.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict
import numpy as np


class VietSpeechState(TypedDict, total=False):
    """Trạng thái toàn diện của pipeline điều phối."""
    # 1. Thông tin Audio đầu vào
    audio_path: str
    audio_data: Optional[np.ndarray]
    sample_rate: int
    duration_sec: float

    # 2. Phân đoạn giọng nói (VAD)
    speech_segments: List[Any]

    # 3. Kết quả ASR thô & Độ tin cậy
    raw_transcript: str
    asr_segments: List[Any]
    asr_confidence: float
    retry_count: int
    quality_status: str  # "PASSED", "LOW_CONFIDENCE", "ERROR"

    # 4. Tinh chỉnh ngữ âm & chuẩn hóa phương ngữ (VPA)
    refined_transcript: str

    # 5. Phân tách người nói & Căn chỉnh từ (Diarization & Alignment)
    speaker_segments: List[Any]
    diarized_turns: List[Any]

    # 6. Dịch thuật Việt - Anh
    translated_segments: List[Any]

    # 7. Trí tuệ & Tóm tắt hội thoại
    intelligence_report: Any

    # 8. Kết quả xuất bản tệp
    final_exports: Dict[str, str]  # {"srt": "...", "vtt": "...", "md": "...", "json": "..."}
    error_message: Optional[str]
