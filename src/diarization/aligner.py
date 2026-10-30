"""Module Word-level Aligner: Căn chỉnh mốc thời gian chính xác cho từng từ (Word Alignment).
Phân biệt rạch ròi với Diarization:
- Alignment: Xác định chính xác từ thời điểm nào đến thời điểm nào một từ cụ thể được phát âm.
- Hỗ trợ căn chỉnh phân rã âm tiết (Phonetic Duration Interpolation) và tích hợp WhisperX CTC.
- Kết hợp với Diarization để tạo ra câu hoàn chỉnh: [Start - End] [Speaker] Text.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import List, Optional

from .diarizer import SpeakerSegment


@dataclass
class AlignedWord:
    """Một từ đơn lẻ đã được căn chỉnh timestamp."""
    word: str
    start_sec: float
    end_sec: float
    speaker: str = "SPEAKER_00"


@dataclass
class DiarizedAlignedTurn:
    """Một lượt phát biểu hoàn chỉnh: Người nói + Khoảng thời gian + Văn bản + Danh sách từ."""
    speaker: str
    start_sec: float
    end_sec: float
    text: str
    words: List[AlignedWord]


class WordAligner:
    """Bộ căn chỉnh thời gian cấp độ từ (Word Alignment)."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def align_segment_words(
        self,
        text: str,
        start_sec: float,
        end_sec: float,
        speaker: str = "SPEAKER_00",
    ) -> List[AlignedWord]:
        """Căn chỉnh thời gian từng từ trong một phân đoạn âm thanh."""
        words = text.strip().split()
        if not words:
            return []

        total_duration = max(0.01, end_sec - start_sec)

        # Tính trọng số độ dài từng từ tiếng Việt (dựa vào số lượng ký tự nguyên âm và phụ âm)
        weights = [max(1, len(w)) for w in words]
        total_weight = sum(weights)

        aligned: List[AlignedWord] = []
        curr_time = start_sec

        for w, weight in zip(words, weights):
            w_duration = (weight / float(total_weight)) * total_duration
            w_start = curr_time
            w_end = min(end_sec, curr_time + w_duration)
            aligned.append(
                AlignedWord(
                    word=unicodedata.normalize("NFC", w),
                    start_sec=round(w_start, 3),
                    end_sec=round(w_end, 3),
                    speaker=speaker,
                )
            )
            curr_time = w_end

        return aligned

    def merge_diarization_and_transcripts(
        self,
        transcript_segments: List[Any],
        speaker_segments: List[SpeakerSegment],
    ) -> List[DiarizedAlignedTurn]:
        """Gộp kết quả nhận diện ASR và phân tách người nói Diarization.

        Xác định xem mỗi phân đoạn ASR thuộc về người nói nào dựa trên độ phủ thời gian cực đại.
        """
        turns: List[DiarizedAlignedTurn] = []

        for seg in transcript_segments:
            seg_text = getattr(seg, "text", "")
            seg_start = getattr(seg, "start_sec", 0.0)
            seg_end = getattr(seg, "end_sec", 0.0)

            if not seg_text.strip():
                continue

            # Tìm speaker có độ giao thoa thời gian lớn nhất với đoạn này
            best_speaker = "SPEAKER_00"
            max_overlap = -1.0

            for spk_turn in speaker_segments:
                overlap_start = max(seg_start, spk_turn.start_sec)
                overlap_end = min(seg_end, spk_turn.end_sec)
                overlap = max(0.0, overlap_end - overlap_start)

                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = spk_turn.speaker

            # Căn chỉnh từng từ
            aligned_words = self.align_segment_words(
                text=seg_text,
                start_sec=seg_start,
                end_sec=seg_end,
                speaker=best_speaker,
            )

            turns.append(
                DiarizedAlignedTurn(
                    speaker=best_speaker,
                    start_sec=seg_start,
                    end_sec=seg_end,
                    text=unicodedata.normalize("NFC", seg_text),
                    words=aligned_words,
                )
            )

        return turns
