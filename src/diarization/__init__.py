"""Module Diarization & Alignment: Phân biệt người nói và căn chỉnh mốc thời gian từng từ."""
from .diarizer import SpeakerDiarizer, SpeakerSegment
from .aligner import WordAligner, AlignedWord, DiarizedAlignedTurn

__all__ = ["SpeakerDiarizer", "SpeakerSegment", "WordAligner", "AlignedWord", "DiarizedAlignedTurn"]
