"""Module Subtitle Exporter: Xuất phụ đề chuẩn SRT (SubRip) và WebVTT (.vtt).
Hỗ trợ:
- Phụ đề tiếng Việt gốc.
- Phụ đề tiếng Anh dịch thuật.
- Phụ đề song ngữ (Bilingual: Tiếng Việt trên, Tiếng Anh dưới).
- Gắn thẻ người nói [Speaker].
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union


def format_timestamp_srt(seconds: float) -> str:
    """Định dạng thời gian theo chuẩn SRT: HH:MM:SS,mmm."""
    seconds = max(0.0, seconds)
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 999
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Định dạng thời gian theo chuẩn WebVTT: HH:MM:SS.mmm."""
    seconds = max(0.0, seconds)
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 999
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


class SubtitleExporter:
    """Bộ tạo và xuất phụ đề chuyên nghiệp."""

    @classmethod
    def to_srt(
        cls,
        segments: List[Any],
        include_speaker: bool = True,
        mode: str = "vi",  # "vi", "en", "bilingual"
    ) -> str:
        """Tạo chuỗi nội dung file phụ đề định dạng .srt."""
        lines = []
        cue_idx = 1

        for seg in segments:
            s_sec = getattr(seg, "start_sec", 0.0)
            e_sec = getattr(seg, "end_sec", 0.0)
            spk = getattr(seg, "speaker", "SPEAKER_00")
            vi_text = getattr(seg, "vi_text", None) or getattr(seg, "text", "")
            en_text = getattr(seg, "en_text", "")

            # Bỏ qua các đoạn không có text
            if not vi_text and not en_text:
                continue

            spk_prefix = f"[{spk}] " if include_speaker and spk else ""

            if mode == "vi":
                caption = f"{spk_prefix}{vi_text}"
            elif mode == "en":
                caption = f"{spk_prefix}{en_text}" if en_text else f"{spk_prefix}{vi_text}"
            elif mode == "bilingual":
                caption = f"{spk_prefix}{vi_text}\n{en_text}" if en_text else f"{spk_prefix}{vi_text}"
            else:
                caption = f"{spk_prefix}{vi_text}"

            lines.append(str(cue_idx))
            lines.append(f"{format_timestamp_srt(s_sec)} --> {format_timestamp_srt(e_sec)}")
            lines.append(caption)
            lines.append("")  # Dòng trống phân cách
            cue_idx += 1

        return "\n".join(lines)

    @classmethod
    def to_vtt(
        cls,
        segments: List[Any],
        include_speaker: bool = True,
        mode: str = "vi",
    ) -> str:
        """Tạo chuỗi nội dung file phụ đề định dạng WebVTT (.vtt)."""
        lines = ["WEBVTT", ""]
        cue_idx = 1

        for seg in segments:
            s_sec = getattr(seg, "start_sec", 0.0)
            e_sec = getattr(seg, "end_sec", 0.0)
            spk = getattr(seg, "speaker", "SPEAKER_00")
            vi_text = getattr(seg, "vi_text", None) or getattr(seg, "text", "")
            en_text = getattr(seg, "en_text", "")

            if not vi_text and not en_text:
                continue

            spk_prefix = f"<v {spk}>" if include_speaker and spk else ""
            spk_suffix = "</v>" if include_speaker and spk else ""

            if mode == "vi":
                caption = f"{spk_prefix}{vi_text}{spk_suffix}"
            elif mode == "en":
                caption = f"{spk_prefix}{en_text or vi_text}{spk_suffix}"
            elif mode == "bilingual":
                caption = f"{spk_prefix}{vi_text}{spk_suffix}\n{en_text}" if en_text else f"{spk_prefix}{vi_text}{spk_suffix}"
            else:
                caption = f"{spk_prefix}{vi_text}{spk_suffix}"

            lines.append(str(cue_idx))
            lines.append(f"{format_timestamp_vtt(s_sec)} --> {format_timestamp_vtt(e_sec)}")
            lines.append(caption)
            lines.append("")
            cue_idx += 1

        return "\n".join(lines)

    @classmethod
    def save_srt(cls, file_path: Union[str, Path], segments: List[Any], mode: str = "vi") -> None:
        """Lưu ra file .srt trên đĩa."""
        content = cls.to_srt(segments, mode=mode)
        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    @classmethod
    def save_vtt(cls, file_path: Union[str, Path], segments: List[Any], mode: str = "vi") -> None:
        """Lưu ra file .vtt trên đĩa."""
        content = cls.to_vtt(segments, mode=mode)
        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
