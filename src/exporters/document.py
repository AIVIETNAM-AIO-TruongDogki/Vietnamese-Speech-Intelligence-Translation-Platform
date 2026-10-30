"""Module Document Exporter: Xuất báo cáo tổng kết Markdown và định dạng dữ liệu bàn giao JSON.
Hỗ trợ đầy đủ các thành phần:
- Metadata âm thanh (độ dài, số lượng speaker, sample rate).
- Full Transcripts & Refined Transcripts.
- Diarized Turns & Word-level timestamps.
- English Translations (song ngữ).
- Intelligence Summary, Key Points & Action Items.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class DocumentExporter:
    """Bộ xuất tài liệu báo cáo Markdown và JSON."""

    @classmethod
    def to_json(
        cls,
        data: Dict[str, Any],
        indent: int = 2,
    ) -> str:
        """Chuyển đổi dữ liệu pipeline sang chuỗi JSON chuẩn Unicode."""
        return json.dumps(data, ensure_ascii=False, indent=indent)

    @classmethod
    def save_json(cls, file_path: Union[str, Path], data: Dict[str, Any]) -> None:
        """Lưu dữ liệu sang file .json."""
        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def to_markdown(
        cls,
        title: str,
        duration_sec: float,
        full_transcript: str,
        refined_transcript: str,
        diarized_turns: List[Any],
        translation_segments: Optional[List[Any]] = None,
        intelligence_report: Optional[Any] = None,
    ) -> str:
        """Tạo báo cáo Markdown trực quan, chuyên nghiệp."""
        md = []
        md.append(f"# {title}")
        md.append(f"> **Thời lượng:** `{duration_sec:.2f}s` | **Trạng thái:** `Hoàn tất xử lý End-to-End`")
        md.append("")

        # 1. Tóm tắt điều hành & Action Items nếu có
        if intelligence_report:
            md.append("## 1. Trí tuệ Hội thoại & Tóm tắt Điều hành (Intelligence Summary)")
            md.append(f"**Tóm lược:** {getattr(intelligence_report, 'executive_summary', '')}\n")

            key_points = getattr(intelligence_report, "key_points", [])
            if key_points:
                md.append("### Các điểm mấu chốt (Key Points):")
                for kp in key_points:
                    md.append(f"- {kp}")
                md.append("")

            action_items = getattr(intelligence_report, "action_items", [])
            if action_items:
                md.append("### Danh sách công việc cần làm (Action Items):")
                for ai in action_items:
                    md.append(f"- [ ] {ai}")
                md.append("")

        # 2. So sánh Trước - Sau tinh chỉnh ngữ âm (Zalo_AI Heritage)
        md.append("## 2. Bản Gỡ Âm & Tinh Chỉnh Phương Ngữ (VPA Refinement)")
        md.append("| Loại bản gỡ | Nội dung |")
        md.append("| :--- | :--- |")
        md.append(f"| **PhoWhisper ASR Thô** | `{full_transcript}` |")
        md.append(f"| **VPA Refined & Filtered** | `{refined_transcript}` |")
        md.append("")

        # 3. Biên bản chi tiết theo người nói & bản dịch tiếng Anh
        md.append("## 3. Biên bản Chi tiết Phân vai & Dịch thuật (Diarization & Translation)")
        md.append("| Thời gian | Người nói | Tiếng Việt (Chuẩn hóa) | Bản dịch Tiếng Anh (English) |")
        md.append("| :--- | :--- | :--- | :--- |")

        for idx, turn in enumerate(diarized_turns):
            s = getattr(turn, "start_sec", 0.0)
            e = getattr(turn, "end_sec", 0.0)
            spk = getattr(turn, "speaker", "SPEAKER_00")
            vi_text = getattr(turn, "text", "")

            en_text = ""
            if translation_segments and idx < len(translation_segments):
                en_text = getattr(translation_segments[idx], "en_text", "")

            md.append(f"| `{s:.2f}s - {e:.2f}s` | **{spk}** | {vi_text} | *{en_text}* |")

        md.append("")
        md.append("---")
        md.append("*Tạo tự động bởi Vietnamese Speech Intelligence & Translation Platform (VietSpeech AI)*")

        return "\n".join(md)

    @classmethod
    def save_markdown(
        cls,
        file_path: Union[str, Path],
        title: str,
        duration_sec: float,
        full_transcript: str,
        refined_transcript: str,
        diarized_turns: List[Any],
        translation_segments: Optional[List[Any]] = None,
        intelligence_report: Optional[Any] = None,
    ) -> None:
        """Lưu báo cáo ra file .md."""
        content = cls.to_markdown(
            title=title,
            duration_sec=duration_sec,
            full_transcript=full_transcript,
            refined_transcript=refined_transcript,
            diarized_turns=diarized_turns,
            translation_segments=translation_segments,
            intelligence_report=intelligence_report,
        )
        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
