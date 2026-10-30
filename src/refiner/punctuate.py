"""Module Punctuation: Khôi phục dấu câu và viết hoa câu tiếng Việt (Punctuation & Capitalization).
Giúp bản gỡ âm tự nhiên, chuẩn văn bản hành chính trước khi đưa vào dịch thuật và xuất phụ đề.
"""
from __future__ import annotations

import re
import unicodedata


class PunctuationRestorer:
    """Bộ khôi phục dấu câu và định dạng viết hoa cho văn bản tiếng Việt."""

    def __init__(self):
        # Các từ kết thúc nghi vấn
        self._question_particles = [
            "hả", "sao", "à", "chăng", "phải không", "đúng không", "được không", "mô", "răng", "chi", "chứ"
        ]

    def restore(self, text: str) -> str:
        """Thêm dấu câu cơ bản, viết hoa đầu câu và dấu chấm/hỏi thích hợp.

        Args:
            text: Chuỗi văn bản tiếng Việt.

        Returns:
            Văn bản đã khôi phục viết hoa và dấu ngắt câu.
        """
        if not text:
            return ""

        text = unicodedata.normalize("NFC", text).strip()

        # 1. Viết hoa ký tự đầu tiên
        if len(text) > 0:
            text = text[0].upper() + text[1:]

        # 2. Viết hoa sau các dấu kết thúc câu (. ! ?)
        def _capitalize_match(match):
            return match.group(1) + match.group(2).upper()

        text = re.sub(r"([.!?]\s+)([a-zà-ỹ])", _capitalize_match, text)

        # 3. Viết hoa các từ xưng hô trang trọng hoặc tên đất nước nếu có
        text = re.sub(r"\bviệt\s+nam\b", "Việt Nam", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhà\s+nội\b", "Hà Nội", text, flags=re.IGNORECASE)
        text = re.sub(r"\bđà\s+nẵng\b", "Đà Nẵng", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhồ\s+chí\s+minh\b", "Hồ Chí Minh", text, flags=re.IGNORECASE)
        text = re.sub(r"\bsài\s+gòn\b", "Sài Gòn", text, flags=re.IGNORECASE)

        # 4. Thêm dấu kết câu nếu chưa có
        if text and text[-1] not in {".", "!", "?", "..."}:
            words = text.lower().split()
            last_word = words[-1] if words else ""
            if any(last_word == q or text.lower().endswith(f" {q}") for q in self._question_particles):
                text += "?"
            else:
                text += "."

        return unicodedata.normalize("NFC", text)
