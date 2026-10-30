"""Module Filter: Chuẩn hóa và làm sạch văn bản ASR trước khi đưa vào dịch thuật hoặc LLM.
Kế thừa & mở rộng toàn diện từ Zalo_AI:
1. Xử lý các hiện tượng nuốt âm, lẫn âm phụ âm đầu (d/đ, tr/ch, s/x, r/d/gi).
2. Chuẩn hóa hệ thống từ vựng phương ngữ 3 miền (Bắc - Trung - Nam).
3. Đảm bảo chuẩn hóa tuyệt đối Unicode NFC.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple


class TranscriptFilter:
    """Bộ lọc xử lý hậu kỳ (Post-ASR Filter) cho giọng nói tiếng Việt 3 miền."""

    def __init__(self, normalize_dialects_to_standard: bool = True):
        self.normalize_dialects = normalize_dialects_to_standard

        # 1. Các mẫu lẫn âm điều kiện thời tiết & hành động (Kế thừa từ Zalo_AI)
        self._weather_cond_patterns: List[Tuple[str, str]] = [
            (r"\b(?:nếu|nhiều)\s+(?:mai|máy)\s+(?:mơi|mất|mốt|mớ|mới)\b", "nếu mai mưa"),
            (r"\b(mai\s+mà)\s+(?:mơi|mất|mốt|mớ|mới)\b", r"\1 mưa"),
            (r"\b(không)\s+(?:mơi|mất|mệt|mớ|mới)(\s+thì\s+thôi)\b", r"\1 mưa\2"),
        ]

        # 2. Các mẫu lẫn âm hành động lịch trình (Kế thừa từ Zalo_AI)
        self._action_patterns: List[Tuple[str, str]] = [
            (r"\b(thì\s+)đời(\s+(?:chạy|lịch|sang|lại|buổi|họp|báo))\b", r"\1dời\2"),
            (r"\b(nếu\s+mai\s+mưa\s+thì)\s+đời\b", r"\1 dời"),
            (r"\b(thì\s+)đời\b", r"\1dời"),
            (r"\bchạy\s+bốn\b", "chạy bộ"),
            (r"\bgiữ\s+nghiêng\b", "giữ nguyên"),
        ]

        # 3. Lẫn âm từ mượn và từ viết tắt công nghệ
        self._tech_and_msg_patterns: List[Tuple[str, str]] = [
            (r"\bnhắn\s+(?:hẹp|sẵn|kịp)\s*(?:ms|em\s*ấy)\b", "nhắn tin SMS"),
            (r"\b(?:hẹp|sẵn)\s*ms\b", "tin nhắn SMS"),
            (r"\b(gửi\s+)em\s*s\b", r"\1SMS"),
            (r"\bem\s*tin\b", "nhắn tin"),
            (r"\be\s+xin\s+cho\s+mình\b", "nhắn tin cho mình"),
            (r"\bda\s*lô\b", "Zalo"),
            (r"\bgúc\s*gồ\b", "Google"),
        ]

        # 4. Các cụm từ khẩu lệnh thông dụng hay bị lẫn âm
        self._common_voice_patterns: List[Tuple[str, str]] = [
            (r"\bbáo\s+thích\b", "báo thức"),
            (r"\bđặt\s+(?:một\s+)?bé\b", "đặt vé"),
            (r"\bnửa\s+chín\b", "nửa tiếng"),
            (r"\bhạ\s+(?:canh|cánh)\s+sông\b", "hạ cánh xong"),
            (r"\bchuyện\s+giúp\s+tôi\b", "chuyển giúp tôi"),
            (r"\bchuyện\s+tiền\b", "chuyển tiền"),
        ]

        # 5. Hệ thống từ vựng phương ngữ miền Trung & miền Nam sang tiếng Việt chuẩn
        # Giúp mô hình Translation dịch chuẩn xác ngữ nghĩa sang tiếng Anh
        self._dialect_standard_patterns: List[Tuple[str, str]] = [
            # Miền Trung
            (r"\bở\s+mô\b", "ở đâu"),
            (r"\bđi\s+mô\b", "đi đâu"),
            (r"\bvề\s+mô\b", "về đâu"),
            (r"\bchi\s+rứa\b", "gì thế"),
            (r"\brăng\s+rứa\b", "sao thế"),
            (r"\blàm\s+chi\b", "làm gì"),
            (r"\bcái\s+chi\b", "cái gì"),
            (r"\bnhìn\s+tê\b", "nhìn kìa"),
            (r"\bngó\s+tê\b", "nhìn kìa"),
            (r"\bđằng\s+nớ\b", "đằng đó"),
            (r"\bbữa\s+ni\b", "hôm nay"),
            (r"\bbữa\s+nớ\b", "hôm đó"),
            (r"\bmần\s+răng\b", "làm sao"),
            (r"\bmần\s+chi\b", "làm gì"),
            (r"\bưng\s+bụng\b", "hài lòng"),
            # Miền Nam
            (r"\bở\s+trển\b", "ở trên đó"),
            (r"\bở\s+trỏng\b", "ở trong đó"),
            (r"\bdưới\s+bển\b", "dưới bên đó"),
            (r"\bbữa\s+hổm\b", "hôm trước"),
            (r"\bhổng\s+có\b", "không có"),
            (r"\bhổng\s+biết\b", "không biết"),
            (r"\bthiệt\s+hông\b", "thật không"),
            (r"\bthiệt\s+hả\b", "thật hả"),
            (r"\bnghen\b", "nhé"),
        ]

    def filter(self, text: str) -> str:
        """Lọc và chuẩn hóa văn bản từ ASR.

        Args:
            text: Chuỗi văn bản thô từ ASR.

        Returns:
            Văn bản đã chuẩn hóa phương ngữ, ngữ âm và Unicode NFC.
        """
        if not text:
            return ""

        # Luôn chuẩn hóa Unicode NFC trước tiên
        filtered = unicodedata.normalize("NFC", text).strip()

        # Áp dụng các mẫu quy tắc từ Zalo_AI
        for pattern, repl in self._weather_cond_patterns:
            filtered = re.sub(pattern, repl, filtered, flags=re.IGNORECASE)

        for pattern, repl in self._action_patterns:
            filtered = re.sub(pattern, repl, filtered, flags=re.IGNORECASE)

        for pattern, repl in self._tech_and_msg_patterns:
            filtered = re.sub(pattern, repl, filtered, flags=re.IGNORECASE)

        for pattern, repl in self._common_voice_patterns:
            filtered = re.sub(pattern, repl, filtered, flags=re.IGNORECASE)

        # Chuẩn hóa phương ngữ nếu bật
        if self.normalize_dialects:
            for pattern, repl in self._dialect_standard_patterns:
                filtered = re.sub(pattern, repl, filtered, flags=re.IGNORECASE)

        # Xóa khoảng trắng thừa
        filtered = re.sub(r"\s+", " ", filtered).strip()
        return unicodedata.normalize("NFC", filtered)
