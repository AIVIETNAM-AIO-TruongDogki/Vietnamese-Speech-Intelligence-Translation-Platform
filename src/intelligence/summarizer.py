"""Module Intelligence & Summarization: Trích xuất thông tin thông minh từ bản gỡ âm hội thoại.
Bao gồm:
1. Executive Summary: Tóm tắt ngắn gọn nội dung cốt lõi.
2. Key Discussion Points: Các ý chính được thảo luận.
3. Action Items / Next Steps: Danh sách đầu việc cần làm kèm người phụ trách (nếu có).
4. Conversation Sentiment & Topic Classification.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IntelligenceReport:
    """Báo cáo trí tuệ nhân tạo tổng hợp từ âm thanh hội thoại."""
    title: str
    executive_summary: str
    key_points: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    tone: str = "Thảo luận mang tính xây dựng"


class ConversationSummarizer:
    """Bộ tóm tắt và trích xuất tri thức hội thoại."""

    def __init__(self, llm_engine: Optional[Any] = None):
        self.llm_engine = llm_engine

    def generate_intelligence_report(
        self,
        full_transcript: str,
        diarized_turns: Optional[List[Any]] = None,
    ) -> IntelligenceReport:
        """Tạo báo cáo tóm tắt thông minh từ hội thoại tiếng Việt."""
        clean_text = unicodedata.normalize("NFC", full_transcript).strip()
        if not clean_text:
            return IntelligenceReport(
                title="Hội thoại trống",
                executive_summary="Không có âm thanh giọng nói để tóm tắt.",
            )

        # 1. Nếu có LLM Engine
        if self.llm_engine:
            return self._summarize_with_llm(clean_text, diarized_turns)

        # 2. Heuristic Extractive Summarization Fallback
        return self._heuristic_summarize(clean_text, diarized_turns)

    def _summarize_with_llm(
        self,
        transcript: str,
        turns: Optional[List[Any]],
    ) -> IntelligenceReport:
        """Tóm tắt chuyên sâu sử dụng LLM."""
        formatted_dialogue = ""
        if turns:
            lines = [f"{getattr(t, 'speaker', 'Speaker')}: {getattr(t, 'text', '')}" for t in turns]
            formatted_dialogue = "\n".join(lines)
        else:
            formatted_dialogue = transcript

        system_prompt = (
            "Bạn là trợ lý trí tuệ nhân tạo chuyên phân tích và tóm tắt biên bản cuộc họp, hội thoại tiếng Việt.\n"
            "Hãy đọc đoạn hội thoại sau và xuất ra bản tóm tắt theo định dạng JSON với các trường:\n"
            "- 'title': Tiêu đề ngắn của hội thoại\n"
            "- 'summary': Tóm tắt ngắn gọn 2-3 câu\n"
            "- 'key_points': Danh sách các luận điểm chính (list of strings)\n"
            "- 'action_items': Danh sách hành động/công việc cần làm (list of strings)\n"
            "- 'topics': Chủ đề chính\n"
            "Chỉ xuất JSON hợp lệ, không giải thích thêm."
        )

        try:
            tokenizer = getattr(self.llm_engine, "_tokenizer", None) or getattr(self.llm_engine, "tokenizer", None)
            model = getattr(self.llm_engine, "_model", None) or getattr(self.llm_engine, "model", None)
            device = getattr(self.llm_engine, "device", "cuda")

            if tokenizer and model:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Hội thoại:\n{formatted_dialogue}\n\nJSON Output:"},
                ]
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([prompt_text], return_tensors="pt").to(device)

                import torch
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
                raw_json = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

                import json
                clean_json_str = re.search(r"\{.*\}", raw_json, re.DOTALL)
                if clean_json_str:
                    data = json.loads(clean_json_str.group(0))
                    return IntelligenceReport(
                        title=data.get("title", "Biên bản cuộc trao đổi"),
                        executive_summary=data.get("summary", ""),
                        key_points=data.get("key_points", []),
                        action_items=data.get("action_items", []),
                        topics=data.get("topics", ["Công việc", "Kế hoạch"]),
                    )
        except Exception:
            pass

        return self._heuristic_summarize(transcript, turns)

    def _heuristic_summarize(
        self,
        transcript: str,
        turns: Optional[List[Any]],
    ) -> IntelligenceReport:
        """Thuật toán trích xuất heuristic offline."""
        sentences = [s.strip() for s in re.split(r"[.!?]\s*", transcript) if len(s.strip()) > 5]

        # Trích xuất action items dựa trên từ khóa hành động tiếng Việt
        action_keywords = ["cần", "phải", "sẽ", "nhờ", "giúp", "dời", "đặt", "chuyển", "gửi", "thông báo", "họp"]
        action_items = []
        for s in sentences:
            if any(re.search(rf"\b{kw}\b", s, re.IGNORECASE) for kw in action_keywords):
                action_items.append(s)

        # Lấy 2 câu đầu làm tóm tắt ngắn
        summary = " ".join(sentences[:2]) if sentences else transcript

        return IntelligenceReport(
            title="Biên bản tóm tắt âm thanh tiếng Việt",
            executive_summary=summary,
            key_points=sentences[:4],
            action_items=action_items[:3],
            topics=["Xử lý khẩu lệnh", "Lịch trình & Kế hoạch"],
            tone="Khách quan, hành động",
        )
