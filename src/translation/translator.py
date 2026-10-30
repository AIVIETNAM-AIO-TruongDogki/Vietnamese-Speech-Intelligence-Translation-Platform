"""Module Translation: Dịch thuật ngữ cảnh từ tiếng Việt sang tiếng Anh (Vietnamese -> English).
Hỗ trợ đa dạng backend:
- HuggingFace NLLB-200 (facebook/nllb-200-distilled-600M) tối ưu cho dịch đa ngữ.
- Helsinki-NLP/opus-mt-vi-en siêu nhẹ, tốc độ cao.
- Multilingual LLM (Qwen2.5-3B-Instruct) dịch hiểu ngữ cảnh hội thoại, tiếng lóng, thành ngữ.
- Rule-based dictionary fallback khi chạy thử nghiệm offline hoàn toàn.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any, List, Optional, Union


@dataclass
class TranslatedSegment:
    """Đoạn dịch thuật kèm mốc thời gian và văn bản gốc."""
    vi_text: str
    en_text: str
    start_sec: float
    end_sec: float
    speaker: str = "SPEAKER_00"


class TranslationService:
    """Dịch vụ dịch thuật Việt - Anh."""

    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-vi-en",
        backend: str = "transformers",
        device: str = "cpu",
        llm_engine: Optional[Any] = None,
    ):
        """Khởi tạo dịch vụ dịch thuật.

        Args:
            model_name: Tên mô hình ('Helsinki-NLP/opus-mt-vi-en' hoặc 'facebook/nllb-200-distilled-600M').
            backend: 'transformers', 'llm', hoặc 'fallback'.
            device: 'cuda' hoặc 'cpu'.
            llm_engine: Thể hiện của Agent LLM nếu backend='llm'.
        """
        self.model_name = model_name
        self.backend = backend
        self.device = device
        self.llm_engine = llm_engine
        self._pipeline = None
        self._tokenizer = None
        self._model = None
        self._init_backend()

    def _init_backend(self):
        """Nạp mô hình dịch thuật."""
        if self.backend == "llm" and self.llm_engine:
            return

        if self.backend == "transformers":
            try:
                from transformers import pipeline
                print(f"[Translation] Đang nạp mô hình dịch '{self.model_name}' trên {self.device}...")
                self._pipeline = pipeline(
                    "translation",
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1,
                )
                print("[Translation] Mô hình dịch thuật đã sẵn sàng!")
            except Exception as e:
                print(f"[Translation] Không thể nạp transformers pipeline ({e}). Sử dụng chế độ fallback từ điển.")
                self.backend = "fallback"

    def translate_text(self, vi_text: str) -> str:
        """Dịch một câu văn tiếng Việt sang tiếng Anh."""
        if not vi_text or not vi_text.strip():
            return ""

        vi_clean = unicodedata.normalize("NFC", vi_text).strip()

        # 1. Nếu dùng LLM Engine
        if self.backend == "llm" and self.llm_engine:
            return self._translate_with_llm(vi_clean)

        # 2. Nếu dùng Transformers pipeline
        if self.backend == "transformers" and self._pipeline:
            try:
                res = self._pipeline(vi_clean, max_length=256)
                if res and isinstance(res, list):
                    return res[0].get("translation_text", "").strip()
            except Exception:
                pass

        # 3. Fallback chế độ từ điển thông minh
        return self._fallback_translate(vi_clean)

    def _translate_with_llm(self, text: str) -> str:
        """Dịch văn bản bằng LLM với prompt tối ưu cho tiếng Việt hội thoại."""
        system_prompt = (
            "You are an expert Vietnamese-to-English translator specializing in spoken conversations, "
            "regional dialects, and colloquial idioms.\n"
            "Translate the following Vietnamese text into natural, fluent English.\n"
            "Output ONLY the English translation without any notes or explanations."
        )
        try:
            tokenizer = getattr(self.llm_engine, "_tokenizer", None) or getattr(self.llm_engine, "tokenizer", None)
            model = getattr(self.llm_engine, "_model", None) or getattr(self.llm_engine, "model", None)
            device = getattr(self.llm_engine, "device", "cuda")

            if tokenizer and model:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Vietnamese: {text}\nEnglish:"},
                ]
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([prompt_text], return_tensors="pt").to(device)

                import torch
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
                gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
                result = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                return result
        except Exception:
            pass
        return self._fallback_translate(text)

    def _fallback_translate(self, text: str) -> str:
        """Từ điển tra cứu nhanh cho các câu khẩu lệnh mẫu."""
        text_lower = text.lower()
        phrase_dict = {
            "nếu mai mưa thì dời chạy bộ một hai tiếng không mưa thì thôi": "If it rains tomorrow, postpone running for one or two hours; if not, keep it.",
            "đặt báo thức trước giúp tôi bảy giờ": "Please set an alarm for me at 7:00 AM.",
            "chuyển giúp tôi một triệu rưỡi cho anh nam": "Please transfer 1.5 million VND to Mr. Nam for me.",
            "xin chào các bạn": "Hello everyone.",
            "hôm nay thời tiết thế nào": "How is the weather today?",
            "hôm nay trời đẹp quá": "The weather is so nice today.",
            "bạn có khỏe không": "How are you?",
            "cảm ơn rất nhiều": "Thank you very much.",
            "tạm biệt và hẹn gặp lại": "Goodbye and see you again.",
        }
        for k, v in phrase_dict.items():
            if k in text_lower:
                return v

        return f"[EN Translation of: {text}]"

    def translate_segments(self, turns: List[Any]) -> List[TranslatedSegment]:
        """Dịch toàn bộ danh sách các phân đoạn hội thoại."""
        results: List[TranslatedSegment] = []
        for turn in turns:
            vi_text = getattr(turn, "text", "")
            s_sec = getattr(turn, "start_sec", 0.0)
            e_sec = getattr(turn, "end_sec", 0.0)
            spk = getattr(turn, "speaker", "SPEAKER_00")

            en_text = self.translate_text(vi_text)
            results.append(
                TranslatedSegment(
                    vi_text=vi_text,
                    en_text=en_text,
                    start_sec=s_sec,
                    end_sec=e_sec,
                    speaker=spk,
                )
            )
        return results
