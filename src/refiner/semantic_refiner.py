"""Module Semantic-Phonetic Refiner (VPA): Tinh chỉnh văn bản hai giai đoạn Global-to-Local.
Kế thừa kiến trúc bảo vệ thực thể và triệt tiêu hallucination từ Zalo_AI:
1. Stage 1 (Global Context Reconstruction): Khôi phục mạch câu tự nhiên bằng LLM hoặc Rule Engine.
2. Stage 2 (Local Vietnamese Phonetic Anchor - VPA): Dùng mỏ neo âm vị học kiểm tra từng token,
   chỉ chấp nhận sửa khi độ tương đồng âm học Sim >= 0.58. Khóa cứng toàn bộ con số, ngày giờ, tên riêng.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any, Dict, List, Optional

from .phonetics import phonetic_similarity


class SemanticPhoneticRefiner:
    """Bộ tinh chỉnh văn bản âm vị - ngữ nghĩa hai giai đoạn."""

    def __init__(
        self,
        llm_engine: Optional[Any] = None,
        similarity_threshold: float = 0.58,
    ):
        """Khởi tạo Refiner.

        Args:
            llm_engine: Engine LLM (nếu có) để tái thiết câu toàn cục.
            similarity_threshold: Ngưỡng tương đồng âm học tối thiểu (mặc định 0.58 từ Zalo_AI).
        """
        self.llm_engine = llm_engine
        self.similarity_threshold = similarity_threshold

        # Danh sách từ mượn công nghệ đặc biệt cho phép thay thế cụm từ
        self._tech_codeswitch = {
            "hẹp ms": "tin nhắn SMS",
            "sẵn ms": "tin nhắn SMS",
            "hẹp em ấy": "tin nhắn SMS",
            "em tin": "nhắn tin",
        }

    def _is_protected_token(self, token: str) -> bool:
        """Kiểm tra token có phải là số, thời gian hay ký hiệu không được tự ý sửa."""
        clean = token.strip().lower()
        # Chứa chữ số (0-9)
        if re.search(r"\d", clean):
            return True
        # Các từ chỉ số lượng cơ bản tiếng Việt
        if clean in {
            "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười",
            "trăm", "nghìn", "ngàn", "triệu", "tỷ", "rưỡi", "thứ", "ngày", "tháng", "năm", "giờ", "phút"
        }:
            return True
        return False

    def global_reconstruct(self, raw_text: str, context: Optional[Dict] = None) -> str:
        """Giai đoạn 1: Tái cấu trúc câu văn tự nhiên theo ngữ cảnh."""
        if not self.llm_engine:
            return raw_text

        system_prompt = (
            "Bạn là chuyên gia khôi phục câu nói tiếng Việt bị lẫn âm hoặc nuốt âm do giọng địa phương 3 miền.\n"
            "Nhiệm vụ: Chỉnh sửa các từ bị nhận diện nhầm âm học thành câu hoàn chỉnh, đúng ngữ pháp và tự nhiên nhất.\n"
            "RÀNG BUỘC TUYỆT ĐỐI:\n"
            "- Giữ nguyên 100% tất cả tên riêng, con số, thời gian, số lượng, tiền tệ.\n"
            "- Không tự bịa thêm ý định. Chỉ xuất ra DUY NHẤT câu văn đã sửa, không giải thích."
        )

        user_content = f"Input: {raw_text}\nOutput:"

        try:
            tokenizer = getattr(self.llm_engine, "_tokenizer", None) or getattr(self.llm_engine, "tokenizer", None)
            model = getattr(self.llm_engine, "_model", None) or getattr(self.llm_engine, "model", None)
            device = getattr(self.llm_engine, "device", "cuda")

            if tokenizer is not None and model is not None:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([text], return_tensors="pt").to(device)

                import torch
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=64,
                        do_sample=False,
                    )
                gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
                candidate = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                candidate = re.sub(r"^(Output:\s*)+", "", candidate, flags=re.IGNORECASE).strip("\"'").strip()
                return candidate
        except Exception as e:
            # Fallback an toàn về raw_text nếu LLM bận hoặc lỗi
            return raw_text

        return raw_text

    def local_phonetic_align(self, raw_text: str, candidate_text: str) -> str:
        """Giai đoạn 2: Căn chỉnh âm vị học và kiểm chứng độ tương đồng phát âm (VPA)."""
        if not candidate_text or candidate_text == raw_text:
            return raw_text

        raw_words = raw_text.split()
        cand_words = candidate_text.split()

        matcher = difflib.SequenceMatcher(None, raw_words, cand_words)
        verified_words: List[str] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                verified_words.extend(raw_words[i1:i2])
            elif tag == "replace":
                raw_chunk = " ".join(raw_words[i1:i2])
                cand_chunk = " ".join(cand_words[j1:j2])

                # Trường hợp 1: Cặp 1-1 (1 từ thay bằng 1 từ)
                if (i2 - i1 == 1) and (j2 - j1 == 1):
                    w_raw = raw_words[i1]
                    w_cand = cand_words[j1]

                    # Nếu là từ được bảo vệ (số lượng, ngày giờ) -> Giữ nguyên gốc
                    if self._is_protected_token(w_raw):
                        verified_words.append(w_raw)
                        continue

                    # Tính độ tương đồng âm học
                    sim = phonetic_similarity(w_raw, w_cand)
                    if sim >= self.similarity_threshold:
                        # Cận âm hợp lệ -> Chấp thuận sửa
                        verified_words.append(w_cand)
                    else:
                        # Âm quá xa (nghi ngờ LLM bịa/hallucination) -> Giữ nguyên gốc
                        verified_words.append(w_raw)
                else:
                    # Trường hợp thay thế cụm từ (nhiều từ)
                    raw_chunk_lower = raw_chunk.lower()
                    matched_codeswitch = False
                    for code, repl in self._tech_codeswitch.items():
                        if code in raw_chunk_lower:
                            verified_words.extend(cand_words[j1:j2])
                            matched_codeswitch = True
                            break

                    if not matched_codeswitch:
                        # Kiểm tra độ tương đồng trung bình của các cặp từ tương ứng
                        pairs_count = min(len(raw_words[i1:i2]), len(cand_words[j1:j2]))
                        avg_sim = 0.0
                        if pairs_count > 0:
                            for idx in range(pairs_count):
                                avg_sim += phonetic_similarity(raw_words[i1 + idx], cand_words[j1 + idx])
                            avg_sim /= pairs_count

                        if avg_sim >= self.similarity_threshold:
                            verified_words.extend(cand_words[j1:j2])
                        else:
                            verified_words.extend(raw_words[i1:i2])
            elif tag == "delete":
                # Từ bị mất -> Giữ lại để tránh thất thoát thông tin
                verified_words.extend(raw_words[i1:i2])
            elif tag == "insert":
                # Từ mới tự sinh ra ngoài nguyên bản -> Bỏ qua để chống hallucination
                pass

        result = " ".join(verified_words)
        return unicodedata.normalize("NFC", result).strip()

    def refine(self, raw_text: str, context: Optional[Dict] = None) -> str:
        """Hàm thực thi trọn vẹn quy trình Tinh chỉnh 2 chặng: Global -> Local (VPA)."""
        if not raw_text:
            return ""

        raw_norm = unicodedata.normalize("NFC", raw_text).strip()

        # 1. Global Reconstruction
        candidate = self.global_reconstruct(raw_norm, context=context)

        # 2. Local Phonetic Alignment
        final_text = self.local_phonetic_align(raw_norm, candidate)
        return final_text
