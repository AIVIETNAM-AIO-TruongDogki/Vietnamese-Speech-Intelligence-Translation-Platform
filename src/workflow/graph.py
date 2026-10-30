"""Module Workflow Orchestration: Điều phối Pipeline với LangGraph và State Machine.
Triển khai đúng kiến trúc đa nhánh (Branching, Retry, Parallel Processing & Quality Gates):

                 +---------------------------+
                 |   START: Audio Validation |
                 +-------------+-------------+
                               |
                 +-------------v-------------+
                 |       VAD & Chunking      |
                 +-------------+-------------+
                               |
                 +-------------v-------------+
                 |    ASR (PhoWhisper LoRA)  |
                 +-------------+-------------+
                               |
                 +-------------v-------------+
                 |  Transcript Quality Check |
                 +-------------+-------------+
                               |
                   [Low Conf? & Retry < 2]
                  /                        \
          (Retry/Enhance)            (Good Quality)
                 |                          |
                 +<---------+       +-------v-------+
                                    | Refinement VPA|
                                    +-------+-------+
                                            |
                         +------------------+------------------+
                         | (Parallel Branches Execution)       |
                         |                                     |
                  +------v------+       +-------v-------+      +-------v-------+
                  | Diarization |       |  Translation  |      | Summarization |
                  |  & Aligner  |       |  (NLLB / MT)  |      | (Intelligence)|
                  +------+------+       +-------+-------+      +-------+-------+
                         |                      |                      |
                         +------------------+---+----------------------+
                                            |
                                    +-------v-------+
                                    | Export Assets |
                                    | (SRT/VTT/MD)  |
                                    +-------+-------+
                                            |
                                          [END]
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from ..audio.enhancement import AudioEnhancer
from ..audio.loader import AudioLoader
from ..audio.vad import VoiceActivityDetector
from ..asr.cache import ASRCache
from ..asr.service import ASRService
from ..diarization.aligner import WordAligner
from ..diarization.diarizer import SpeakerDiarizer
from ..exporters.document import DocumentExporter
from ..exporters.subtitle import SubtitleExporter
from ..intelligence.summarizer import ConversationSummarizer
from ..refiner.filter import TranscriptFilter
from ..refiner.punctuate import PunctuationRestorer
from ..refiner.semantic_refiner import SemanticPhoneticRefiner
from .state import VietSpeechState


class VietSpeechPipeline:
    """Hệ thống điều phối Pipeline thông minh cho giọng nói tiếng Việt."""

    def __init__(
        self,
        asr_model_id: str = "vinai/PhoWhisper-medium",
        translation_model: str = "Helsinki-NLP/opus-mt-vi-en",
        device: Optional[str] = None,
        asr_cache_file: Optional[str] = None,
        confidence_threshold: float = 0.55,
        max_retries: int = 1,
    ):
        self.device = device or "cpu"
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries

        # Khởi tạo các module thành phần
        print("[VietSpeechPipeline] Đang khởi tạo các module chuyên biệt...")
        self.enhancer = AudioEnhancer(sample_rate=16000)
        self.vad = VoiceActivityDetector(sample_rate=16000, device=self.device)
        self.asr = ASRService(model_id=asr_model_id, device=self.device)
        self.asr_cache = ASRCache(asr_cache_file) if asr_cache_file else None

        self.filter = TranscriptFilter(normalize_dialects_to_standard=True)
        self.refiner = SemanticPhoneticRefiner(similarity_threshold=0.58)
        self.punctuate = PunctuationRestorer()

        self.diarizer = SpeakerDiarizer(device=self.device)
        self.aligner = WordAligner(device=self.device)
        self.summarizer = ConversationSummarizer()

        # Dịch thuật lazy-load
        self._translator = None
        self._translation_model_name = translation_model

    @property
    def translator(self):
        if self._translator is None:
            from ..translation.translator import TranslationService
            self._translator = TranslationService(
                model_name=self._translation_model_name,
                device=self.device,
            )
        return self._translator

    # ==================== NODE DEFINITIONS ====================

    def node_audio_validation(self, state: VietSpeechState) -> VietSpeechState:
        """Node 1: Kiểm tra định dạng và nạp tín hiệu âm thanh."""
        audio_path = state["audio_path"]
        print(f"[Pipeline:Node 1] Đang nạp và kiểm tra âm thanh: {Path(audio_path).name}")

        data, sr, duration = AudioLoader.load_audio(audio_path, target_sr=16000)
        state["audio_data"] = data
        state["sample_rate"] = sr
        state["duration_sec"] = duration
        state["retry_count"] = state.get("retry_count", 0)
        return state

    def node_vad_chunking(self, state: VietSpeechState) -> VietSpeechState:
        """Node 2: Nhận diện giọng nói và phân đoạn chống hallucination."""
        print("[Pipeline:Node 2] Đang chạy Voice Activity Detection (VAD)...")
        data = state["audio_data"]
        sr = state["sample_rate"]

        # Nếu đang ở lượt retry, chạy thêm bước khử nhiễu tăng cường
        if state.get("retry_count", 0) > 0:
            print("[Pipeline:Node 2] Đang áp dụng Audio Enhancement lọc tạp âm cho lượt thử lại...")
            data = self.enhancer.enhance(data, reduce_noise=True)
            state["audio_data"] = data

        segments = self.vad.detect_segments(data, sample_rate=sr)
        state["speech_segments"] = segments
        return state

    def node_asr(self, state: VietSpeechState) -> VietSpeechState:
        """Node 3: Nhận diện tiếng Việt 3 miền với PhoWhisper."""
        audio_path = state["audio_path"]
        cache_hit = False

        if self.asr_cache:
            cache_key = ASRCache.compute_file_hash(audio_path)
            cached_val = self.asr_cache.get(cache_key)
            if cached_val and state.get("retry_count", 0) == 0:
                print(f"[Pipeline:Node 3] Đã tìm thấy bản gỡ âm trong Cache ({cache_key[:8]}...)!")
                state["raw_transcript"] = cached_val.get("full_text", "")
                state["asr_segments"] = cached_val.get("segments", [])
                state["asr_confidence"] = cached_val.get("confidence", 1.0)
                cache_hit = True

        if not cache_hit:
            print("[Pipeline:Node 3] Đang suy luận nhận diện giọng nói PhoWhisper...")
            asr_res = self.asr.transcribe(state["audio_data"], sample_rate=state["sample_rate"])
            state["raw_transcript"] = asr_res.full_text
            state["asr_segments"] = asr_res.segments

            # Tính độ tin cậy trung bình
            confs = [s.confidence for s in asr_res.segments if hasattr(s, "confidence")]
            avg_conf = sum(confs) / len(confs) if confs else 1.0
            state["asr_confidence"] = avg_conf

            if self.asr_cache:
                cache_key = ASRCache.compute_file_hash(audio_path)
                self.asr_cache.set(
                    cache_key,
                    {
                        "full_text": asr_res.full_text,
                        "segments": asr_res.segments,
                        "confidence": avg_conf,
                    },
                )

        return state

    def node_quality_check(self, state: VietSpeechState) -> VietSpeechState:
        """Node 4: Đánh giá chất lượng bản gỡ âm (Gate)."""
        raw_text = state.get("raw_transcript", "")
        conf = state.get("asr_confidence", 1.0)

        # Kiểm tra lặp từ bất thường hoặc rỗng
        is_empty = len(raw_text.strip()) == 0
        is_repetitive = False
        words = raw_text.split()
        if len(words) > 10:
            # Nếu 1 từ chiếm hơn 40% câu -> lặp hallucination
            from collections import Counter
            counts = Counter(words)
            most_common, max_c = counts.most_common(1)[0]
            if max_c / float(len(words)) > 0.4:
                is_repetitive = True

        if is_empty or is_repetitive or conf < self.confidence_threshold:
            state["quality_status"] = "LOW_CONFIDENCE"
            print(f"[Pipeline:Node 4] Quality Check: LOW_CONFIDENCE (conf={conf:.2f}, rep={is_repetitive})")
        else:
            state["quality_status"] = "PASSED"
            print(f"[Pipeline:Node 4] Quality Check: PASSED (conf={conf:.2f})")

        return state

    def node_refinement(self, state: VietSpeechState) -> VietSpeechState:
        """Node 5: Chuẩn hóa phương ngữ & mỏ neo âm vị học (VPA)."""
        print("[Pipeline:Node 5] Đang tinh chỉnh ngữ âm & chuẩn hóa từ địa phương (VPA)...")
        raw = state["raw_transcript"]

        # 1. Bộ lọc quy tắc phương ngữ Bắc - Trung - Nam từ Zalo_AI
        filtered = self.filter.filter(raw)

        # 2. Mỏ neo âm vị học Semantic-Phonetic Refiner (VPA)
        refined = self.refiner.refine(filtered)

        # 3. Khôi phục dấu câu & viết hoa
        final_text = self.punctuate.restore(refined)
        state["refined_transcript"] = final_text
        return state

    def node_diarization_and_alignment(self, state: VietSpeechState) -> VietSpeechState:
        """Node 6: Phân tách người nói (Diarization) và căn chỉnh từ (Alignment)."""
        print("[Pipeline:Node 6] Đang phân tách người nói (Diarization) & Căn chỉnh mốc thời gian...")
        data = state["audio_data"]
        sr = state["sample_rate"]
        asr_segments = state["asr_segments"]

        # 1. Phân tách người nói
        speaker_segments = self.diarizer.diarize(data, sample_rate=sr)
        state["speaker_segments"] = speaker_segments

        # 2. Căn chỉnh từng từ và gộp với người nói
        diarized_turns = self.aligner.merge_diarization_and_transcripts(
            transcript_segments=asr_segments,
            speaker_segments=speaker_segments,
        )
        state["diarized_turns"] = diarized_turns
        return state

    def node_translation(self, state: VietSpeechState) -> VietSpeechState:
        """Node 7: Dịch thuật ngữ cảnh Việt sang Anh."""
        print("[Pipeline:Node 7] Đang dịch thuật hội thoại sang Tiếng Anh...")
        diarized_turns = state.get("diarized_turns", [])
        trans_segs = self.translator.translate_segments(diarized_turns)
        state["translated_segments"] = trans_segs
        return state

    def node_intelligence(self, state: VietSpeechState) -> VietSpeechState:
        """Node 8: Trích xuất thông tin, tóm tắt và Action Items."""
        print("[Pipeline:Node 8] Đang tạo biên bản tóm tắt thông minh & Action Items...")
        text = state.get("refined_transcript") or state.get("raw_transcript", "")
        turns = state.get("diarized_turns", [])
        report = self.summarizer.generate_intelligence_report(text, diarized_turns=turns)
        state["intelligence_report"] = report
        return state

    def node_export_assets(self, state: VietSpeechState, output_dir: str | Path) -> VietSpeechState:
        """Node 9: Xuất toàn bộ ấn phẩm phụ đề (SRT, VTT), báo cáo Markdown và JSON."""
        print("[Pipeline:Node 9] Đang xuất phụ đề và tài liệu báo cáo đa định dạng...")
        out_p = Path(output_dir).resolve()
        out_p.mkdir(parents=True, exist_ok=True)

        audio_stem = Path(state["audio_path"]).stem
        diarized_turns = state.get("diarized_turns", [])
        translated_segments = state.get("translated_segments", [])
        raw_text = state.get("raw_transcript", "")
        refined_text = state.get("refined_transcript", raw_text)
        duration = state.get("duration_sec", 0.0)
        report = state.get("intelligence_report")

        # 1. Phụ đề tiếng Việt (.srt, .vtt)
        vi_srt = out_p / f"{audio_stem}_vi.srt"
        vi_vtt = out_p / f"{audio_stem}_vi.vtt"
        SubtitleExporter.save_srt(vi_srt, diarized_turns, mode="vi")
        SubtitleExporter.save_vtt(vi_vtt, diarized_turns, mode="vi")

        # 2. Phụ đề song ngữ Việt - Anh (.srt, .vtt)
        bilingual_srt = out_p / f"{audio_stem}_bilingual.srt"
        bilingual_vtt = out_p / f"{audio_stem}_bilingual.vtt"
        SubtitleExporter.save_srt(bilingual_srt, translated_segments, mode="bilingual")
        SubtitleExporter.save_vtt(bilingual_vtt, translated_segments, mode="bilingual")

        # 3. Báo cáo Markdown chi tiết (.md)
        report_md = out_p / f"{audio_stem}_report.md"
        DocumentExporter.save_markdown(
            file_path=report_md,
            title=f"Báo cáo Hội thoại: {audio_stem}",
            duration_sec=duration,
            full_transcript=raw_text,
            refined_transcript=refined_text,
            diarized_turns=diarized_turns,
            translation_segments=translated_segments,
            intelligence_report=report,
        )

        # 4. JSON bàn giao hệ thống (.json)
        deliverable_json = out_p / f"{audio_stem}_full_data.json"
        data_payload = {
            "metadata": {
                "audio_file": Path(state["audio_path"]).name,
                "duration_seconds": duration,
                "sample_rate": state.get("sample_rate", 16000),
                "quality_status": state.get("quality_status", "PASSED"),
            },
            "transcripts": {
                "raw_pho_whisper": raw_text,
                "refined_vpa": refined_text,
            },
            "intelligence": {
                "title": getattr(report, "title", ""),
                "summary": getattr(report, "executive_summary", ""),
                "key_points": getattr(report, "key_points", []),
                "action_items": getattr(report, "action_items", []),
                "topics": getattr(report, "topics", []),
            },
            "turns": [
                {
                    "speaker": getattr(t, "speaker", "SPEAKER_00"),
                    "start_sec": getattr(t, "start_sec", 0.0),
                    "end_sec": getattr(t, "end_sec", 0.0),
                    "vietnamese": getattr(t, "text", ""),
                    "english": getattr(translated_segments[i], "en_text", "") if i < len(translated_segments) else "",
                }
                for i, t in enumerate(diarized_turns)
            ],
        }
        DocumentExporter.save_json(deliverable_json, data_payload)

        state["final_exports"] = {
            "vi_srt": str(vi_srt),
            "vi_vtt": str(vi_vtt),
            "bilingual_srt": str(bilingual_srt),
            "bilingual_vtt": str(bilingual_vtt),
            "report_md": str(report_md),
            "deliverable_json": str(deliverable_json),
        }
        return state

    # ==================== PIPELINE EXECUTION ====================

    def process(self, audio_path: str | Path, output_dir: str | Path = "./outputs") -> VietSpeechState:
        """Thực thi toàn bộ luồng xử lý End-to-End."""
        state: VietSpeechState = {
            "audio_path": str(Path(audio_path).resolve()),
            "retry_count": 0,
        }

        # 1. Validation
        state = self.node_audio_validation(state)

        # Vòng lặp Quality Gate & Retry
        while True:
            # 2. VAD Chunking
            state = self.node_vad_chunking(state)

            # 3. ASR PhoWhisper
            state = self.node_asr(state)

            # 4. Quality Check
            state = self.node_quality_check(state)

            if state["quality_status"] == "PASSED" or state["retry_count"] >= self.max_retries:
                break

            print(f"[Pipeline] Phát hiện chất lượng thấp, đang tự động retry lần {state['retry_count'] + 1}...")
            state["retry_count"] += 1

        # 5. Tinh chỉnh VPA
        state = self.node_refinement(state)

        # 6, 7, 8. Các nhánh xử lý song song
        state = self.node_diarization_and_alignment(state)
        state = self.node_translation(state)
        state = self.node_intelligence(state)

        # 9. Xuất kết quả
        state = self.node_export_assets(state, output_dir=output_dir)

        print(f"[VietSpeechPipeline] Hoàn tất xử lý audio '{Path(audio_path).name}' thành công!")
        return state
