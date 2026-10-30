"""Module ASR Service: Nhận diện giọng nói tiếng Việt 3 miền (Bắc - Trung - Nam).
Kế thừa trọn vẹn và mở rộng kiến trúc từ Zalo_AI:
1. Tự động phát hiện PEFT LoRA Adapter và thực hiện merge_and_unload() vào Base Model FP16.
2. Dọn sạch rác VRAM (gc.collect + empty_cache) để tối ưu cho GPU Laptop RTX 4060 8GB.
3. Fallback mượt mà sang faster-whisper (CTranslate2) hoặc transformers pipeline.
4. Chuẩn hóa nghiêm ngặt Unicode NFC.
5. Hỗ trợ xuất chi tiết mốc thời gian (timestamps) cho từng phân đoạn câu.
"""
from __future__ import annotations

import gc
import json
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class ASRSegmentResult:
    """Kết quả nhận diện một phân đoạn câu kèm mốc thời gian."""
    text: str
    start_sec: float
    end_sec: float
    confidence: float = 1.0


@dataclass
class ASRTranscriptionResult:
    """Kết quả nhận diện toàn bộ file âm thanh."""
    full_text: str
    segments: List[ASRSegmentResult]
    duration_sec: float
    language: str = "vi"


class ASRService:
    """Dịch vụ nhận diện tiếng Việt sử dụng PhoWhisper."""

    def __init__(
        self,
        model_id: str = "vinai/PhoWhisper-medium",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        beam_size: int = 1,
        temperature: float = 0.0,
    ):
        """Khởi tạo mô hình ASR.

        Args:
            model_id: Tên model HF hoặc đường dẫn LoRA Adapter cục bộ (vd: phowhisper_lora_final).
            device: 'cuda' hoặc 'cpu' (tự động nhận diện nếu None).
            compute_type: 'float16', 'int8', 'float32'.
            beam_size: Mặc định 1 (Greedy search để đảm bảo tính tất định và tốc độ cao).
            temperature: 0.0 chống hiện tượng hallucination.
        """
        import torch

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if compute_type is None:
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = compute_type

        self.model_id = str(model_id)
        self.beam_size = beam_size
        self.temperature = temperature
        self._model = None
        self._processor = None
        self._backend = "faster_whisper"
        self._init_model()

    def _init_model(self):
        """Khởi tạo mô hình theo cơ chế thông minh từ Zalo_AI."""
        import torch

        # 1. Kiểm tra nếu model_id là LoRA Adapter cục bộ hoặc có adapter_config.json
        is_peft = False
        base_model_name = "vinai/PhoWhisper-medium"
        adapter_cfg_path = os.path.join(self.model_id, "adapter_config.json")

        if os.path.isdir(self.model_id) and os.path.exists(adapter_cfg_path):
            is_peft = True
            try:
                with open(adapter_cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                base_model_name = cfg.get("base_model_name_or_path", "vinai/PhoWhisper-medium")
            except Exception:
                pass
        elif "/" in self.model_id and not os.path.exists(self.model_id):
            try:
                from peft import PeftConfig
                p_cfg = PeftConfig.from_pretrained(self.model_id)
                is_peft = True
                base_model_name = p_cfg.base_model_name_or_path or "vinai/PhoWhisper-medium"
            except Exception:
                is_peft = False

        if is_peft:
            print(f"[ASRService] Phát hiện LoRA Adapter tại '{self.model_id}'. Base Model: '{base_model_name}'.")
            try:
                from peft import PeftModel
                from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

                dtype = torch.float16 if self.device == "cuda" else torch.float32
                print(f"[ASRService] Đang nạp Base Model '{base_model_name}' trên {self.device} ({dtype})...")
                base_model = WhisperForConditionalGeneration.from_pretrained(
                    base_model_name,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
                print(f"[ASRService] Đang hợp nhất trọng số LoRA từ '{self.model_id}' (merge_and_unload)...")
                peft_model = PeftModel.from_pretrained(base_model, self.model_id)
                merged_model = peft_model.merge_and_unload()
                merged_model.eval()
                if self.device == "cuda":
                    merged_model = merged_model.to("cuda")

                # Giải phóng tensor trung gian để bảo toàn VRAM GPU Laptop 8GB
                del peft_model
                del base_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                try:
                    processor = WhisperProcessor.from_pretrained(self.model_id)
                except Exception:
                    processor = WhisperProcessor.from_pretrained(base_model_name)

                self._processor = processor
                self._model = pipeline(
                    "automatic-speech-recognition",
                    model=merged_model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    torch_dtype=dtype,
                    return_timestamps=True,
                )
                self._backend = "transformers"
                print(f"[ASRService] Hợp nhất LoRA PhoWhisper thành công trên {self.device}!")
                return
            except Exception as peft_err:
                print(f"[ASRService] Cảnh báo lỗi nạp LoRA ({peft_err}). Thử nạp bằng faster_whisper...")

        # 2. Thử nạp bằng faster_whisper (CTranslate2)
        try:
            from faster_whisper import WhisperModel
            print(f"[ASRService] Đang nạp faster-whisper '{self.model_id}' trên {self.device} ({self.compute_type})...")
            self._model = WhisperModel(
                self.model_id,
                device=self.device,
                compute_type=self.compute_type,
            )
            self._backend = "faster_whisper"
            print("[ASRService] Đã sẵn sàng faster-whisper!")
            return
        except Exception as e_fw:
            print(f"[ASRService] faster-whisper không thể nạp ({e_fw}). Fallback sang transformers pipeline...")

        # 3. Fallback sang transformers tiêu chuẩn
        from transformers import pipeline
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._model = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=0 if self.device == "cuda" else -1,
            torch_dtype=dtype,
            return_timestamps=True,
        )
        self._backend = "transformers"
        print("[ASRService] Đã nạp transformers ASR pipeline!")

    def transcribe(
        self,
        audio_input: Union[str, Path, np.ndarray],
        sample_rate: int = 16000,
    ) -> ASRTranscriptionResult:
        """Nhận diện file âm thanh hoặc mảng numpy, trả về kết quả kèm timestamps và Unicode NFC.

        Args:
            audio_input: Đường dẫn file (.wav, .mp3) hoặc mảng numpy float32 16kHz.
            sample_rate: Tần số lấy mẫu (mặc định 16000).

        Returns:
            ASRTranscriptionResult: Bản gỡ âm đầy đủ và các phân đoạn có timestamp.
        """
        import soundfile as sf
        from ..audio.loader import AudioLoader

        if isinstance(audio_input, (str, Path)):
            audio_data, sr, duration = AudioLoader.load_audio(audio_input, target_sr=16000)
        elif isinstance(audio_input, np.ndarray):
            audio_data = audio_input
            sr = sample_rate
            duration = len(audio_data) / float(sr)
        else:
            raise ValueError(f"Kiểu đầu vào không hợp lệ: {type(audio_input)}")

        segments_res: List[ASRSegmentResult] = []

        if self._backend == "faster_whisper":
            fw_segments, info = self._model.transcribe(
                audio_data,
                language="vi",
                beam_size=self.beam_size,
                temperature=self.temperature,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400),
            )
            full_texts = []
            for seg in fw_segments:
                seg_text = unicodedata.normalize("NFC", seg.text).strip()
                if seg_text:
                    full_texts.append(seg_text)
                    segments_res.append(
                        ASRSegmentResult(
                            text=seg_text,
                            start_sec=float(seg.start),
                            end_sec=float(seg.end),
                            confidence=float(np.exp(seg.avg_logprob)) if hasattr(seg, "avg_logprob") else 1.0,
                        )
                    )
            full_text = " ".join(full_texts)
        else:
            # Backend transformers
            import torch
            with torch.inference_mode():
                out = self._model(
                    {"raw": audio_data, "sampling_rate": sr},
                    generate_kwargs={"language": "vi", "task": "transcribe"},
                )
            raw_text = out.get("text", "")
            full_text = unicodedata.normalize("NFC", raw_text).strip()

            chunks = out.get("chunks", [])
            if chunks:
                for c in chunks:
                    c_text = unicodedata.normalize("NFC", c.get("text", "")).strip()
                    ts = c.get("timestamp", (0.0, duration))
                    s_sec = ts[0] if ts[0] is not None else 0.0
                    e_sec = ts[1] if ts[1] is not None else duration
                    if c_text:
                        segments_res.append(
                            ASRSegmentResult(
                                text=c_text,
                                start_sec=float(s_sec),
                                end_sec=float(e_sec),
                                confidence=1.0,
                            )
                        )
            else:
                segments_res.append(
                    ASRSegmentResult(
                        text=full_text,
                        start_sec=0.0,
                        end_sec=duration,
                        confidence=1.0,
                    )
                )

        return ASRTranscriptionResult(
            full_text=full_text,
            segments=segments_res,
            duration_sec=duration,
            language="vi",
        )
