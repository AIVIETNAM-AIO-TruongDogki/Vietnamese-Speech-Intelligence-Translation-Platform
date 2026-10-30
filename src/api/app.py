"""Module API: Cung cấp RESTful API hiệu năng cao xây dựng bằng FastAPI.
Cung cấp các cổng giao tiếp chuẩn hóa cho hệ thống:
- POST /api/v1/transcribe: Gỡ âm nhanh ASR + Tinh chỉnh VPA.
- POST /api/v1/process: Xử lý trọn gói End-to-End (Audio -> Subtitle SRT/VTT + Báo cáo + Dịch thuật).
- GET /api/v1/health: Kiểm tra tình trạng sức khỏe hệ thống và GPU VRAM.
- GET /api/v1/download/{filename}: Tải tệp phụ đề hoặc báo cáo Markdown.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
except ImportError:
    FastAPI = None

from ..workflow.graph import VietSpeechPipeline


def create_app(
    asr_model_id: str = "vinai/PhoWhisper-medium",
    translation_model: str = "Helsinki-NLP/opus-mt-vi-en",
    output_dir: str = "./outputs",
):
    """Khởi tạo FastAPI Application."""
    if FastAPI is None:
        raise ImportError("FastAPI chưa được cài đặt. Vui lòng cài đặt: pip install fastapi uvicorn python-multipart")

    app = FastAPI(
        title="VietSpeech AI Platform API",
        description="Nền tảng Trí tuệ Giọng nói & Dịch thuật Đa ngữ Tiếng Việt 3 Miền (Vietnamese Speech Intelligence & Translation)",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # Khởi tạo Pipeline một lần duy nhất trong vòng đời API
    pipeline = VietSpeechPipeline(
        asr_model_id=asr_model_id,
        translation_model=translation_model,
    )

    @app.get("/api/v1/health")
    async def health_check():
        import torch
        gpu_info = "Không có GPU (CPU Mode)"
        if torch.cuda.is_available():
            vram_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            gpu_info = f"{torch.cuda.get_device_name(0)} (VRAM: {vram_alloc:.1f}MB / {vram_total:.1f}MB)"

        return {
            "status": "healthy",
            "service": "VietSpeech AI Platform",
            "gpu": gpu_info,
            "asr_model": asr_model_id,
            "translation_model": translation_model,
        }

    @app.post("/api/v1/process")
    async def process_audio(
        file: UploadFile = File(...),
        enable_translation: bool = Form(True),
        enable_diarization: bool = Form(True),
    ):
        """Xử lý toàn diện file âm thanh tải lên."""
        suffix = Path(file.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name

        try:
            state = pipeline.process(tmp_path, output_dir=out_path)
            return {
                "filename": file.filename,
                "duration_seconds": state.get("duration_sec", 0.0),
                "quality_status": state.get("quality_status", "PASSED"),
                "raw_transcript": state.get("raw_transcript", ""),
                "refined_transcript": state.get("refined_transcript", ""),
                "turns_count": len(state.get("diarized_turns", [])),
                "intelligence_summary": getattr(state.get("intelligence_report"), "executive_summary", ""),
                "action_items": getattr(state.get("intelligence_report"), "action_items", []),
                "downloads": {
                    k: f"/api/v1/download/{Path(v).name}" for k, v in state.get("final_exports", {}).items()
                },
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lỗi xử lý âm thanh: {str(e)}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @app.get("/api/v1/download/{filename}")
    async def download_file(filename: str):
        target = out_path / filename
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Không tìm thấy tệp yêu cầu.")
        return FileResponse(path=str(target), filename=filename)

    return app
