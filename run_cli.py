"""Giao diện dòng lệnh (CLI): Chạy phân tích âm thanh End-to-End với VietSpeech AI.
Sử dụng:
    python run_cli.py --audio sample.wav --output-dir ./outputs
    python run_cli.py --audio sample.wav --asr-model ../Zalo_AI/phowhisper_lora_final
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Đảm bảo UTF-8 cho console Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.workflow.graph import VietSpeechPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="VietSpeech AI: Vietnamese Speech Intelligence & Translation Platform CLI"
    )
    parser.add_argument(
        "--audio",
        "-a",
        type=str,
        required=True,
        help="Đường dẫn file âm thanh đầu vào (WAV, MP3, MP4, FLAC, M4A).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./outputs",
        help="Thư mục xuất các tệp phụ đề (SRT, VTT), báo cáo Markdown và JSON.",
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default="vinai/PhoWhisper-medium",
        help="Tên Hugging Face model hoặc thư mục chứa checkpoint LoRA adapter.",
    )
    parser.add_argument(
        "--translation-model",
        type=str,
        default="Helsinki-NLP/opus-mt-vi-en",
        help="Tên mô hình dịch thuật Việt - Anh.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Thiết bị tính toán: 'cuda' hoặc 'cpu' (mặc định tự động nhận diện).",
    )
    parser.add_argument(
        "--asr-cache",
        type=str,
        default="./asr_cache.json",
        help="Đường dẫn file lưu bộ nhớ đệm ASR để tái sử dụng.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    audio_path = Path(args.audio).resolve()
    if not audio_path.is_file():
        print(f"[LỖI] Không tìm thấy file âm thanh: {audio_path}")
        sys.exit(1)

    print("=" * 70)
    print("  VIETSPEECH AI: VIETNAMESE SPEECH INTELLIGENCE & TRANSLATION PLATFORM")
    print("=" * 70)
    print(f"[*] File âm thanh:       {audio_path.name}")
    print(f"[*] Thư mục đầu ra:      {args.output_dir}")
    print(f"[*] Mô hình ASR:         {args.asr_model}")
    print(f"[*] Mô hình dịch thuật:  {args.translation_model}")
    print("-" * 70)

    start_time = time.time()

    pipeline = VietSpeechPipeline(
        asr_model_id=args.asr_model,
        translation_model=args.translation_model,
        device=args.device,
        asr_cache_file=args.asr_cache,
    )

    state = pipeline.process(audio_path, output_dir=args.output_dir)
    total_elapsed = time.time() - start_time

    duration = state.get("duration_sec", 0.0)
    rtf = round(total_elapsed / duration, 3) if duration > 0 else 0.0

    print("\n" + "=" * 70)
    print("               KẾT QUẢ XỬ LÝ HỘI THOẠI")
    print("=" * 70)
    print(f"• Thời lượng âm thanh:       {duration:.2f}s")
    print(f"• Tổng thời gian xử lý:     {total_elapsed:.2f}s (RTF: {rtf})")
    print(f"• Bản gỡ âm PhoWhisper:     {state.get('raw_transcript', '')}")
    print(f"• Bản chuẩn hóa VPA:        {state.get('refined_transcript', '')}")
    print("-" * 70)

    report = state.get("intelligence_report")
    if report:
        print(f"• Tóm tắt điều hành:        {getattr(report, 'executive_summary', '')}")
        action_items = getattr(report, "action_items", [])
        if action_items:
            print("• Danh sách công việc (Action Items):")
            for item in action_items:
                print(f"    [ ] {item}")

    print("\n• Các tệp đã xuất bản:")
    for k, v in state.get("final_exports", {}).items():
        print(f"    - {k:15s}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()
