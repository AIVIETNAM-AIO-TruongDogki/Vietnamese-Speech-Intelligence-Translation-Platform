"""Module Evaluation: Đo lường chất lượng ASR (WER, CER), Dịch (BLEU), Tốc độ (RTF, Latency)."""
from .metrics import (
    compute_wer,
    compute_cer,
    compute_rtf,
    compute_simple_bleu,
    compute_levenshtein_distance,
    BenchmarkResult,
)

__all__ = [
    "compute_wer",
    "compute_cer",
    "compute_rtf",
    "compute_simple_bleu",
    "compute_levenshtein_distance",
    "BenchmarkResult",
]
