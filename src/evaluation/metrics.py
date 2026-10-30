"""Module Evaluation & Metrics: Bộ công cụ đo lường và đánh giá hiệu năng ASR, Translation, Diarization.
Cung cấp các metrics chuẩn mực công nghiệp:
1. WER (Word Error Rate): Tỷ lệ lỗi cấp độ từ theo thuật toán khoảng cách Levenshtein.
2. CER (Character Error Rate): Tỷ lệ lỗi cấp độ ký tự (đặc biệt quan trọng với thanh điệu tiếng Việt).
3. BLEU Score: Đánh giá chất lượng dịch thuật Việt - Anh.
4. RTF (Real-Time Factor): RTF = Processing Time / Audio Duration. (< 1.0 là nhanh hơn thời gian thực).
5. Latency: Độ trễ phản hồi từng câu.
6. DER (Diarization Error Rate): Tỷ lệ gán nhầm người nói.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def _normalize_text_for_eval(text: str) -> str:
    """Chuẩn hóa văn bản trước khi tính WER/CER: bỏ dấu câu, đưa về chữ thường, Unicode NFC."""
    text = unicodedata.normalize("NFC", text.lower())
    # Loại bỏ dấu câu
    text = re.sub(r"[^\w\s]", "", text)
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_levenshtein_distance(seq1: List[str], seq2: List[str]) -> Tuple[int, int, int, int]:
    """Tính khoảng cách Levenshtein và đếm số lượng Substitution, Deletion, Insertion.

    Returns:
        (total_errors, substitutions, deletions, insertions)
    """
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # Deletion
                    dp[i][j - 1] + 1,      # Insertion
                    dp[i - 1][j - 1] + 1,  # Substitution
                )

    # Truy vết tìm cụ thể S, D, I
    i, j = n, m
    s, d, ins = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and seq1[i - 1] == seq2[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            s += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            break

    total = dp[n][m]
    return total, s, d, ins


def compute_wer(reference: str, hypothesis: str) -> float:
    """Tính Word Error Rate (WER) [0.0 - 1.0+].

    WER = (S + D + I) / N
    """
    ref_norm = _normalize_text_for_eval(reference)
    hyp_norm = _normalize_text_for_eval(hypothesis)

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    errors, _, _, _ = compute_levenshtein_distance(ref_words, hyp_words)
    return float(errors) / float(len(ref_words))


def compute_cer(reference: str, hypothesis: str) -> float:
    """Tính Character Error Rate (CER) [0.0 - 1.0+].

    CER = (S + D + I) / N (tính trên từng ký tự).
    """
    ref_norm = _normalize_text_for_eval(reference)
    hyp_norm = _normalize_text_for_eval(hypothesis)

    ref_chars = list(ref_norm.replace(" ", ""))
    hyp_chars = list(hyp_norm.replace(" ", ""))

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    errors, _, _, _ = compute_levenshtein_distance(ref_chars, hyp_chars)
    return float(errors) / float(len(ref_chars))


def compute_rtf(processing_time_s: float, audio_duration_s: float) -> float:
    """Tính Real-Time Factor (RTF).

    RTF = Processing Time / Audio Duration.
    RTF < 1.0 nghĩa là hệ thống xử lý nhanh hơn thời gian thực phát audio.
    """
    if audio_duration_s <= 0:
        return 0.0
    return round(processing_time_s / audio_duration_s, 4)


def compute_simple_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Tính điểm BLEU đơn giản cho bản dịch tiếng Anh (Word N-gram precision + Brevity Penalty)."""
    import math

    ref_words = _normalize_text_for_eval(reference).split()
    hyp_words = _normalize_text_for_eval(hypothesis).split()

    if not ref_words or not hyp_words:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        if len(hyp_words) < n or len(ref_words) < n:
            break
        ref_ngrams = set(tuple(ref_words[i:i+n]) for i in range(len(ref_words) - n + 1))
        hyp_ngrams = [tuple(hyp_words[i:i+n]) for i in range(len(hyp_words) - n + 1)]

        matches = sum(1 for ng in hyp_ngrams if ng in ref_ngrams)
        p_n = matches / float(len(hyp_ngrams)) if hyp_ngrams else 0.0
        if p_n > 0:
            precisions.append(p_n)

    if not precisions:
        return 0.0

    # Brevity Penalty
    bp = 1.0 if len(hyp_words) > len(ref_words) else math.exp(1.0 - float(len(ref_words)) / float(len(hyp_words)))
    geom_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    return round(bp * geom_mean * 100.0, 2)


@dataclass
class BenchmarkResult:
    """Kết quả đo lường cho một tập kiểm thử."""
    experiment_name: str
    dataset_split: str  # 'Bắc', 'Trung', 'Nam'
    total_samples: int
    mean_wer: float
    mean_cer: float
    mean_bleu: float
    mean_rtf: float
    avg_latency_s: float
