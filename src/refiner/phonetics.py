"""Module Phonetics: Phân tích âm vị học và đo lường khoảng cách phát âm tiếng Việt.
Kế thừa nguyên bản và nâng cấp từ Zalo_AI:
Cung cấp ma trận phân rã âm tiết tiếng Việt thành (Onset, Nucleus, Coda, Tone)
và tính toán độ tương đồng âm học (Acoustic-Phonetic Similarity) giữa 2 âm tiết.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

# Bảng dấu thanh tiếng Việt
# 0: ngang, 1: huyền, 2: sắc, 3: hỏi, 4: ngã, 5: nặng
TONE_MAP: Dict[str, Tuple[str, int]] = {
    "a": ("a", 0), "à": ("a", 1), "á": ("a", 2), "ả": ("a", 3), "ã": ("a", 4), "ạ": ("a", 5),
    "ă": ("ă", 0), "ằ": ("ă", 1), "ắ": ("ă", 2), "ẳ": ("ă", 3), "ẵ": ("ă", 4), "ặ": ("ă", 5),
    "â": ("â", 0), "ầ": ("â", 1), "ấ": ("â", 2), "ẩ": ("â", 3), "ẫ": ("â", 4), "ậ": ("â", 5),
    "e": ("e", 0), "è": ("e", 1), "é": ("e", 2), "ẻ": ("e", 3), "ẽ": ("e", 4), "ẹ": ("e", 5),
    "ê": ("ê", 0), "ề": ("ê", 1), "ế": ("ê", 2), "ể": ("ê", 3), "ễ": ("ê", 4), "ệ": ("ê", 5),
    "i": ("i", 0), "ì": ("i", 1), "í": ("i", 2), "ỉ": ("i", 3), "ĩ": ("i", 4), "ị": ("i", 5),
    "o": ("o", 0), "ò": ("o", 1), "ó": ("o", 2), "ỏ": ("o", 3), "õ": ("o", 4), "ọ": ("o", 5),
    "ô": ("ô", 0), "ồ": ("ô", 1), "ố": ("ô", 2), "ổ": ("ô", 3), "ỗ": ("ô", 4), "ộ": ("ô", 5),
    "ơ": ("ơ", 0), "ờ": ("ơ", 1), "ớ": ("ơ", 2), "ở": ("ơ", 3), "ỡ": ("ơ", 4), "ợ": ("ơ", 5),
    "u": ("u", 0), "ù": ("u", 1), "ú": ("u", 2), "ủ": ("u", 3), "ũ": ("u", 4), "ụ": ("u", 5),
    "ư": ("ư", 0), "ừ": ("ư", 1), "ứ": ("ư", 2), "ử": ("ư", 3), "ữ": ("ư", 4), "ự": ("ư", 5),
    "y": ("y", 0), "ỳ": ("y", 1), "ý": ("y", 2), "ỷ": ("y", 3), "ỹ": ("y", 4), "ỵ": ("y", 5),
}

# Danh sách phụ âm đầu sắp xếp theo độ dài giảm dần
ONSET_CONSONANTS = [
    "ngh", "ng", "nh", "th", "ch", "tr", "kh", "ph", "gh", "gi", "qu",
    "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "x"
]

# Danh sách phụ âm cuối sắp xếp theo độ dài giảm dần
CODA_CONSONANTS = ["ng", "nh", "ch", "c", "m", "n", "p", "t", "i", "y", "u", "o"]

# Nhóm phụ âm cận âm (Onset Confusion Groups theo phương ngữ 3 miền)
ONSET_CONFUSION_GROUPS = [
    {"d", "đ"},
    {"s", "x"},
    {"tr", "ch"},
    {"l", "n"},
    {"r", "d", "gi"},
    {"c", "k", "q"},
    {"g", "gh"},
    {"ng", "ngh"},
    {"v", "b"},
]

# Nhóm nguyên âm biến dạng do nuốt âm hoặc ngữ âm miền Trung / miền Nam
NUCLEUS_CONFUSION_GROUPS = [
    {"ưa", "ươ", "ơi", "ơ", "â", "u", "ư", "o"},  # mưa <-> mơi <-> mất <-> mốt
    {"uô", "ua", "o", "ô", "u"},
    {"iê", "ia", "e", "ê", "i", "y"},
    {"ă", "a", "â"},
    {"ơ", "ô", "u"},
]


def extract_tone(syllable: str) -> Tuple[str, int]:
    """Tách thanh điệu và trả về (chuỗi không dấu thanh, mã thanh 0..5)."""
    syllable = unicodedata.normalize("NFC", syllable.lower())
    tone = 0
    clean_chars = []
    for char in syllable:
        if char in TONE_MAP:
            base, t = TONE_MAP[char]
            clean_chars.append(base)
            if t > 0:
                tone = t
        else:
            clean_chars.append(char)
    return "".join(clean_chars), tone


def decompose_syllable(syllable: str) -> Tuple[str, str, str, int]:
    """Phân rã âm tiết tiếng Việt thành: (Onset, Nucleus, Coda, Tone)."""
    raw = unicodedata.normalize("NFC", syllable.lower().strip())
    raw = re.sub(r"^[^\w\s]+|[^\w\s]+$", "", raw)
    if not raw:
        return ("", "", "", 0)

    clean_text, tone = extract_tone(raw)

    # 1. Tách Onset (phụ âm đầu)
    onset = ""
    remainder = clean_text
    for c in ONSET_CONSONANTS:
        if remainder.startswith(c):
            onset = c
            remainder = remainder[len(c):]
            break

    # 2. Tách Coda (phụ âm cuối)
    coda = ""
    if len(remainder) > 1:
        for c in CODA_CONSONANTS:
            if remainder.endswith(c):
                if len(remainder) > len(c):
                    coda = c
                    remainder = remainder[:-len(c)]
                    break

    # 3. Phần còn lại là Nucleus (nguyên âm chính)
    nucleus = remainder
    return (onset, nucleus, coda, tone)


def onset_distance(o1: str, o2: str) -> float:
    """Tính khoảng cách giữa 2 phụ âm đầu [0.0 - 1.0]."""
    if o1 == o2:
        return 0.0
    for group in ONSET_CONFUSION_GROUPS:
        if o1 in group and o2 in group:
            return 0.25  # Cận âm (ví dụ d <-> đ, tr <-> ch)
    return 1.0


def nucleus_distance(n1: str, n2: str) -> float:
    """Tính khoảng cách giữa 2 âm chính [0.0 - 1.0]."""
    if n1 == n2:
        return 0.0
    for group in NUCLEUS_CONFUSION_GROUPS:
        if n1 in group and n2 in group:
            return 0.25  # Biến dạng do nuốt âm (ví dụ ưa <-> ơi/ất)
    return 1.0


def coda_distance(c1: str, c2: str) -> float:
    """Tính khoảng cách giữa 2 phụ âm cuối [0.0 - 1.0]."""
    if c1 == c2:
        return 0.0
    if {c1, c2} in [{"c", "ch"}, {"n", "ng"}, {"t", "c"}, {"m", "n"}, {"i", "y"}]:
        return 0.25
    if (c1 in {"i", "u", "o"} and c2 == "") or (c2 in {"i", "u", "o"} and c1 == ""):
        return 0.3
    return 1.0


def tone_distance(t1: int, t2: int) -> float:
    """Tính khoảng cách giữa 2 thanh điệu [0.0 - 1.0]."""
    if t1 == t2:
        return 0.0
    # Hỏi (3) và Ngã (4) hay bị gộp ở miền Nam và Trung
    if {t1, t2} == {3, 4}:
        return 0.2
    # Sắc (2) và Nặng (5) ở một số giọng miền Trung (Hà Tĩnh, Quảng Bình)
    if {t1, t2} in [{2, 5}, {0, 1}]:
        return 0.3
    return 0.7


def phonetic_distance(w1: str, w2: str) -> float:
    """Tính khoảng cách âm vị học tổng hợp giữa 2 âm tiết tiếng Việt [0.0 - 1.0]."""
    o1, n1, c1, t1 = decompose_syllable(w1)
    o2, n2, c2, t2 = decompose_syllable(w2)

    # Trọng số: Onset (35%), Nucleus (35%), Coda (15%), Tone (15%)
    d_onset = onset_distance(o1, o2)
    d_nucleus = nucleus_distance(n1, n2)
    d_coda = coda_distance(c1, c2)
    d_tone = tone_distance(t1, t2)

    total_dist = (0.35 * d_onset) + (0.35 * d_nucleus) + (0.15 * d_coda) + (0.15 * d_tone)
    return min(max(total_dist, 0.0), 1.0)


def phonetic_similarity(w1: str, w2: str) -> float:
    """Tính độ tương đồng âm vị học [0.0 - 1.0]."""
    if not w1 or not w2:
        return 0.0
    if w1.lower() == w2.lower():
        return 1.0
    return 1.0 - phonetic_distance(w1, w2)
