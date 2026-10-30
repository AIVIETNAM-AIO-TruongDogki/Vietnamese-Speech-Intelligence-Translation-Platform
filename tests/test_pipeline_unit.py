"""Bộ kiểm thử đơn vị (Unit Tests) cho toàn bộ các module của VietSpeech AI."""
import numpy as np
import pytest

from src.audio.loader import AudioLoader
from src.audio.vad import VoiceActivityDetector
from src.evaluation.metrics import compute_cer, compute_rtf, compute_wer
from src.exporters.document import DocumentExporter
from src.exporters.subtitle import SubtitleExporter, format_timestamp_srt, format_timestamp_vtt
from src.refiner.filter import TranscriptFilter
from src.refiner.phonetics import (
    decompose_syllable,
    extract_tone,
    phonetic_distance,
    phonetic_similarity,
)
from src.refiner.punctuate import PunctuationRestorer
from src.refiner.semantic_refiner import SemanticPhoneticRefiner


def test_vietnamese_phonetics():
    # 1. Tách thanh điệu
    clean_w, tone = extract_tone("nguyễn")
    assert clean_w == "nguyen"
    assert tone == 4  # Ngã

    # 2. Phân rã âm tiết: Onset, Nucleus, Coda, Tone
    onset, nucleus, coda, tone = decompose_syllable("tiếng")
    assert onset == "t"
    assert nucleus == "iê"
    assert coda == "ng"
    assert tone == 2  # Sắc

    # 3. Độ tương đồng âm học giữa các cặp cận âm từ Zalo_AI
    # mưa <-> mơi (biến dạng nuốt âm của miền Trung)
    sim_mua_moi = phonetic_similarity("mưa", "mơi")
    assert sim_mua_moi >= 0.55

    # dời <-> đời (cận âm d - đ)
    sim_doi = phonetic_similarity("dời", "đời")
    assert sim_doi >= 0.70

    # Hai từ giống nhau tuyệt đối
    assert phonetic_similarity("Hà Nội", "Hà Nội") == 1.0


def test_transcript_filter_rules():
    filter_engine = TranscriptFilter(normalize_dialects_to_standard=True)

    # Test các mẫu lẫn âm thời tiết từ Zalo_AI
    raw1 = "nếu mai mơi thì đời chạy bộ một hai tiếng"
    cleaned1 = filter_engine.filter(raw1)
    assert "nếu mai mưa" in cleaned1
    assert "thì dời" in cleaned1

    # Test khẩu lệnh công nghệ
    raw2 = "nhắn hẹp ms cho anh nam báo thức bảy giờ"
    cleaned2 = filter_engine.filter(raw2)
    assert "nhắn tin SMS" in cleaned2

    # Test phương ngữ miền Trung
    raw3 = "bữa ni đi mô mần chi rứa"
    cleaned3 = filter_engine.filter(raw3)
    assert "hôm nay đi đâu làm gì thế" in cleaned3


def test_semantic_phonetic_refiner_protection():
    refiner = SemanticPhoneticRefiner(similarity_threshold=0.58)

    # Đảm bảo con số và thời gian được bảo vệ tuyệt đối
    raw = "đặt báo thức lúc bảy giờ ba mươi"
    cand = "đặt báo thức lúc tám giờ bốn mươi"
    aligned = refiner.local_phonetic_align(raw, cand)
    # Vì bảy, ba mươi là số nên bắt buộc giữ nguyên bảy, ba mươi
    assert "bảy" in aligned
    assert "ba mươi" in aligned


def test_punctuation_restorer():
    restorer = PunctuationRestorer()
    res1 = restorer.restore("hôm nay thời tiết thế nào hả")
    assert res1.endswith("?")
    assert res1[0].isupper()

    res2 = restorer.restore("tôi muốn chuyển tiền cho anh nam")
    assert res2.endswith(".")
    assert res2[0].isupper()


def test_evaluation_metrics():
    ref = "nếu mai mưa thì dời chạy bộ"
    hyp_perfect = "Nếu mai mưa thì dời chạy bộ"
    hyp_wrong = "nếu mai mơi thì đời chạy bộ"

    assert compute_wer(ref, hyp_perfect) == 0.0
    assert compute_cer(ref, hyp_perfect) == 0.0

    wer_err = compute_wer(ref, hyp_wrong)
    assert wer_err > 0.0

    rtf = compute_rtf(processing_time_s=1.0, audio_duration_s=4.0)
    assert rtf == 0.25


def test_subtitle_exporters():
    class DummyTurn:
        start_sec = 1.25
        end_sec = 4.75
        speaker = "SPEAKER_01"
        text = "Xin chào Việt Nam"
        vi_text = "Xin chào Việt Nam"
        en_text = "Hello Vietnam"

    turns = [DummyTurn()]
    srt_out = SubtitleExporter.to_srt(turns, mode="bilingual")
    assert "00:00:01,250 --> 00:00:04,750" in srt_out
    assert "Xin chào Việt Nam" in srt_out
    assert "Hello Vietnam" in srt_out

    vtt_out = SubtitleExporter.to_vtt(turns, mode="vi")
    assert "WEBVTT" in vtt_out
    assert "00:00:01.250 --> 00:00:04.750" in vtt_out


def test_audio_loader_synthetic():
    # Tạo tín hiệu sine 16kHz giả lập
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine_wave = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    data, out_sr, out_dur = AudioLoader.load_audio(sine_wave, sample_rate=sr)
    assert out_sr == 16000
    assert abs(out_dur - 1.0) < 0.01
    assert data.ndim == 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
