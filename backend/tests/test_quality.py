from quality_metrics import cer, wer, normalize_arabic

def test_arabic_normalization_removes_tashkeel_and_tatweel():
    assert normalize_arabic("السَّلَامُ عَلَيْكُمْ") == "السلام عليكم"
    assert normalize_arabic("مـرحباً") == "مرحبا"

def test_exact_match_scores_zero():
    text = "هذا اختبار للتعرف الضوئي على الحروف العربية"
    assert cer(text, text) == 0
    assert wer(text, text) == 0

def test_single_character_error_is_measured():
    assert cer("كتاب", "كتات") > 0
    assert wer("كتاب جديد", "كتاب قديم") > 0

def test_empty_reference_is_safe():
    assert cer("", "نص") > 0
    assert wer("", "نص") > 0
