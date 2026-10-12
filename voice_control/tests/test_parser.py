from voice_control.intents import Intent
from voice_control.parser import parse_utterance, response_language


def test_floor_synonyms_are_deterministic():
    for text in ("去三楼", "帮我按三楼", "三楼谢谢"):
        parsed = parse_utterance(text)
        assert parsed.intent is Intent.PRESS_FLOOR
        assert parsed.floor == 3


def test_floor_change_phrases_are_controlled_floor_requests():
    for text in (
        "去四楼",
        "我想去四楼",
        "想去四楼",
        "帮我按四楼",
        "直接去四楼",
        "改成四楼",
        "我想改成四楼",
        "改成直接去四楼",
        "换成四楼",
        "换到四楼",
        "还是去四楼吧",
    ):
        parsed = parse_utterance(text)
        assert parsed.intent is Intent.PRESS_FLOOR, text
        assert parsed.floor == 4, text


def test_cancel_marker_followed_by_explicit_floor_request_replaces_target():
    for text in (
        "算了，我想去四楼",
        "算了我要去4楼",
        "取消，帮我按四楼",
        "不用了，换成四楼",
    ):
        parsed = parse_utterance(text)
        assert parsed.intent is Intent.PRESS_FLOOR, text
        assert parsed.floor == 4, text

    assert parse_utterance("算了").intent is Intent.CANCEL
    assert parse_utterance("算了，确认四楼").intent is Intent.CANCEL
    assert parse_utterance("算了，我不想去四楼").intent is Intent.CANCEL


def test_natural_lead_in_keeps_explicit_floor_action():
    for text in (
        "等下，不对，现在我想去四楼",
        "我现在想去四楼",
        "我其实还是想去四楼",
        "不对，我重新想去四楼",
        "不对，我要去四楼",
        "现在帮我按四楼",
        "算了，还是帮我按四楼吧",
        "我改主意了，去四楼",
    ):
        parsed = parse_utterance(text)
        assert parsed.intent is Intent.PRESS_FLOOR, text
        assert parsed.floor == 4, text


def test_explicit_old_floor_new_floor_correction_uses_new_target():
    for text in (
        "不是三楼，是四楼",
        "不是三楼，我想去四楼",
        "三楼改成四楼",
        "去三楼换成四楼吧",
    ):
        parsed = parse_utterance(text)
        assert parsed.intent is Intent.PRESS_FLOOR, text
        assert parsed.floor == 4, text


def test_flexible_floor_extraction_still_rejects_mentions_and_choices():
    for text in (
        "我朋友想去四楼",
        "我朋友在四楼",
        "三楼还是四楼",
        "不是三楼还是四楼",
    ):
        parsed = parse_utterance(text)
        assert parsed.intent is Intent.UNKNOWN, text


def test_floor_forms_and_rejections():
    assert parse_utterance("去地下二层").floor == -2
    assert parse_utterance("按B1").floor == -1
    assert parse_utterance("去B一层").floor == -1
    assert parse_utterance("按b二楼").floor == -2
    assert parse_utterance("去B三层").reject_reason == "floor_out_of_range"
    assert parse_utterance("去十二楼").floor == 12
    assert parse_utterance("去三楼和四楼").reject_reason == "ambiguous_floor"
    assert parse_utterance("去99楼", floor_max=40).reject_reason == "floor_out_of_range"


def test_floor_context_fails_closed_on_asr_homophone():
    parsed = parse_utterance("去富三楼")
    assert parsed.intent is Intent.UNKNOWN
    assert parsed.floor is None
    assert parsed.reject_reason == "unsafe_floor_context"

    assert parse_utterance("去负三楼").reject_reason == "floor_out_of_range"


def test_repeated_identical_asr_transcript_remains_valid():
    parsed = parse_utterance("去12楼去12楼。")
    assert parsed.intent is Intent.PRESS_FLOOR
    assert parsed.floor == 12


def test_stop_has_priority_over_floor_command():
    parsed = parse_utterance("去三楼，不，停止")
    assert parsed.intent is Intent.STOP


def test_controlled_non_floor_intents():
    assert parse_utterance("确认按三楼").intent is Intent.CONFIRM
    assert parse_utterance("确认").reject_reason == "confirmation_missing_floor"
    assert parse_utterance("算了").intent is Intent.CANCEL
    assert parse_utterance("现在什么状态").intent is Intent.STATUS
    for first in ("小艺", "小易", "小一"):
        for second in ("小艺", "小易", "小一"):
            assert parse_utterance(first + second).intent is Intent.WAKE
    assert parse_utterance("小艺").intent is Intent.UNKNOWN
    assert parse_utterance("小易").intent is Intent.UNKNOWN
    assert parse_utterance("小一").intent is Intent.UNKNOWN
    assert parse_utterance("小艺小易小一").intent is Intent.UNKNOWN
    assert parse_utterance("小创小创").intent is Intent.UNKNOWN
    assert parse_utterance("讲个笑话").intent is Intent.UNKNOWN


def test_floor_mentions_and_negations_never_become_commands():
    for text in ("我住三楼", "三楼在哪里", "我不去三楼", "不要按三楼"):
        parsed = parse_utterance(text)
        assert parsed.intent is Intent.UNKNOWN


def test_observed_english_wake_variants_are_exactly_controlled():
    for text in ("Hey Siri", "嘿 siri.", "嘿sri。", "嘿y, siri.", "嘿y siri."):
        assert parse_utterance(text).intent is Intent.WAKE, text
        assert response_language(text) == "en"
    for text in ("Siri", "Hey series", "Hey city"):
        assert parse_utterance(text).intent is Intent.UNKNOWN, text


def test_english_floor_requests_and_confirmations():
    cases = {
        "Go to floor three": 3,
        "I want to go to the third floor": 3,
        "Please press floor 4": 4,
        "Take me to B1": -1,
        "Go to basement one": -1,
        "Change it to floor four": 4,
        "Actually, I want to go to floor four": 4,
    }
    for text, floor in cases.items():
        parsed = parse_utterance(text)
        assert parsed.intent is Intent.PRESS_FLOOR, text
        assert parsed.floor == floor, text

    confirmed = parse_utterance("Confirm floor four")
    assert confirmed.intent is Intent.CONFIRM
    assert confirmed.floor == 4
    bare_yes = parse_utterance("Yes.")
    assert bare_yes.intent is Intent.CONFIRM
    assert bare_yes.floor is None
    assert parse_utterance("Confirm").reject_reason == "confirmation_missing_floor"
    for text in ("OK", "Okay", "Sure"):
        assert parse_utterance(text).reject_reason == "confirmation_missing_floor"


def test_observed_sensevoice_floor_artifacts_are_narrowly_normalized():
    for text in (
        "f three。",
        "go to4 three.",
        "floor3.",
        "go to floor3.",
        "level3",
        "floor3rd",
    ):
        parsed = parse_utterance(text)
        assert parsed.intent is Intent.PRESS_FLOOR, text
        assert parsed.floor == 3, text

    actual_four = parse_utterance("Go to 4th floor")
    assert actual_four.intent is Intent.PRESS_FLOOR
    assert actual_four.floor == 4


def test_english_control_and_safety_rules():
    assert parse_utterance("Cancel").intent is Intent.CANCEL
    assert parse_utterance("Never mind").intent is Intent.CANCEL
    assert parse_utterance("Stop now").intent is Intent.STOP
    assert parse_utterance("Go to floor three, then stop").intent is Intent.STOP
    assert parse_utterance("Do not go to floor three").reject_reason == "negated_floor_command"
    assert parse_utterance("Floor three or floor four").reject_reason == "ambiguous_floor"
    assert parse_utterance("My friend is on floor four").reject_reason == "unsafe_floor_context"


def test_concise_english_no_floor_correction_requires_confirmation_context():
    for text in ("No, first floor", "No, floor one", "No, the first floor please"):
        outside_confirmation = parse_utterance(text)
        assert outside_confirmation.intent is Intent.UNKNOWN, text

        during_confirmation = parse_utterance(
            text, allow_english_no_floor_correction=True)
        assert during_confirmation.intent is Intent.PRESS_FLOOR, text
        assert during_confirmation.floor == 1, text

    for text in ("No", "No, not first floor", "No, first floor or second floor"):
        parsed = parse_utterance(
            text, allow_english_no_floor_correction=True)
        assert parsed.intent is Intent.UNKNOWN, text


def test_english_floor_correction_selects_only_explicit_new_target():
    for text in (
        "Change from floor three to floor four",
        "Not floor three but floor four",
    ):
        parsed = parse_utterance(text)
        assert parsed.intent is Intent.PRESS_FLOOR, text
        assert parsed.floor == 4, text
