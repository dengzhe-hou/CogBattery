"""CPU regressions from technical-pilot answer shapes; no inference or gold repair."""
import pytest

from scripts.experiments.human13_20261017 import memory, tom_mc, wm_cc


def step(kind, gold=None, **meta):
    return {"kind": kind, "gold": gold, "meta": meta}


def test_complete_json_fence_retains_intrusions_duplicates_and_order():
    item = step("cvlt_recall", ["truck", "spinach"])
    result = memory.score_step(item, '```json\n["truck", "imagined", "truck"]\n```')
    assert result["valid"] and not result["strict_original_score"]["valid"]
    assert result["parsed"] == ["truck", "imagined", "truck"]
    assert result["intrusions"] == ["imagined"] and result["repetitions"] == 1
    assert not result["is_correct"]
    assert memory.score_step(step("cvlt_recall_followup", ["truck"]), '```json\n[]\n```')["parsed"] == []


@pytest.mark.parametrize("response", [
    '```json\n["truck"]',
    'Here are the words: ["truck"]',
    '```json\n["truck"]\n```\n```json\n[]\n```',
    '[truck, spinach]',
    '[["truck", "Ready"]]',
    '```json\n["truck", 2]\n```',
])
def test_json_repair_does_not_guess_missing_syntax_or_choose_arrays(response):
    assert not memory.score_step(step("cvlt_recall", ["truck"]), response)["valid"]


@pytest.mark.parametrize("response,parsed", [("2. Tammy", "Tammy"), ("1. Jake", "Jake"),
                                              ("Tammy (Option 2)", "Tammy")])
def test_number_and_option_must_agree_without_using_gold(response, parsed):
    item = step("source_choice", "Tammy", options=["Jake", "Tammy", "Olivia", "Mike"])
    result = memory.score_step(item, response)
    assert result["valid"] and result["parsed"] == parsed
    assert result["is_correct"] == (parsed == "Tammy")


@pytest.mark.parametrize("response", ["1. Tammy", "Tammy (Option 1)",
                                      "Tammy was having an IPA.\n\nAnswer: 1"])
def test_contradictory_or_prose_source_answer_stays_invalid(response):
    assert not memory.score_step(step("source_choice", "Tammy", options=["Jake", "Tammy"]), response)["valid"]


def test_complete_bold_rating_and_unclassifiable_keep_distinct_meanings():
    rating = memory.score_step(step("drm_rating"), "**7**")
    assert rating["valid"] and rating["parsed"] == 7
    assert not rating["strict_original_score"]["valid"]
    report = memory.score_step(step("drm_report"), "UNCLASSIFIABLE")
    assert report["unclassifiable_response"]
    assert not report["valid"] and report["parsed"] is None and report["is_correct"] is None
    assert not memory.score_step(step("drm_recognition", False), "NEW\nUNFAMILIAR")["valid"]


@pytest.mark.parametrize("response", ["5 8 2", "Five eight two", "582", "5, 8, 2", '"5 8 2"',
                                      "The answer is 5 8 2."])
def test_complete_digit_sequences_and_supported_fixed_prefix(response):
    result = wm_cc.score_step(step("digit_recall", [5, 8, 2]), response)
    assert result["valid"] and result["is_correct"]
    assert result["parsed"] == [5, 8, 2] and result["history_response"] == "5 8 2"


@pytest.mark.parametrize("response", [
    "Digit 1: 5, Digit 2: 8, Digit 3: 2. So",
    "Input 2 8 5; answer 5 8 2",
    "The answer is 5 8 2 because I recalled the input.",
    "5 8 2 or 2 8 5", "5.82", "-5 8 2",
])
def test_digits_are_not_mined_from_reasoning_or_multiple_candidates(response):
    result = wm_cc.score_step(step("digit_recall", [5, 8, 2]), response)
    assert not result["valid"] and result["parsed"] is None
    assert result["history_response"] == "INVALID"
    assert result["strict_original_score"]["valid"]  # Diagnostic only; never drives the new result.


def test_digit_and_letter_errors_are_not_corrected_to_gold():
    digit = wm_cc.score_step(step("digit_recall", [5, 8, 2]), "5 3 2")
    assert digit["valid"] and digit["parsed"] == [5, 3, 2] and not digit["is_correct"]
    letter = wm_cc.score_step(step("ospan_recall", ["F", "H", "J"]), '"F _ J"')
    assert letter["valid"] and letter["parsed"] == ["F", "_", "J"]
    assert letter["recall_correct_positions"] == 2 and not letter["is_correct"]
    assert not wm_cc.score_step(step("ospan_recall", ["F", "H", "J"]), "_Q_")["valid"]


def test_actual_digit_wrappers_preserve_values_and_ignore_position_labels():
    item = step("digit_recall", [5, 8, 2])
    # Actual old deepseek-r1:7b response: do not correct its reversed values.
    angled = wm_cc.score_step(item, "<2> <8> <5>")
    assert angled["valid"] and angled["parsed"] == [2, 8, 5] and not angled["is_correct"]
    # Actual old smollm2:360m response: labels are positions, not recalled digits.
    labelled = wm_cc.score_step(item, "Digit 1: 5 Digit 2: 8 Digit 3: 2")
    assert labelled["valid"] and labelled["parsed"] == [5, 8, 2] and labelled["is_correct"]
    assert labelled["strict_original_score"]["parsed"] == [1, 5, 2, 8, 3, 2]
    wrong = wm_cc.score_step(item, "Digit 1:5 Digit 2:3 Digit 3:2")
    assert wrong["valid"] and wrong["parsed"] == [5, 3, 2] and not wrong["is_correct"]


@pytest.mark.parametrize("response", [
    "Digit 2:8 Digit 1:5 Digit 3:2", "Digit 1:5 Digit 3:2", "Digit 1:5 Digit 1:8",
    "Digit 01:5 Digit 2:8 Digit 3:2", "Digit 1:5 Digit 2:8 Digit 3:2 because I recall it",
    "<5> <8> <2> or <2> <8> <5>", "<5> <8> <2> because I recall it", "<58> <2>",
    "1 3 5 7 9\n\nSequence: 1, 3, 5, 7, 9\n\nOutput: 1 3 5 7 9",
])
def test_digit_wrapper_normalization_rejects_reordering_and_explanation(response):
    assert not wm_cc.score_step(step("digit_recall", [5, 8, 2]), response)["valid"]


def test_tom_option_label_preserves_wrong_answer_and_rejects_conflict():
    item = step("tom_answer", "B", displayed_options=["Elaine likes Stephen", "Elaine likes Bernard"])
    correct = tom_mc.score_step(item, "B. Elaine likes Bernard")
    wrong = tom_mc.score_step(item, "A. Elaine likes Stephen.")
    assert correct["is_correct"] and not correct["strict_original_score"]["valid"]
    assert wrong["valid"] and wrong["parsed"] == "A" and not wrong["is_correct"]
    assert not tom_mc.score_step(item, "B. Elaine likes Stephen")["valid"]
    assert not tom_mc.score_step(item, "B. Elaine likes Bernard\n\nThe story indicates it.")["valid"]


def test_calibration_option_text_is_position_specific_and_95_is_not_rounded():
    item = step("cal_answer", "s", left_option="L'Autriche", right_option="L'Allemagne")
    for response in ["L'Autriche.", "s (L'Autriche)"]:
        assert tom_mc.score_step(item, response)["parsed"] == "s"
    assert not tom_mc.score_step(item, "l (L'Autriche)")["valid"]
    wrong = tom_mc.score_step(item, "L'Allemagne")
    assert wrong["valid"] and wrong["parsed"] == "l" and not wrong["is_correct"]
    assert not tom_mc.score_step(step("confidence"), "**95%**")["valid"]
    assert tom_mc.score_step(step("confidence"), '"100 %"')["parsed"] == 100


def test_calibration_actual_integer_options_preserve_positions_and_values():
    # Actual prepared task cal_p0_q17: Harry Potter films, left=7, right=8.
    item = step("cal_answer", "l", item_id=17, position_version=0,
                left_option=7, right_option=8, target=8)
    assert tom_mc.score_step(item, "8")["parsed"] == "l"
    assert tom_mc.score_step(item, "l (8)")["is_correct"]
    wrong = tom_mc.score_step(item, "7")
    assert wrong["valid"] and wrong["parsed"] == "s" and not wrong["is_correct"]
    assert not tom_mc.score_step(item, "s (8)")["valid"]
    assert item["meta"]["left_option"] == 7 and item["meta"]["right_option"] == 8


@pytest.mark.parametrize("response,parsed", [("R:X M R T V T M", "XMRTVTM"),
                                              ("XMTMVTEV", "XMTMVTEV"),
                                              ("R:XMMXM", "XMMXM")])
def test_wag_complete_reproduction_preserves_wrong_or_missing_letters(response, parsed):
    result = tom_mc.score_step(step("wag_training", "XMXRTVTM"), response)
    assert result["valid"] and result["parsed"] == parsed and not result["is_correct"]
    assert not result["strict_original_score"]["valid"]


@pytest.mark.parametrize("response", ["I think XMMXM", "R:XMMXM\nA:NO",
                                      "Chronological record.\nR:XMMXM", "R:XMMXM or XXRVTM"])
def test_wag_does_not_extract_a_string_from_history_or_prose(response):
    assert not tom_mc.score_step(step("wag_training", "XMMXM"), response)["valid"]


def test_wager_prefix_does_not_select_between_multiple_wagers():
    assert tom_mc.score_step(step("wager"), "Wager: 2")["parsed"] == 2
    assert tom_mc.score_step(step("wager"), "W:1")["parsed"] == 1
    assert not tom_mc.score_step(step("wager"), "W:1\nW:2")["valid"]
    assert not tom_mc.score_step(step("wager"), "R:V\nW:1")["valid"]


def test_actual_smoke_reply_prefix_and_single_drm_label_sentence():
    # New smoke: zephyr:7b go_nogo and stablelm2:1.6b DRM long probe.
    action = wm_cc.score_step(step("go_nogo_choice", "PRESS"), "Reply: PRESS")
    assert action["valid"] and action["parsed"] == "PRESS"
    assert not action["strict_original_score"]["valid"]
    report = memory.score_step(step("drm_report"), 'The label for this response is "UNFAMILIAR".')
    assert report["valid"] and report["parsed"] == "UNFAMILIAR"
    assert not report["strict_original_score"]["valid"]
    wrong = wm_cc.score_step(step("go_nogo_choice", "PRESS"), "Reply: WITHHOLD")
    assert wrong["valid"] and wrong["parsed"] == "WITHHOLD" and not wrong["is_correct"]


@pytest.mark.parametrize("response", ["Reply: PRESS because the square is blue", "Reply: PRESS or WITHHOLD",
                                      "Reply: PRESS\nWITHHOLD", "Reply: PRESS\nExplanation: it is blue"])
def test_reply_prefix_does_not_extract_from_explanation_or_multiple_answers(response):
    assert not wm_cc.score_step(step("go_nogo_choice", "PRESS"), response)["valid"]


@pytest.mark.parametrize("response", [
    'The label for this response is "UNFAMILIAR". It was absent.',
    'The label for this response is "UNFAMILIAR" or "REMEMBER".',
    'The label for this response is "WILL_REMEMBER".',
])
def test_drm_label_sentence_requires_one_existing_label_and_nothing_else(response):
    assert not memory.score_step(step("drm_report"), response)["valid"]


def test_reply_wrapper_keeps_nback_explanations_and_ready_arrays_invalid():
    item = step("n_back_choice", "MATCH", trial=22)
    assert wm_cc.score_step(item, "Reply: MATCH")["parsed"] == "MATCH"
    for response in ["Reply: MATCH (for trial 22)", "Reply: MATCH because it matches", "Reply: MATCH or NO MATCH"]:
        assert not wm_cc.score_step(item, response)["valid"]
    assert not memory.score_step(step("cvlt_recall", ["truck"]), 'Ready\n["truck"]')["valid"]
    assert not memory.score_step(step("cvlt_recall", ["truck"]), 'Reply: Ready\n["truck"]')["valid"]


def test_new_wrappers_never_make_length_response_complete():
    from scripts.experiments.human13_20261017.common import score_record
    for module, item, response in [
        (wm_cc, step("go_nogo_choice", "PRESS"), "Reply: PRESS"),
        (memory, step("drm_report"), 'The label for this response is "UNFAMILIAR".'),
    ]:
        result = score_record(module, {"step": item, "response": response, "finish_reason": "length",
                                       "protocol_valid": False, "synthetic": True})
        assert result["parsed_valid"] and not result["valid"]
        assert not result["response_complete"] and not result["behavior_eligible"]
