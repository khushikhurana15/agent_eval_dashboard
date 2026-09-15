# backend/scoring.py
#
# Per-question scoring, driven by the `scoring_method` field already tagged
# on each golden_dataset.json entry (exact_match / semantic_match /
# refusal_check / tool_only). No RAGAS or second LLM-judge call — simple,
# transparent heuristics that are easy to explain and easy to debug when a
# score looks wrong, which matters more than precision for a portfolio-scale
# eval suite.

import re

REFUSAL_PHRASES = [
    "don't know", "do not know", "not sure", "cannot answer", "can't answer",
    "unable to answer", "no information", "not available in the documents",
    "not able to find", "couldn't find", "not something i can",
    "i'm not able to", "i am not able to", "no way to know", "not possible for me",
    # Added after run_2 review: these are real refusals the original list
    # missed, e.g. "I don't have access to that personal information" and
    # "there are no ancient Sumerian records... no basis in real historical
    # documents" — both correct refusals/corrections that went undetected.
    "don't have access", "do not have access", "no access to",
    "there are no", "no basis in", "no records", "doesn't exist",
    "does not exist", "no such", "not aware of any",
    # Added after run_4: ANOTHER distinct missed phrasing — "I don't have
    # any information about X" — confirming this is a genuine whack-a-mole
    # problem (two separate runs, two separate new phrasings found), not a
    # one-off gap. See README "Known Limitations" for the accepted tradeoff:
    # patching known misses is fine, but chasing every future phrasing isn't
    # worth it at this scale — a production system would use an LLM-judge
    # call for refusal detection instead of substring matching.
    "don't have any information", "do not have any information",
    "have any information about",
]
# KNOWN RISK: this is substring matching, not phrase-boundary-aware, so a
# broad phrase like "no such" can misfire on a confident, correct technical
# answer that happens to contain it in a non-refusal sense — e.g. "There is
# no such thing as a universal regularization technique, it depends on the
# model" is a substantive answer, not a refusal, but would be flagged as one.
# Not observed in the current 27-question golden dataset, but worth
# re-checking whenever new questions are added, since a false positive here
# would silently mis-score a correct answer as a refusal.


def is_refusal(text: str) -> bool:
    text = (text or "").lower()
    return any(phrase in text for phrase in REFUSAL_PHRASES)


def _normalize_word(word: str) -> str:
    """
    Crude suffix-stripping so trivial morphological differences (plural vs
    singular, -ing vs -s) don't count as a mismatch — e.g. "neurons" vs
    "neuron", "ensembles" vs "ensemble", "improving" vs "improves" all
    normalize to the same root. Not a real stemmer (no dictionary, no
    handling of irregulars), but good enough to stop obviously-correct
    answers from failing semantic_match purely on word form.
    """
    for suffix in ("ing", "ed", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def semantic_overlap_score(answer: str, expected_summary: str) -> float:
    """
    Lightweight stand-in for semantic matching: what fraction of the
    "significant" words (4+ letters, to skip filler like 'is'/'the') in the
    expected summary also show up in the actual answer, after normalizing
    for simple plural/verb-tense differences. Not as good as an LLM-judge,
    but zero extra API calls, deterministic, and good enough to catch
    "agent didn't mention the key concept at all".
    """
    def significant_words(s):
        words = re.findall(r"[a-zA-Z]+", s.lower())
        return set(_normalize_word(w) for w in words if len(w) >= 4)

    expected_words = significant_words(expected_summary)
    if not expected_words:
        return 0.0
    answer_words = significant_words(answer)
    overlap = expected_words & answer_words
    return len(overlap) / len(expected_words)


def score_question(golden_item, tools_used, final_answer, rag_gated):
    """
    golden_item: one entry from golden_dataset.json
    tools_used: list of tool names the agent actually called, in order
    final_answer: the agent's final text answer
    rag_gated: True/False/None — whether rag_tool hit LOW_CONFIDENCE during
               this run (None if rag_tool was never called)
    """
    category = golden_item["category"]
    scoring_method = golden_item["scoring_method"]

    # ---- Tool correctness ----
    if category == "unanswerable":
        # expected_tool is "none" here — real judgment happens via
        # refusal_check on the answer below, not the tool choice itself.
        tool_correct = True
    elif category == "ambiguous":
        acceptable = golden_item.get("acceptable_tools", [])
        tool_correct = any(t in acceptable for t in tools_used)
    else:
        expected_tool = golden_item.get("expected_tool")
        tool_correct = expected_tool in tools_used

    # ---- Answer correctness ----
    answer_correct = None
    if scoring_method == "exact_match":
        expected = str(golden_item.get("expected_answer", "")).strip().lower()
        # Strip comma thousand-separators before comparing — otherwise a
        # correctly formatted answer like "8,100" fails to match the
        # expected "8100" purely because of formatting, not correctness.
        normalized_answer = (final_answer or "").lower().replace(",", "")
        answer_correct = expected in normalized_answer
    elif scoring_method == "semantic_match":
        expected_summary = golden_item.get("expected_answer_summary", "")
        overlap = semantic_overlap_score(final_answer or "", expected_summary)
        # 0.4, not 0.5 — the suffix-stripping normalizer isn't a real
        # stemmer (e.g. "improve"/"improving" still don't fully align
        # because of the silent-e case), so a stricter cutoff kept
        # producing false negatives on genuinely correct, well-paraphrased
        # answers. Genuinely wrong/off-topic answers score far lower than
        # this (typically 0.1-0.2), so this still discriminates well.
        answer_correct = overlap >= 0.4
    elif scoring_method == "refusal_check":
        answer_correct = is_refusal(final_answer)
    elif scoring_method == "tool_only":
        answer_correct = None  # non-deterministic content, not auto-graded

    # ---- Hallucination flag ----
    # Heuristic: rag_tool explicitly signaled LOW_CONFIDENCE (gated=True) but
    # the agent still produced a confident, non-refusal answer anyway — i.e.
    # it likely answered from ungrounded general knowledge instead of saying
    # "not in the documents". This directly uses the confidence-gate signal
    # you already built into rag_tool, rather than a separate fact-checking
    # pass.
    hallucination_flag = bool(rag_gated) and not is_refusal(final_answer)

    # ---- Overall pass/fail for this question ----
    if scoring_method == "tool_only":
        passed = tool_correct
    else:
        passed = bool(tool_correct) and bool(answer_correct)

    return {
        "tool_correct": tool_correct,
        "answer_correct": answer_correct,
        "hallucination_flag": hallucination_flag,
        "passed": passed,
    }