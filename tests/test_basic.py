"""Visible tests for Activity 5 - Constituent Services Hub.

Students can run these locally to verify their work before submission.
Run with: pytest tests/ -v
"""
import json
import os
import re

import pytest


PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULT_PATH = os.path.join(PROJECT_ROOT, "result.json")
MAIN_PATH = os.path.join(PROJECT_ROOT, "app", "main.py")
COMPLAINTS_PATH = os.path.join(PROJECT_ROOT, "data", "complaints.json")
INTENT_EXAMPLES_PATH = os.path.join(PROJECT_ROOT, "data", "intent_examples.json")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def result():
    """Load the student's result.json."""
    if not os.path.exists(RESULT_PATH):
        pytest.skip("result.json not found - run 'python app/main.py' first")
    with open(RESULT_PATH) as f:
        return json.load(f)


@pytest.fixture
def complaints():
    """Load the complaints dataset."""
    with open(COMPLAINTS_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Canary: result.json must exist
# ---------------------------------------------------------------------------

def test_result_exists():
    """result.json file must exist."""
    assert os.path.exists(RESULT_PATH), (
        "Run 'python app/main.py' to generate result.json"
    )


# ---------------------------------------------------------------------------
# 2. Contract: required fields and task name
# ---------------------------------------------------------------------------

def test_required_fields(result):
    """result.json must have required top-level fields."""
    for field in ("task", "status", "outputs", "metadata"):
        assert field in result, f"Missing required field: {field}"


def test_task_name(result):
    """Task name must be 'constituent_hub'."""
    assert result["task"] == "constituent_hub"


def test_status_valid(result):
    """Status must be one of the allowed values."""
    assert result["status"] in ("success", "partial", "error")


def test_metadata_has_timestamp(result):
    """Metadata must include a timestamp."""
    assert "timestamp" in result.get("metadata", {}), (
        "metadata.timestamp is required"
    )


def test_metadata_has_sdk_version(result):
    """Metadata must include sdk_version."""
    assert "sdk_version" in result.get("metadata", {}), (
        "metadata.sdk_version is required"
    )


# ---------------------------------------------------------------------------
# 3. PII Detection and Redaction
# ---------------------------------------------------------------------------

def test_outputs_has_pii_results(result):
    """Outputs must contain pii_results key."""
    assert "pii_results" in result["outputs"]


def test_pii_results_is_list(result):
    """pii_results must be a list."""
    assert isinstance(result["outputs"]["pii_results"], list)


def test_pii_result_has_redacted_text(result):
    """Each PII result should have a redacted_text field."""
    pii = result["outputs"].get("pii_results", [])
    if not pii:
        pytest.fail("pii_results is empty - implement Step 1 first")
    for i, item in enumerate(pii):
        assert "redacted_text" in item, (
            f"PII result {i} missing 'redacted_text'"
        )


def test_pii_result_has_entities(result):
    """Each PII result should have an entities list."""
    pii = result["outputs"].get("pii_results", [])
    if not pii:
        pytest.fail("pii_results is empty - implement Step 1 first")
    for i, item in enumerate(pii):
        assert "entities" in item, f"PII result {i} missing 'entities'"
        assert isinstance(item["entities"], list), (
            f"PII result {i} entities must be a list"
        )


def test_pii_redacted_differs_from_original(result):
    """Redacted text should differ from original when PII is present."""
    pii = result["outputs"].get("pii_results", [])
    if not pii:
        pytest.fail("pii_results is empty - implement Step 1 first")
    # At least one result should show redaction (complaints.json has PII)
    has_redaction = any(
        item.get("redacted_text") != item.get("original_text")
        for item in pii
        if "original_text" in item and "redacted_text" in item
    )
    if any("original_text" in item for item in pii):
        assert has_redaction, (
            "No PII was redacted — check that recognize_pii_entities is working"
        )


def test_pii_entity_has_category(result):
    """PII entities should include a category field."""
    pii = result["outputs"].get("pii_results", [])
    entities = [e for item in pii for e in item.get("entities", [])]
    if not entities:
        pytest.fail("No PII entities found - implement Step 1 first")
    for entity in entities:
        assert "category" in entity, "PII entity missing 'category'"
        assert "confidence_score" in entity, (
            "PII entity missing 'confidence_score'"
        )


# ---------------------------------------------------------------------------
# 4. Sentiment Analysis and Key Phrases
# ---------------------------------------------------------------------------

def test_outputs_has_sentiment_results(result):
    """Outputs must contain sentiment_results key."""
    assert "sentiment_results" in result["outputs"]


def test_sentiment_results_is_list(result):
    """sentiment_results must be a list."""
    assert isinstance(result["outputs"]["sentiment_results"], list)


def test_sentiment_has_valid_value(result):
    """Each sentiment result must have a valid sentiment string."""
    valid = {"positive", "negative", "neutral", "mixed"}
    sentiments = result["outputs"].get("sentiment_results", [])
    if not sentiments:
        pytest.fail("sentiment_results is empty - implement Step 2 first")
    for i, s in enumerate(sentiments):
        assert s.get("sentiment") in valid, (
            f"Sentiment result {i}: '{s.get('sentiment')}' not in {valid}"
        )


def test_sentiment_has_confidence_scores(result):
    """Each sentiment result must have confidence_scores dict."""
    sentiments = result["outputs"].get("sentiment_results", [])
    if not sentiments:
        pytest.fail("sentiment_results is empty - implement Step 2 first")
    for i, s in enumerate(sentiments):
        assert "confidence_scores" in s, (
            f"Sentiment result {i} missing 'confidence_scores'"
        )
        scores = s["confidence_scores"]
        assert isinstance(scores, dict), (
            f"Sentiment result {i}: confidence_scores must be a dict"
        )


def test_sentiment_has_key_phrases(result):
    """Each sentiment result must have key_phrases list."""
    sentiments = result["outputs"].get("sentiment_results", [])
    if not sentiments:
        pytest.fail("sentiment_results is empty - implement Step 2 first")
    for i, s in enumerate(sentiments):
        assert "key_phrases" in s, (
            f"Sentiment result {i} missing 'key_phrases'"
        )
        assert isinstance(s["key_phrases"], list), (
            f"Sentiment result {i}: key_phrases must be a list"
        )


# ---------------------------------------------------------------------------
# 5. Language Detection and Translation
# ---------------------------------------------------------------------------

def test_outputs_has_translation_results(result):
    """Outputs must contain translation_results key."""
    assert "translation_results" in result["outputs"]


def test_translation_results_is_list(result):
    """translation_results must be a list."""
    assert isinstance(result["outputs"]["translation_results"], list)


def test_translation_has_detected_language(result):
    """Each translation result must have detected_language."""
    translations = result["outputs"].get("translation_results", [])
    if not translations:
        pytest.fail("translation_results is empty - implement Step 3 first")
    for i, t in enumerate(translations):
        assert "detected_language" in t, (
            f"Translation result {i} missing 'detected_language'"
        )


def test_translation_has_translated_text(result):
    """Each translation result must have translated_text."""
    translations = result["outputs"].get("translation_results", [])
    if not translations:
        pytest.fail("translation_results is empty - implement Step 3 first")
    for i, t in enumerate(translations):
        assert "translated_text" in t, (
            f"Translation result {i} missing 'translated_text'"
        )


def test_translation_has_was_translated(result):
    """Each translation result must have was_translated bool."""
    translations = result["outputs"].get("translation_results", [])
    if not translations:
        pytest.fail("translation_results is empty - implement Step 3 first")
    for i, t in enumerate(translations):
        assert "was_translated" in t, (
            f"Translation result {i} missing 'was_translated'"
        )
        assert isinstance(t["was_translated"], bool), (
            f"Translation result {i}: was_translated must be a bool"
        )


# ---------------------------------------------------------------------------
# 6. Intent Recognition
# ---------------------------------------------------------------------------

def test_outputs_has_intent_results(result):
    """Outputs must contain intent_results key."""
    assert "intent_results" in result["outputs"]


def test_intent_results_is_list(result):
    """intent_results must be a list."""
    assert isinstance(result["outputs"]["intent_results"], list)


def test_intent_has_top_intent(result):
    """Each intent result must have top_intent."""
    intents = result["outputs"].get("intent_results", [])
    if not intents:
        pytest.fail("intent_results is empty - implement Step 4 first")
    for i, intent in enumerate(intents):
        assert "top_intent" in intent, (
            f"Intent result {i} missing 'top_intent'"
        )


def test_intent_has_valid_values(result):
    """Intent values must be one of the three expected intents."""
    valid = {"report-issue", "check-status", "ask-question"}
    intents = result["outputs"].get("intent_results", [])
    if not intents:
        pytest.fail("intent_results is empty - implement Step 4 first")
    for i, intent in enumerate(intents):
        assert intent.get("top_intent") in valid, (
            f"Intent result {i}: '{intent.get('top_intent')}' not in {valid}"
        )


def test_intent_has_confidence(result):
    """Each intent result must have a confidence score."""
    intents = result["outputs"].get("intent_results", [])
    if not intents:
        pytest.fail("intent_results is empty - implement Step 4 first")
    for i, intent in enumerate(intents):
        assert "confidence" in intent, (
            f"Intent result {i} missing 'confidence'"
        )
        assert isinstance(intent["confidence"], (int, float)), (
            f"Intent result {i}: confidence must be a number"
        )


# ---------------------------------------------------------------------------
# 7. Data files
# ---------------------------------------------------------------------------

def test_complaints_file_exists():
    """data/complaints.json must exist."""
    assert os.path.exists(COMPLAINTS_PATH), (
        "data/complaints.json is required"
    )


def test_complaints_valid_schema(complaints):
    """Each complaint must have id, text, and metadata."""
    for i, c in enumerate(complaints):
        assert "id" in c, f"Complaint {i} missing 'id'"
        assert "text" in c, f"Complaint {i} missing 'text'"
        assert "metadata" in c, f"Complaint {i} missing 'metadata'"
        assert isinstance(c["text"], str) and len(c["text"]) > 10, (
            f"Complaint {i}: text must be a non-trivial string"
        )


def test_complaints_has_variety(complaints):
    """Dataset should have at least 6 complaints for meaningful testing."""
    assert len(complaints) >= 6, (
        f"Expected ≥6 complaints, got {len(complaints)}"
    )


# ---------------------------------------------------------------------------
# 8. Security
# ---------------------------------------------------------------------------

def test_no_hardcoded_keys():
    """Source code must not contain hardcoded API keys."""
    with open(MAIN_PATH, encoding="utf-8") as f:
        source = f.read()
    suspicious = [
        r'["\']https?://\S+\.cognitiveservices\.azure\.com\S*["\']',
        r'["\'][A-Fa-f0-9]{32}["\']',
    ]
    for pattern in suspicious:
        matches = re.findall(pattern, source)
        real = [
            m for m in matches
            if "example" not in m.lower()
            and "your-" not in m.lower()
            and "placeholder" not in m.lower()
        ]
        assert len(real) == 0, (
            f"Possible hardcoded credential: {real[0][:50]}"
        )


def test_no_hardcoded_endpoints():
    """Source code must not contain hardcoded service endpoints."""
    with open(MAIN_PATH, encoding="utf-8") as f:
        source = f.read()
    # Check for real endpoint patterns (not comments or env var references)
    lines = source.split("\n")
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            continue
        if "your-resource" in stripped or "example" in stripped:
            continue
        if re.search(
            r'=\s*["\']https://\S+\.cognitiveservices\.azure\.com',
            stripped,
        ):
            assert False, (
                f"Line {line_num}: possible hardcoded endpoint"
            )


# ---------------------------------------------------------------------------
# 9. Pipeline summary
# ---------------------------------------------------------------------------

def test_pipeline_summary_exists(result):
    """Outputs must contain pipeline_summary."""
    assert "pipeline_summary" in result["outputs"], (
        "outputs.pipeline_summary is required"
    )


def test_pipeline_summary_total(result):
    """pipeline_summary.total_complaints must match data file."""
    summary = result["outputs"].get("pipeline_summary", {})
    if not summary:
        pytest.fail("pipeline_summary not found")
    assert "total_complaints" in summary, (
        "pipeline_summary missing 'total_complaints'"
    )
    with open(COMPLAINTS_PATH, encoding="utf-8") as f:
        expected = len(json.load(f))
    assert summary["total_complaints"] == expected, (
        f"total_complaints={summary['total_complaints']}, expected {expected}"
    )


def test_pipeline_summary_steps(result):
    """pipeline_summary.steps_completed must be a list."""
    summary = result["outputs"].get("pipeline_summary", {})
    if not summary:
        pytest.fail("pipeline_summary not found")
    assert "steps_completed" in summary, (
        "pipeline_summary missing 'steps_completed'"
    )
    assert isinstance(summary["steps_completed"], list), (
        "steps_completed must be a list"
    )
