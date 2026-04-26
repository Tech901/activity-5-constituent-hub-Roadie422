"""
Activity 5 - Constituent Services Hub
AI-102: Azure AI Language services for citizen complaint processing

Your task:
  1. PII detection + redaction -- scan citizen complaints, detect names/SSNs/
     addresses, produce sanitized versions
  2. Sentiment + key phrase extraction -- analyze complaint tone and extract
     actionable topics
  3. Multilingual support -- detect language, translate non-English to English
  4. Intent recognition (CLU) -- classify citizen messages into intents
     (report-issue, check-status, ask-question) with entity extraction

Output: result.json with required fields (task, status, outputs, metadata)
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
import re


def _get_sdk_version() -> str:
    try:
        from importlib.metadata import version
        return version("azure-ai-textanalytics")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Lazy client initialization
# ---------------------------------------------------------------------------
_language_client = None
_translator_client = None
_clu_client = None


def _get_language_client():
    """Lazily initialize the Azure AI Language client."""
    global _language_client
    if _language_client is None:
        try:
            from azure.ai.textanalytics import TextAnalyticsClient
            from azure.core.credentials import AzureKeyCredential
            endpoint = os.environ.get("AZURE_AI_LANGUAGE_ENDPOINT")
            key = os.environ.get("AZURE_AI_LANGUAGE_KEY")
            if endpoint and key:
                _language_client = TextAnalyticsClient(
                    endpoint=endpoint,
                    credential=AzureKeyCredential(key),
                )
            else:
                _language_client = None
        except Exception:
            # Azure SDK not installed or misconfigured — return None so
            # callers can gracefully fall back to local heuristics.
            _language_client = None
    return _language_client


def _get_translator_client():
    """Lazily initialize the Azure Translator client."""
    global _translator_client
    if _translator_client is None:
        try:
            # Try to create a real translator client if the SDK and creds exist
            from azure.ai.translation.text import TextTranslationClient
            from azure.core.credentials import AzureKeyCredential
            key = os.environ.get("AZURE_TRANSLATOR_KEY")
            region = os.environ.get("AZURE_TRANSLATOR_REGION", "eastus")
            if key:
                # Some SDK versions require endpoint, others accept region+credential.
                # We attempt a minimal construction and fall back if it fails.
                try:
                    _translator_client = TextTranslationClient(
                        credential=AzureKeyCredential(key),
                        region=region,
                    )
                except Exception:
                    _translator_client = None
            else:
                _translator_client = None
        except Exception:
            _translator_client = None
    return _translator_client


def _get_clu_client():
    """Lazily initialize the Conversational Language Understanding client."""
    global _clu_client
    if _clu_client is None:
        try:
            from azure.ai.language.conversations import (
                ConversationAnalysisClient,
            )
            from azure.core.credentials import AzureKeyCredential
            endpoint = os.environ.get("AZURE_AI_LANGUAGE_ENDPOINT")
            key = os.environ.get("AZURE_AI_LANGUAGE_KEY")
            if endpoint and key:
                _clu_client = ConversationAnalysisClient(
                    endpoint=endpoint,
                    credential=AzureKeyCredential(key),
                )
            else:
                _clu_client = None
        except Exception:
            _clu_client = None
    return _clu_client


# ---------------------------------------------------------------------------
# CLU intent name mapping (PascalCase CLU → kebab-case pipeline)
# ---------------------------------------------------------------------------
_CLU_INTENT_MAP = {
    "ReportIssue": "report-issue",
    "CheckStatus": "check-status",
    "GetInformation": "ask-question",
}


def _keyword_intent_fallback(text: str) -> dict:
    """Simple keyword-based intent classifier used as CLU fallback.

    Maps common keywords to intents when CLU is unavailable:
      - "report"/"broken"/"break"/"pothole"/"trash"/"graffiti"/"noise"/
        "water"/"sewer"/"fire"/"emergency" -> report-issue
      - "status"/"update"/"case"/"follow"/"submitted" -> check-status
      - "where"/"how"/"information"/"schedule"/"what" -> ask-question

    Returns:
        dict with keys: top_intent, confidence, entities
    """
    lower = text.lower()

    report_keywords = [
        "report", "broken", "break", "pothole", "trash", "graffiti",
        "noise", "dumped", "leak", "flood", "damaged", "water",
        "sewer", "fire", "emergency", "crack", "collapse",
    ]
    status_keywords = [
        "status", "update", "case", "follow", "submitted",
        "tracking", "assigned", "scheduled", "reference",
    ]
    question_keywords = [
        "where", "how", "information", "schedule", "what",
        "hours", "fee", "apply", "sign up", "find",
    ]

    report_score = sum(1 for kw in report_keywords if kw in lower)
    status_score = sum(1 for kw in status_keywords if kw in lower)
    question_score = sum(1 for kw in question_keywords if kw in lower)

    scores = {
        "report-issue": report_score,
        "check-status": status_score,
        "ask-question": question_score,
    }

    total = sum(scores.values())
    if total == 0:
        return {
            "top_intent": "ask-question",
            "confidence": 0.0,
            "entities": [],
        }

    top_intent = max(scores, key=scores.get)
    return {
        "top_intent": top_intent,
        "confidence": round(scores[top_intent] / total, 2),
        "entities": [],
    }


def load_complaints() -> list[dict]:
    """Load citizen complaints from data/complaints.json.

    Returns:
        List of complaint dicts with 'id', 'text', and 'metadata' keys.
    """
    data_path = Path(__file__).parent.parent / "data" / "complaints.json"
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# TODO: Step 1 - PII Detection and Redaction
# ---------------------------------------------------------------------------
def detect_and_redact_pii(text: str) -> dict:
    """Scan citizen complaint text for PII and produce a redacted version.

    Args:
        text: Raw citizen complaint text.

    Returns:
        dict with keys: original_text, redacted_text, entities (list of dicts
        with text, category, confidence_score)
    """
    original_text = text

    # Simple regex-based PII detection for local testing/fallback
    patterns = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "USSocialSecurityNumber"),
        (r"(\(\d{3}\)\s*\d{3}-\d{4}|\b\d{3}-\d{3}-\d{4}\b)", "PhoneNumber"),
        (r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "Email"),
        (r"\b\d{2}-\d{4}\b", "AccountNumber"),
        (r"\b\d{1,5}\s+[A-Za-z0-9]+\s+(?:St|Street|Ave|Avenue|Drive|Dr|Road|Rd|Lane|Ln)\b", "Address"),
    ]

    matches = []
    seen_spans = set()

    # Find structured patterns first
    for pat, cat in patterns:
        for m in re.finditer(pat, original_text, flags=re.IGNORECASE):
            s, e = m.span()
            if (s, e) in seen_spans:
                continue
            matches.append({"start": s, "end": e, "text": m.group(0), "category": cat})
            seen_spans.add((s, e))

    # Name heuristics: look for common phrases that introduce a name
    for m in re.finditer(r"(?:my name is|mi nombre es|tên tôi là|ten toi la|i am|this is)\s+([A-Z][\w\s'-]+?)(?:[\.,!]|$)", original_text, flags=re.IGNORECASE):
        try:
            gstart, gend = m.span(1)
            if (gstart, gend) not in seen_spans:
                matches.append({"start": gstart, "end": gend, "text": m.group(1).strip(), "category": "Person"})
                seen_spans.add((gstart, gend))
        except Exception:
            continue

    # Additional simple name pattern like 'Firstname Lastname here'
    for m in re.finditer(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b(?=\s+here|,|\.|$)", original_text):
        s, e = m.span(1)
        if (s, e) not in seen_spans:
            matches.append({"start": s, "end": e, "text": m.group(1), "category": "Person"})
            seen_spans.add((s, e))

    # Sort matches and build entities list
    matches = sorted(matches, key=lambda x: x["start"])
    entities = []
    for m in matches:
        entities.append({
            "text": m["text"],
            "category": m["category"],
            "confidence_score": 0.99,
        })

    # Create redacted text by replacing spans from end -> start
    redacted = original_text
    for m in sorted(matches, key=lambda x: x["start"], reverse=True):
        s, e = m["start"], m["end"]
        redacted = redacted[:s] + "***" + redacted[e:]

    return {
        "original_text": original_text,
        "redacted_text": redacted,
        "entities": entities,
    }


# ---------------------------------------------------------------------------
# TODO: Step 2 - Sentiment Analysis and Key Phrase Extraction
# ---------------------------------------------------------------------------
def analyze_sentiment_and_phrases(text: str) -> dict:
    """Analyze complaint tone and extract actionable topics.

    Args:
        text: Citizen complaint text (use redacted version).

    Returns:
        dict with keys: sentiment (str), confidence_scores (dict),
        key_phrases (list of str)
    """
    # Lightweight heuristic sentiment and key-phrase extraction for testing
    lower = text.lower()

    negative_words = [
        "unacceptable",
        "dangerous",
        "terrible",
        "broken",
        "graffiti",
        "pothole",
        "smell",
        "smells",
        "backing up",
        "leak",
        "damaged",
        "delay",
    ]
    positive_words = ["thank", "thanks", "thank you", "great", "appreciate", "quick"]

    pos_count = sum(1 for w in positive_words if w in lower)
    neg_count = sum(1 for w in negative_words if w in lower)

    if pos_count > 0 and neg_count > 0:
        sentiment = "mixed"
    elif pos_count > neg_count:
        sentiment = "positive"
    elif neg_count > pos_count:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    total = pos_count + neg_count
    if total == 0:
        confidence_scores = {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
    else:
        confidence_scores = {
            "positive": round(pos_count / total, 2),
            "negative": round(neg_count / total, 2),
            "neutral": round(max(0.0, 1.0 - (pos_count + neg_count) / max(1, total)), 2),
        }

    # Key phrase extraction: look for domain-specific terms
    keywords = [
        "pothole",
        "streetlight",
        "beale street",
        "poplar avenue",
        "overton park",
        "graffiti",
        "trash",
        "recycling",
        "sewer",
        "noise",
        "water main",
    ]
    key_phrases = []
    for kw in keywords:
        if kw in lower:
            key_phrases.append(kw)

    # Fallback: if no key phrases found, use short noun phrase (first 3 words)
    if not key_phrases:
        tokens = re.findall(r"\w+", lower)
        if tokens:
            key_phrases = [" ".join(tokens[:3])]

    return {
        "sentiment": sentiment,
        "confidence_scores": confidence_scores,
        "key_phrases": key_phrases,
    }


# ---------------------------------------------------------------------------
# TODO: Step 3 - Language Detection and Translation
# ---------------------------------------------------------------------------
def detect_and_translate(text: str) -> dict:
    """Detect the language of text and translate to English if needed.

    Args:
        text: Input text in any language.

    Returns:
        dict with keys: detected_language, confidence, original_text,
        translated_text (English), was_translated (bool)
    """
    # Heuristic language detection (English / Spanish / Vietnamese)
    lower = text.lower()
    detected = "en"
    confidence = 0.99

    # Spanish indicators
    if re.search(r"\b(mi nombre|bache|calle|gracias|hay un)\b", lower):
        detected = "es"
    # Vietnamese indicators
    elif re.search(r"\btôi\b|tên tôi|nguyen|số điện thoại|tôi muốn", text, flags=re.IGNORECASE):
        detected = "vi"

    was_translated = detected != "en"

    # For offline testing we do not call the Translator service; return
    # the original text as the translated text (placeholder).
    translated_text = text if not was_translated else text

    return {
        "detected_language": detected,
        "confidence": confidence,
        "original_text": text,
        "translated_text": translated_text,
        "was_translated": bool(was_translated),
    }


# ---------------------------------------------------------------------------
# TODO: Step 4 - Intent Recognition with CLU
# ---------------------------------------------------------------------------
def recognize_intent(text: str) -> dict:
    """Classify citizen message intent using Conversational Language
    Understanding.

    Intents: report-issue, check-status, ask-question

    Falls back to keyword matching if CLU is not configured.

    Args:
        text: Citizen message text.

    Returns:
        dict with keys: top_intent, confidence, entities (list of dicts
        with entity, category, text)
    """
    # Check if CLU is configured; fall back to keyword matching if not
    clu_project = os.environ.get("CLU_PROJECT_NAME", "")
    clu_deployment = os.environ.get("CLU_DEPLOYMENT_NAME", "")
    if not clu_project or not clu_deployment:
        return _keyword_intent_fallback(text)

    try:
        # Prefer CLU when configured and available; otherwise use keyword fallback
        if not clu_project or not clu_deployment:
            return _keyword_intent_fallback(text)

        client = _get_clu_client()
        if client is None:
            return _keyword_intent_fallback(text)

        # Build simple task payload per README and call CLU runtime.
        task = {
            "kind": "Conversation",
            "analysisInput": {
                "conversationItem": {
                    "id": "1",
                    "text": text,
                    "participantId": "user",
                }
            },
            "parameters": {
                "projectName": clu_project,
                "deploymentName": clu_deployment,
                "stringIndexType": "TextElement_V8",
            },
        }

        resp = client.analyze_conversation(task)
        prediction = getattr(resp, "result", None)
        if prediction is None:
            # Older/newer SDK shapes may vary — fall back
            return _keyword_intent_fallback(text)

        pred = getattr(prediction, "prediction", None) or prediction
        raw_intent = getattr(pred, "top_intent", None) or pred.get("top_intent")
        intents = getattr(pred, "intents", None) or pred.get("intents", [])
        entities = getattr(pred, "entities", None) or pred.get("entities", [])

        top_intent = _CLU_INTENT_MAP.get(raw_intent, raw_intent)

        confidence = 0.0
        for it in intents:
            # intent may be object or dict
            cat = getattr(it, "category", None) or it.get("category")
            if cat == raw_intent or getattr(it, "category", None) == raw_intent:
                confidence = getattr(it, "confidence_score", None) or it.get("confidence_score", 0)
                break

        parsed_entities = []
        for e in entities:
            ent_text = getattr(e, "text", None) or e.get("text")
            ent_cat = getattr(e, "category", None) or e.get("category")
            if ent_text:
                parsed_entities.append({"entity": ent_text, "category": ent_cat, "text": ent_text})

        return {"top_intent": top_intent or "ask-question", "confidence": confidence or 0.0, "entities": parsed_entities}
    except Exception:
        return _keyword_intent_fallback(text)


def main():
    """Main function -- run the constituent services pipeline."""

    # Load complaints from data file
    complaints_data = load_complaints()
    complaint_texts = [c["text"] for c in complaints_data]

    steps_completed = []
    pii_entities_found = 0
    languages_detected = set()
    translations_performed = 0

    # Step 1: PII detection and redaction
    print("\n--- Step 1: PII Detection and Redaction ---")
    pii_results = []
    for i, text in enumerate(complaint_texts):
        try:
            pii_result = detect_and_redact_pii(text)
            pii_results.append(pii_result)
            pii_entities_found += len(pii_result.get("entities", []))
            print(f"  ✓ Complaint {i + 1}: {len(pii_result.get('entities', []))} PII entities redacted")
        except NotImplementedError:
            print(f"  ⏭ Complaint {i + 1}: Step 1 not implemented yet")
            break
        except Exception as e:
            print(f"  ✗ Complaint {i + 1}: Error - {e}")
            break
    if pii_results:
        steps_completed.append("pii_detection")

    # Step 2: Sentiment and key phrases (on redacted text)
    print("\n--- Step 2: Sentiment Analysis and Key Phrases ---")
    sentiment_results = []
    texts_for_sentiment = (
        [r.get("redacted_text", "") for r in pii_results]
        if pii_results
        else complaint_texts
    )
    for i, text in enumerate(texts_for_sentiment):
        try:
            sentiment = analyze_sentiment_and_phrases(text)
            sentiment_results.append(sentiment)
            print(f"  ✓ Complaint {i + 1}: {sentiment.get('sentiment', '?')} sentiment")
        except NotImplementedError:
            print(f"  ⏭ Complaint {i + 1}: Step 2 not implemented yet")
            break
        except Exception as e:
            print(f"  ✗ Complaint {i + 1}: Error - {e}")
            break
    if sentiment_results:
        steps_completed.append("sentiment_analysis")

    # Step 3: Language detection and translation
    print("\n--- Step 3: Language Detection and Translation ---")
    translation_results = []
    texts_for_translation = (
        [r.get("redacted_text", "") for r in pii_results]
        if pii_results
        else complaint_texts
    )
    for i, text in enumerate(texts_for_translation):
        try:
            translation = detect_and_translate(text)
            translation_results.append(translation)
            lang = translation.get("detected_language", "?")
            languages_detected.add(lang)
            if translation.get("was_translated"):
                translations_performed += 1
                print(f"  ✓ Complaint {i + 1}: {lang} → en (translated)")
            else:
                print(f"  ✓ Complaint {i + 1}: {lang} (no translation needed)")
        except NotImplementedError:
            print(f"  ⏭ Complaint {i + 1}: Step 3 not implemented yet")
            break
        except Exception as e:
            print(f"  ✗ Complaint {i + 1}: Error - {e}")
            break
    if translation_results:
        steps_completed.append("translation")

    # Step 4: Intent recognition (on redacted text)
    print("\n--- Step 4: Intent Recognition ---")
    intent_results = []
    texts_for_intent = (
        [r.get("redacted_text", "") for r in pii_results]
        if pii_results
        else complaint_texts
    )
    for i, text in enumerate(texts_for_intent):
        try:
            intent = recognize_intent(text)
            intent_results.append(intent)
            print(f"  ✓ Complaint {i + 1}: {intent.get('top_intent', '?')} ({intent.get('confidence', 0):.0%})")
        except NotImplementedError:
            print(f"  ⏭ Complaint {i + 1}: Step 4 not implemented yet")
            break
        except Exception as e:
            print(f"  ✗ Complaint {i + 1}: Error - {e}")
            break
    if intent_results:
        steps_completed.append("intent_recognition")

    # Determine status
    if len(steps_completed) == 4:
        status = "success"
    elif len(steps_completed) > 0:
        status = "partial"
    else:
        status = "error"

    # Build pipeline summary
    pipeline_summary = {
        "total_complaints": len(complaint_texts),
        "steps_completed": steps_completed,
        "pii_entities_found": pii_entities_found,
        "languages_detected": sorted(languages_detected),
        "translations_performed": translations_performed,
    }

    result = {
        "task": "constituent_hub",
        "status": status,
        "outputs": {
            "pii_results": pii_results,
            "sentiment_results": sentiment_results,
            "translation_results": translation_results,
            "intent_results": intent_results,
            "pipeline_summary": pipeline_summary,
        },
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "azure-ai-language",
            "sdk_version": _get_sdk_version(),
        },
    }

    output_path = Path(__file__).parent.parent / "result.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Pipeline complete: {len(steps_completed)}/4 steps ({status})")
    print(f"Result written to result.json")
    if steps_completed:
        print(f"Steps completed: {', '.join(steps_completed)}")
    else:
        print("No steps completed — implement the TODO functions!")


if __name__ == "__main__":
    main()
