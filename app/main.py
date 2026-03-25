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
        # TODO: Uncomment and configure the TextAnalyticsClient
        #   1. Import TextAnalyticsClient from azure.ai.textanalytics
        #   2. Import AzureKeyCredential from azure.core.credentials
        #   3. Read AZURE_AI_LANGUAGE_ENDPOINT from environment
        #   4. Read AZURE_AI_LANGUAGE_KEY from environment
        #   5. Create the client with endpoint and credential
        #
        #   from azure.ai.textanalytics import TextAnalyticsClient
        #   from azure.core.credentials import AzureKeyCredential
        #   _language_client = TextAnalyticsClient(
        #       endpoint=os.environ["AZURE_AI_LANGUAGE_ENDPOINT"],
        #       credential=AzureKeyCredential(
        #           os.environ["AZURE_AI_LANGUAGE_KEY"]
        #       ),
        #   )
        raise NotImplementedError("Configure the AI Language client")
    return _language_client


def _get_translator_client():
    """Lazily initialize the Azure Translator client."""
    global _translator_client
    if _translator_client is None:
        # TODO: Uncomment and configure the TextTranslationClient
        #   1. Import TextTranslationClient from azure.ai.translation.text
        #   2. Import AzureKeyCredential from azure.core.credentials
        #   3. Read AZURE_TRANSLATOR_KEY from environment
        #   4. Read AZURE_TRANSLATOR_REGION from environment (default: "eastus")
        #   5. Create the client with credential and region
        #
        #   from azure.ai.translation.text import TextTranslationClient
        #   from azure.core.credentials import AzureKeyCredential
        #   _translator_client = TextTranslationClient(
        #       credential=AzureKeyCredential(
        #           os.environ["AZURE_TRANSLATOR_KEY"]
        #       ),
        #       region=os.environ.get("AZURE_TRANSLATOR_REGION", "eastus"),
        #   )
        raise NotImplementedError("Configure the Translator client")
    return _translator_client


def _get_clu_client():
    """Lazily initialize the Conversational Language Understanding client."""
    global _clu_client
    if _clu_client is None:
        # TODO: Uncomment and configure the ConversationAnalysisClient
        #   1. Import ConversationAnalysisClient from
        #      azure.ai.language.conversations
        #   2. Import AzureKeyCredential from azure.core.credentials
        #   3. Read AZURE_AI_LANGUAGE_ENDPOINT from environment
        #   4. Read AZURE_AI_LANGUAGE_KEY from environment
        #   5. Create the client with endpoint and credential
        #
        #   from azure.ai.language.conversations import (
        #       ConversationAnalysisClient,
        #   )
        #   from azure.core.credentials import AzureKeyCredential
        #   _clu_client = ConversationAnalysisClient(
        #       endpoint=os.environ["AZURE_AI_LANGUAGE_ENDPOINT"],
        #       credential=AzureKeyCredential(
        #           os.environ["AZURE_AI_LANGUAGE_KEY"]
        #       ),
        #   )
        raise NotImplementedError("Configure the CLU client")
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
    # TODO: Step 1.1 - Get the Language client using _get_language_client()
    # TODO: Step 1.2 - Call client.recognize_pii_entities([text])
    #   The SDK accepts a list of documents; pass [text] for a single doc.
    #   The response is a list of RecognizePiiEntitiesResult objects.
    # TODO: Step 1.3 - Extract result.redacted_text for the sanitized version
    #   The SDK automatically replaces PII with category placeholders like
    #   "***" — use this directly instead of building your own redaction.
    # TODO: Step 1.4 - Iterate result.entities to build the entities list
    #   Each entity has: .text, .category, .confidence_score
    #   Example: {"text": "John Smith", "category": "Person",
    #             "confidence_score": 0.98}
    # TODO: Step 1.5 - Return dict with original_text, redacted_text, entities
    raise NotImplementedError("Implement detect_and_redact_pii in Step 1")


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
    # TODO: Step 2.1 - Get the Language client using _get_language_client()
    # TODO: Step 2.2 - Call client.analyze_sentiment([text])
    #   Response contains .sentiment ("positive"/"negative"/"neutral"/"mixed")
    #   and .confidence_scores with .positive, .negative, .neutral floats.
    # TODO: Step 2.3 - Call client.extract_key_phrases([text])
    #   Response contains .key_phrases — a list of strings like
    #   ["broken streetlight", "Beale Street"].
    # TODO: Step 2.4 - Build confidence_scores dict:
    #   {"positive": 0.01, "negative": 0.95, "neutral": 0.04}
    # TODO: Step 2.5 - Return dict with sentiment, confidence_scores,
    #   key_phrases
    raise NotImplementedError(
        "Implement analyze_sentiment_and_phrases in Step 2"
    )


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
    # TODO: Step 3.1 - Get the Language client using _get_language_client()
    # TODO: Step 3.2 - Call client.detect_language([text])
    #   Response contains .primary_language.iso6391_name (e.g., "en", "es")
    #   and .primary_language.confidence_score (float 0-1).
    # TODO: Step 3.3 - If language is not "en", get the Translator client
    #   using _get_translator_client()
    # TODO: Step 3.4 - Call translator.translate(
    #       body=[text], to_language=["en"])
    #   NOTE: body is a list of strings (one string per document to translate).
    #   Response is a list; first item has .translations[0].text for the
    #   English translation.
    # TODO: Step 3.5 - Return dict with detected_language, confidence,
    #   original_text, translated_text, was_translated
    #   If already English: translated_text = original_text, was_translated = False
    raise NotImplementedError("Implement detect_and_translate in Step 3")


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
        # TODO: Step 4.1 - Get the CLU client using _get_clu_client()
        # TODO: Step 4.2 - Build the analysis input dict:
        #   {
        #       "kind": "Conversation",
        #       "analysisInput": {
        #           "conversationItem": {
        #               "id": "1",
        #               "text": text,
        #               "participantId": "user",
        #           }
        #       },
        #       "parameters": {
        #           "projectName": clu_project,
        #           "deploymentName": clu_deployment,
        #           "stringIndexType": "TextElement_V8",
        #       },
        #   }
        # TODO: Step 4.3 - Call client.analyze_conversation(task)
        #   The response contains .result.prediction with:
        #     .top_intent (str), .intents (list with .category and
        #     .confidence_score), .entities (list with .category, .text)
        # TODO: Step 4.4 - Extract top_intent and map to pipeline format
        #   The CLU model uses PascalCase intents (ReportIssue, CheckStatus,
        #   GetInformation) but our pipeline uses kebab-case. Use _CLU_INTENT_MAP
        #   to convert:
        #     raw_intent = prediction.top_intent
        #     top_intent = _CLU_INTENT_MAP.get(raw_intent, raw_intent)
        # TODO: Step 4.5 - Extract confidence and entities, return dict
        #   Return dict with top_intent (mapped), confidence, entities
        raise NotImplementedError("Implement recognize_intent in Step 4")
    except NotImplementedError:
        raise
    except Exception:
        # CLU call failed — fall back to keyword matching
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
