---
title: "Activity 5 Reflection - Constituent Services Hub"
type: reflection
version: "1.0.0"
---

# Activity 5 Reflection

Answer each question in 3-5 sentences. Thoughtful, specific responses earn full credit.

## 1. PII Redaction in Practice

What PII categories did the Azure AI Language service detect in the Memphis 311 complaints? Did you notice any false positives (non-PII flagged as sensitive) or false negatives (PII that was missed)? How might PII detection requirements differ between a government agency processing citizen complaints and a commercial customer service system?

## 2. Sentiment as a Routing Signal

How could sentiment analysis be used to prioritize Memphis 311 complaints — for example, routing highly negative complaints to senior staff? What are the risks of relying solely on sentiment for prioritization (consider complaints that are neutral in tone but urgent in nature)? How would you combine sentiment with other signals for a more robust routing system?

## 3. Multilingual Challenges

How accurately did the Language service detect the language of short versus long text samples? What challenges might arise with code-switching (mixing languages in a single message), which is common in multilingual communities? How would you handle a complaint where language detection confidence is low?

## 4. CLU vs. Keyword Matching

Compare the CLU model's intent classification with the keyword-based fallback. In what scenarios did CLU perform better, and where did keyword matching suffice? What are the trade-offs of training and maintaining a custom CLU model versus using a simpler rule-based approach for a city's 311 system? *(If CLU was not configured in your environment, discuss how you would expect it to differ based on the training data in `data/intent_examples.json`.)*

## 5. Pipeline Design

Which step in your NLP pipeline was most critical to get right first, and why? If you were deploying this pipeline for production use handling thousands of complaints daily, what monitoring, fallback mechanisms, or error handling would you add? How would you measure pipeline health over time?
