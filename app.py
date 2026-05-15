from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_INDEX = DATA_DIR / "index.json"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could",
    "did", "do", "does", "doing", "for", "from", "get", "got", "had", "has", "have",
    "he", "her", "hers", "him", "his", "how", "i", "if", "im", "in", "is", "it",
    "its", "just", "like", "me", "my", "of", "on", "or", "our", "really", "so",
    "that", "the", "their", "them", "then", "there", "they", "this", "to", "too",
    "very", "was", "we", "well", "were", "what", "when", "where", "who", "with",
    "would", "you", "your", "youre", "about", "also", "dont", "ive", "ill", "thats",
    "thanks", "thank", "hello", "hi", "hey", "good", "great", "nice", "awesome",
}

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z']+")
MESSAGE_RE = re.compile(r"(?m)^(User\s+\d+):\s*(.*)$")

INTENT_LABELS = ["reminder", "emotional-support", "action-item", "small-talk", "unknown"]

POSITIVE_WORDS = {
    "amazing", "awesome", "best", "calm", "cool", "delicious", "enjoy", "excited",
    "fun", "glad", "good", "great", "happy", "hope", "love", "nice", "positive",
    "relax", "thanks", "wonderful",
}
NEGATIVE_WORDS = {
    "angry", "anxious", "bad", "busy", "daunting", "difficult", "frustrated", "hard",
    "hate", "miss", "nervous", "overwhelmed", "sad", "scared", "sorry", "stress",
    "stressed", "tired", "worried",
}
EMOTIONAL_WORDS = POSITIVE_WORDS | NEGATIVE_WORDS | {
    "feel", "feeling", "hurt", "lonely", "proud", "upset", "cry", "afraid",
}


@dataclass
class Message:
    id: int
    row: int
    speaker: str
    text: str


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in TOKEN_RE.findall(text.lower().replace("'", "")):
        if raw not in STOPWORDS and len(raw) > 2:
            tokens.append(raw)
    return tokens


def cosine_counts(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def parse_csv(csv_path: Path) -> list[Message]:
    messages: list[Message] = []
    with csv_path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        for row_idx, row in enumerate(csv.reader(f), start=1):
            if not row:
                continue
            blob = row[0]
            for match in MESSAGE_RE.finditer(blob):
                text = " ".join(match.group(2).strip().split())
                if text:
                    messages.append(Message(len(messages) + 1, row_idx, match.group(1), text))
    return messages


def summarize_messages(messages: list[Message], max_sentences: int = 4) -> str:
    if not messages:
        return ""
    corpus_counts = Counter()
    for msg in messages:
        corpus_counts.update(tokenize(msg.text))
    top_terms = [term for term, _ in corpus_counts.most_common(10)]

    scored: list[tuple[float, int, Message]] = []
    for idx, msg in enumerate(messages):
        toks = tokenize(msg.text)
        if not toks:
            score = 0.0
        else:
            score = sum(corpus_counts[t] for t in toks) / math.sqrt(len(toks))
            if any(marker in msg.text.lower() for marker in ("i am", "i'm", "i love", "i enjoy", "i work", "i study")):
                score *= 1.15
        scored.append((score, idx, msg))

    chosen = sorted(scored, reverse=True)[:max_sentences]
    chosen = sorted(chosen, key=lambda item: item[1])
    lines = []
    if top_terms:
        lines.append("Main signals: " + ", ".join(top_terms[:7]) + ".")
    for _, _, msg in chosen:
        lines.append(f"{msg.speaker}: {msg.text}")
    return " ".join(lines)


def build_topic_checkpoints(
    messages: list[Message],
    threshold: float = 0.07,
    min_topic_messages: int = 10,
    max_topic_messages: int = 80,
) -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    segment: list[Message] = []
    profile: Counter[str] = Counter()
    recent_scores: deque[float] = deque(maxlen=3)

    def flush() -> None:
        nonlocal segment, profile, recent_scores
        if not segment:
            return
        terms = Counter()
        for item in segment:
            terms.update(tokenize(item.text))
        checkpoints.append(
            {
                "topic_id": len(checkpoints) + 1,
                "start_message": segment[0].id,
                "end_message": segment[-1].id,
                "start_row": segment[0].row,
                "end_row": segment[-1].row,
                "message_count": len(segment),
                "keywords": [term for term, _ in terms.most_common(12)],
                "summary": summarize_messages(segment),
                "evidence": [f"{m.speaker}: {m.text}" for m in segment[:2] + segment[-2:]],
            }
        )
        segment = []
        profile = Counter()
        recent_scores = deque(maxlen=3)

    for msg in messages:
        msg_counts = Counter(tokenize(msg.text))
        score = cosine_counts(msg_counts, profile)
        recent_scores.append(score)
        can_split = len(segment) >= min_topic_messages
        low_recent = len(recent_scores) == 3 and statistics.mean(recent_scores) < threshold
        forced = len(segment) >= max_topic_messages

        if segment and can_split and (low_recent or forced):
            flush()

        segment.append(msg)
        profile.update(msg_counts)
        if len(segment) > 35:
            # Let old context decay so later turns can move the topic.
            profile = Counter(dict(profile.most_common(80)))

    flush()
    return checkpoints


def build_hundred_checkpoints(messages: list[Message]) -> list[dict[str, Any]]:
    checkpoints = []
    for start in range(0, len(messages), 100):
        segment = messages[start : start + 100]
        checkpoints.append(
            {
                "checkpoint_id": len(checkpoints) + 1,
                "start_message": segment[0].id,
                "end_message": segment[-1].id,
                "message_count": len(segment),
                "summary": summarize_messages(segment, max_sentences=5),
            }
        )
    return checkpoints


def build_chunks(messages: list[Message], size: int = 24, overlap: int = 6) -> list[dict[str, Any]]:
    chunks = []
    step = max(1, size - overlap)
    for start in range(0, len(messages), step):
        segment = messages[start : start + size]
        if not segment:
            continue
        chunks.append(
            {
                "chunk_id": len(chunks) + 1,
                "start_message": segment[0].id,
                "end_message": segment[-1].id,
                "text": "\n".join(f"{m.speaker}: {m.text}" for m in segment),
                "summary": summarize_messages(segment, max_sentences=3),
            }
        )
    return chunks


def add_evidence(bucket: dict[str, dict[str, Any]], key: str, value: str, evidence: str) -> None:
    item = bucket.setdefault(key, {"label": value, "count": 0, "evidence": []})
    item["count"] += 1
    if len(item["evidence"]) < 5 and evidence not in item["evidence"]:
        item["evidence"].append(evidence)


def extract_persona(messages: list[Message]) -> dict[str, Any]:
    habits: dict[str, dict[str, Any]] = {}
    facts: dict[str, dict[str, Any]] = {}
    traits: dict[str, dict[str, Any]] = {}
    user_texts = [m.text for m in messages if m.speaker == "User 1"]
    lengths = [len(t.split()) for t in user_texts]
    questions = exclamations = emojis = 0

    fact_patterns = [
        (re.compile(r"\b(?:i am|i'm)\s+(?:a|an)\s+([^.!?]{3,80})", re.I), "identity_or_role"),
        (re.compile(r"\bi work (?:as|at|in)\s+([^.!?]{3,80})", re.I), "work"),
        (re.compile(r"\bi (?:study|studying|am studying)\s+([^.!?]{3,80})", re.I), "education"),
        (re.compile(r"\bi live in\s+([^.!?]{3,80})", re.I), "location"),
        (re.compile(r"\bi(?:'m| am) moving to\s+([^.!?]{3,80})", re.I), "life_event"),
        (re.compile(r"\bmy (?:wife|husband|mom|mother|dad|father|parents|brother|sister|family|kids|children)\b[^.!?]*", re.I), "relationships"),
    ]
    habit_patterns = [
        (re.compile(r"\bi (?:like|love|enjoy)\s+([^.!?]{3,80})", re.I), "likes_or_enjoys"),
        (re.compile(r"\bi usually\s+([^.!?]{3,80})", re.I), "usually"),
        (re.compile(r"\bi always\s+([^.!?]{3,80})", re.I), "always"),
        (re.compile(r"\bi often\s+([^.!?]{3,80})", re.I), "often"),
        (re.compile(r"\bi (?:play|run|cook|read|hike|travel|write|paint|draw|sing)\b[^.!?]*", re.I), "activities"),
    ]

    for text in user_texts:
        lower = text.lower()
        questions += text.count("?")
        exclamations += text.count("!")
        emojis += len(re.findall(r"[\U0001F300-\U0001FAFF]", text))

        for pattern, key in fact_patterns:
            for match in pattern.finditer(text):
                value = match.group(0).strip()
                if is_low_value_fact(value):
                    continue
                add_evidence(facts, key + ":" + value.lower()[:45], value, text)

        for pattern, key in habit_patterns:
            for match in pattern.finditer(text):
                value = match.group(0).strip()
                add_evidence(habits, key + ":" + value.lower()[:45], value, text)

        content_len = len(tokenize(text))
        is_simple_greeting = lower.strip(" !?.") in {"hi", "hello", "hey", "hi there", "hello there"}

        if any(word in lower for word in ("haha", "lol", "funny", "joke")) and content_len >= 3:
            add_evidence(traits, "humorous", "Shows humor or playful phrasing", text)
        if (
            any(word in lower for word in ("excited", "amazing", "awesome", "love", "wonderful"))
            or ("!" in text and content_len >= 4 and not is_simple_greeting)
        ):
            add_evidence(traits, "enthusiastic", "Often sounds positive or enthusiastic", text)
        if any(word in lower for word in ("i feel sad", "i'm nervous", "i am nervous", "i'm worried", "i am worried", "i'm scared", "i am scared", "i miss", "stressed")):
            add_evidence(traits, "emotionally_open", "Shares emotional states directly", text)
        if "?" in text and content_len >= 3:
            add_evidence(traits, "curious", "Asks follow-up questions", text)

    avg_len = round(statistics.mean(lengths), 2) if lengths else 0
    style = {
        "average_words_per_user1_message": avg_len,
        "question_marks": questions,
        "exclamation_marks": exclamations,
        "emoji_count": emojis,
        "style_observations": [],
    }
    if avg_len < 9:
        style["style_observations"].append("Mostly short messages.")
    elif avg_len < 20:
        style["style_observations"].append("Mostly concise conversational messages.")
    else:
        style["style_observations"].append("Often writes detailed messages.")
    if questions > len(user_texts) * 0.25:
        style["style_observations"].append("Frequently asks questions and keeps dialogue moving.")
    if exclamations > len(user_texts) * 0.15:
        style["style_observations"].append("Uses exclamation marks often, giving an upbeat tone.")
    if emojis:
        style["style_observations"].append("Uses emoji in some messages.")
    else:
        style["style_observations"].append("Emoji usage is minimal or absent in this dataset.")

    return {
        "subject": "User 1",
        "message_count_analyzed": len(user_texts),
        "habits": sorted(habits.values(), key=lambda x: x["count"], reverse=True)[:40],
        "personal_facts": sorted(facts.values(), key=lambda x: x["count"], reverse=True)[:40],
        "personality_traits": sorted(traits.values(), key=lambda x: x["count"], reverse=True),
        "communication_style": style,
        "note": "All fields are based on direct User 1 conversation signals and stored evidence snippets.",
    }


def tone_features(texts: list[str]) -> dict[str, Any]:
    joined = " ".join(texts)
    tokens = tokenize(joined)
    token_count = max(len(tokens), 1)
    lower = joined.lower()
    pos = sum(1 for token in tokens if token in POSITIVE_WORDS)
    neg = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    questions = joined.count("?")
    exclamations = joined.count("!")
    emojis = len(re.findall(r"[\U0001F300-\U0001FAFF]", joined))
    casual = sum(lower.count(marker) for marker in ("lol", "haha", "hey", "yeah", "cool", "awesome"))
    formal = sum(lower.count(marker) for marker in ("thank you", "thanks", "appreciate", "please", "certainly"))

    sentiment = (pos - neg) / token_count
    question_rate = questions / max(len(texts), 1)
    exclamation_rate = exclamations / max(len(texts), 1)
    emotion_rate = sum(1 for token in tokens if token in EMOTIONAL_WORDS) / token_count

    tones = []
    if question_rate >= 0.45:
        tones.append("curious")
    if sentiment <= -0.015 or neg >= pos + 2:
        tones.append("frustrated" if any(w in lower for w in ("frustrated", "stress", "stressed", "hard", "tired")) else "concerned")
    if sentiment >= 0.018 or pos >= neg + 3:
        tones.append("positive")
    if casual >= formal + 2 or exclamation_rate >= 0.45:
        tones.append("casual")
    if formal >= casual + 2:
        tones.append("formal")
    if emotion_rate >= 0.06:
        tones.append("emotional")
    if not tones:
        tones.append("neutral")

    return {
        "tone": " & ".join(tones[:3]),
        "sentiment_score": round(sentiment, 4),
        "question_rate": round(question_rate, 3),
        "exclamation_rate": round(exclamation_rate, 3),
        "emotion_rate": round(emotion_rate, 4),
        "positive_terms": pos,
        "negative_terms": neg,
        "casual_markers": casual,
        "formal_markers": formal,
    }


def infer_trigger(texts: list[str], previous_keywords: list[str]) -> dict[str, Any]:
    joined = " ".join(texts)
    tokens = tokenize(joined)
    terms = [term for term, _ in Counter(tokens).most_common(8)]
    new_terms = [term for term in terms if term not in previous_keywords[:12]]
    person_match = re.search(
        r"\b(?:my|your)\s+(mom|mother|dad|father|parents|brother|sister|wife|husband|family|friend|kids|children|son|daughter)\b",
        joined,
        re.I,
    )
    event_match = re.search(
        r"\b(moving|job|work|school|college|study|studying|birthday|vacation|travel|wedding|interview|exam|project|deadline)\b",
        joined,
        re.I,
    )
    if person_match:
        kind = "person"
        label = person_match.group(0)
    elif event_match:
        kind = "event"
        label = event_match.group(1)
    elif new_terms:
        kind = "topic"
        label = ", ".join(new_terms[:3])
    else:
        kind = "tone"
        label = "change in message tone"
    return {"type": kind, "label": label, "keywords": terms[:8]}


def build_persona_drift(messages: list[Message]) -> list[dict[str, Any]]:
    rows: dict[int, list[Message]] = defaultdict(list)
    for msg in messages:
        if msg.speaker == "User 1":
            rows[msg.row].append(msg)

    timeline = []
    previous: dict[str, Any] | None = None
    previous_keywords: list[str] = []
    for row in sorted(rows):
        day_messages = rows[row]
        texts = [m.text for m in day_messages]
        features = tone_features(texts)
        trigger = infer_trigger(texts, previous_keywords)
        drift = previous is None
        drift_reasons = []
        if previous is not None:
            if features["tone"] != previous["tone"]:
                drift = True
                drift_reasons.append(f"tone changed from {previous['tone']} to {features['tone']}")
            sentiment_delta = features["sentiment_score"] - previous["sentiment_score"]
            if abs(sentiment_delta) >= 0.025:
                drift = True
                direction = "more positive" if sentiment_delta > 0 else "more negative"
                drift_reasons.append(f"sentiment became {direction}")
            if trigger["keywords"][:3] != previous_keywords[:3]:
                drift_reasons.append("main keywords changed")
        timeline.append(
            {
                "day": row,
                "start_message": day_messages[0].id,
                "end_message": day_messages[-1].id,
                "message_count": len(day_messages),
                "mood_tone": features["tone"],
                "features": features,
                "drift_from_previous_day": drift,
                "drift_reasons": drift_reasons or ["baseline day"],
                "trigger": trigger,
                "evidence": [m.text for m in day_messages[:3]],
            }
        )
        previous = features
        previous_keywords = trigger["keywords"]
    return timeline


def intent_seed_label(text: str) -> str:
    lower = text.lower()
    if re.search(r"\b(remind me|remember to|don't let me forget|dont let me forget|tomorrow|tonight|later|next week|alarm)\b", lower):
        return "reminder"
    if re.search(r"\b(i feel|i'm sad|i am sad|worried|nervous|scared|stressed|overwhelmed|miss|sorry to hear|hope you feel)\b", lower):
        return "emotional-support"
    if re.search(r"\b(need to|have to|should|can you|could you|please|let's|todo|to do|plan|schedule|book|call|send|finish)\b", lower):
        return "action-item"
    if re.search(r"\b(hi|hello|hey|how are you|what do you do for fun|favorite|hobby|hobbies|nice talking)\b", lower):
        return "small-talk"
    return "unknown"


def train_intent_classifier(messages: list[Message]) -> dict[str, Any]:
    docs: list[tuple[str, str]] = []
    for msg in messages:
        if msg.speaker != "User 1":
            continue
        label = intent_seed_label(msg.text)
        docs.append((label, msg.text))

    priors = {label: 1 for label in INTENT_LABELS}
    token_counts = {label: Counter() for label in INTENT_LABELS}
    totals = {label: 0 for label in INTENT_LABELS}
    vocab: Counter[str] = Counter()
    examples = {label: [] for label in INTENT_LABELS}

    for label, text in docs:
        priors[label] += 1
        tokens = tokenize(text)
        vocab.update(tokens)
        token_counts[label].update(tokens)
        totals[label] += len(tokens)
        if len(examples[label]) < 5:
            examples[label].append(text)

    trimmed_vocab = {term for term, _ in vocab.most_common(3500)}
    model_counts = {}
    for label in INTENT_LABELS:
        model_counts[label] = {term: count for term, count in token_counts[label].items() if term in trimmed_vocab}
        totals[label] = sum(model_counts[label].values())

    return {
        "type": "multinomial_naive_bayes",
        "labels": INTENT_LABELS,
        "max_vocab": len(trimmed_vocab),
        "trained_from": "conversation messages with deterministic weak labels; no external API",
        "priors": priors,
        "token_counts": model_counts,
        "token_totals": totals,
        "examples": examples,
    }


class IntentClassifier:
    def __init__(self, model: dict[str, Any]):
        self.model = model
        self.labels = model.get("labels", INTENT_LABELS)
        self.priors = model.get("priors", {})
        self.token_counts = model.get("token_counts", {})
        self.token_totals = model.get("token_totals", {})
        vocab = set()
        for counts in self.token_counts.values():
            vocab.update(counts)
        self.vocab_size = max(len(vocab), 1)
        self.total_docs = max(sum(self.priors.values()), 1)

    def predict(self, text: str) -> dict[str, Any]:
        rule_label = intent_seed_label(text)
        tokens = tokenize(text)
        scores = {}
        for label in self.labels:
            logp = math.log(self.priors.get(label, 1) / self.total_docs)
            denom = self.token_totals.get(label, 0) + self.vocab_size
            counts = self.token_counts.get(label, {})
            for token in tokens:
                logp += math.log((counts.get(token, 0) + 1) / denom)
            if rule_label == label:
                logp += 1.25
            scores[label] = logp
        best = max(scores, key=scores.get)
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else 0.0
        if margin < 0.15 and rule_label == "unknown":
            best = "unknown"
        return {
            "intent": best,
            "confidence_margin": round(margin, 4),
            "scores": {label: round(score, 4) for label, score in ordered},
            "latency_target": "CPU-only and designed for sub-200ms single-message classification",
        }


def is_low_value_fact(value: str) -> bool:
    lower = value.lower()
    blocked = (
        "i'm glad",
        "i am glad",
        "i'm sorry",
        "i am sorry",
        "i'm sure",
        "i am sure",
        "i'm doing",
        "i am doing",
        "i'm happy to",
        "i am happy to",
    )
    if lower.startswith(blocked):
        return True
    return len(tokenize(value)) < 2


def build_vocab(docs: list[str]) -> dict[str, float]:
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(tokenize(doc)))
    total = max(len(docs), 1)
    return {term: math.log((1 + total) / (1 + freq)) + 1 for term, freq in df.items()}


def vectorize(text: str, idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokenize(text))
    if not counts:
        return {}
    vec = {term: (1 + math.log(count)) * idf.get(term, 1.0) for term, count in counts.items()}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    return {term: value / norm for term, value in vec.items()} if norm else vec


def vector_score(q_vec: dict[str, float], doc_vec: dict[str, float]) -> float:
    return sum(value * doc_vec.get(term, 0.0) for term, value in q_vec.items())


class RagEngine:
    def __init__(self, index: dict[str, Any]):
        self.index = index
        docs = [t["summary"] + " " + " ".join(t.get("keywords", [])) for t in index["topic_checkpoints"]]
        docs += [c["text"] + " " + c["summary"] for c in index["chunks"]]
        self.idf = build_vocab(docs)
        self.topic_vectors = [
            vectorize(t["summary"] + " " + " ".join(t.get("keywords", [])), self.idf)
            for t in index["topic_checkpoints"]
        ]
        self.chunk_vectors = [vectorize(c["text"] + " " + c["summary"], self.idf) for c in index["chunks"]]
        self.intent_classifier = IntentClassifier(index.get("intent_model", train_intent_classifier([])))

    def search(self, query: str, top_topics: int = 4, top_chunks: int = 5) -> dict[str, Any]:
        q_vec = vectorize(query, self.idf)
        topic_hits = sorted(
            ((vector_score(q_vec, vec), item) for vec, item in zip(self.topic_vectors, self.index["topic_checkpoints"])),
            key=lambda x: x[0],
            reverse=True,
        )[:top_topics]
        chunk_hits = sorted(
            ((vector_score(q_vec, vec), item) for vec, item in zip(self.chunk_vectors, self.index["chunks"])),
            key=lambda x: x[0],
            reverse=True,
        )[:top_chunks]
        return {
            "topics": [{"score": round(score, 4), **item} for score, item in topic_hits if score > 0],
            "chunks": [{"score": round(score, 4), **item} for score, item in chunk_hits if score > 0],
        }

    def answer(self, query: str) -> dict[str, Any]:
        q = query.lower()
        if self.is_conflict_query(q):
            return self.resolve_conflict_query(query)
        if "intent" in q or q.startswith("classify:"):
            text = query.split(":", 1)[1].strip() if ":" in query else query
            prediction = self.intent_classifier.predict(text)
            return {
                "answer": f"Offline intent classifier: {prediction['intent']} (margin {prediction['confidence_margin']}).",
                "retrieval": {"topics": [], "chunks": []},
                "persona": self.index["persona"],
                "intent": prediction,
            }
        if any(word in q for word in ("drift", "mood", "tone timeline", "timeline")):
            timeline = self.index.get("persona_drift", [])[:8]
            lines = [
                f"Day {item['day']} -> {item['mood_tone']} | trigger: {item['trigger']['type']}={item['trigger']['label']}"
                for item in timeline
            ]
            return {
                "answer": "Persona drift timeline:\n" + "\n".join(lines),
                "retrieval": {"topics": [], "chunks": []},
                "persona": self.index["persona"],
                "persona_drift": timeline,
            }
        hits = self.search(query)
        persona = self.index["persona"]
        parts = []

        if any(word in q for word in ("habit", "routine", "likes", "enjoy")):
            parts.append("Habits and preferences: " + render_persona_items(persona.get("habits", [])[:8]))
        elif any(word in q for word in ("talk", "communicat", "style", "tone", "message")):
            style = persona["communication_style"]
            parts.append(
                "Communication style: "
                + " ".join(style["style_observations"])
                + f" Average User 1 message length is {style['average_words_per_user1_message']} words."
            )
        elif any(word in q for word in ("person", "persona", "traits", "personality", "kind of")):
            parts.append("Persona traits: " + render_persona_items(persona.get("personality_traits", [])[:8]))
            if persona.get("habits"):
                parts.append("Common interests or habits: " + render_persona_items(persona["habits"][:5]))
            if persona.get("personal_facts"):
                parts.append("Personal facts found: " + render_persona_items(persona["personal_facts"][:5]))
        else:
            parts.append("Relevant conversation evidence was retrieved from topic checkpoints and message chunks.")

        if hits["topics"]:
            topic_lines = []
            for topic in hits["topics"][:3]:
                topic_lines.append(
                    f"Topic {topic['topic_id']} (messages {topic['start_message']}-{topic['end_message']}): {topic['summary']}"
                )
            parts.append("Relevant topic summaries: " + " ".join(topic_lines))

        if hits["chunks"]:
            chunk_lines = []
            for chunk in hits["chunks"][:2]:
                chunk_lines.append(
                    f"Messages {chunk['start_message']}-{chunk['end_message']}: {chunk['summary']}"
                )
            parts.append("Relevant message chunks: " + " ".join(chunk_lines))

        return {"answer": "\n\n".join(parts), "retrieval": hits, "persona": persona}

    def is_conflict_query(self, q: str) -> bool:
        family_terms = ("sister", "brother", "mother", "mom", "father", "dad", "parents", "family")
        return any(term in q for term in family_terms) and any(
            marker in q for marker in ("mention", "anything", "did i", "what did", "contradict", "conflict")
        )

    def resolve_conflict_query(self, query: str) -> dict[str, Any]:
        family_terms = [
            "sister", "brother", "mother", "mom", "father", "dad", "parents",
            "family", "wife", "husband", "son", "daughter", "kids", "children",
        ]
        query_tokens = tokenize(query)
        exact_family_terms = [term for term in family_terms if term in query_tokens]
        terms = exact_family_terms or [token for token in query_tokens if token not in {"mention", "anything"}]
        hits = self.search(query, top_topics=8, top_chunks=16)
        max_msg = max((chunk["end_message"] for chunk in self.index["chunks"]), default=1)
        ranked = []
        for chunk in hits["chunks"]:
            text = chunk["text"]
            relevance = chunk.get("score", 0.0)
            recency = chunk["end_message"] / max_msg
            emotional = emotional_weight(text)
            exact_hits = sum(len(re.findall(rf"\b{re.escape(term)}\b", text, re.I)) for term in terms)
            if exact_family_terms and exact_hits == 0:
                continue
            term_boost = min(exact_hits, 6) * 0.12
            score = relevance + (0.25 * recency) + (0.2 * emotional) + term_boost
            ranked.append((score, recency, emotional, chunk))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[:6]
        contradiction = detect_contradictions([item[3]["text"] for item in selected], terms)

        evidence_lines = []
        for score, recency, emotional, chunk in selected[:4]:
            evidence_lines.append(
                f"Messages {chunk['start_message']}-{chunk['end_message']} "
                f"(rank {score:.3f}, recency {recency:.2f}, emotional {emotional:.2f}): {chunk['summary']}"
            )

        if selected:
            answer = (
                "Yes. I found relevant mentions and ranked them by lexical relevance, recency, and emotional weight. "
                "Because the same family term appears in different contexts, I would treat the result as a merged view "
                "instead of a single fact."
            )
        else:
            answer = "I did not find strong matching evidence for that family term in the indexed chunks."
        if contradiction["has_contradiction"]:
            answer += " Contradictory signals were flagged, so the safest answer should preserve the uncertainty."
        else:
            answer += " No direct contradiction was flagged in the top-ranked chunks, but multiple contexts are still kept separate."
        answer += "\n\nTop evidence:\n" + "\n".join(evidence_lines)

        return {
            "answer": answer,
            "retrieval": {"topics": hits["topics"], "chunks": [item[3] for item in selected]},
            "persona": self.index["persona"],
            "conflict_resolution": {
                "query_terms": terms,
                "ranking_formula": "tf-idf relevance + 0.25*recency + 0.20*emotional_weight + term boost",
                "contradiction": contradiction,
            },
        }


def render_persona_items(items: list[dict[str, Any]]) -> str:
    if not items:
        return "No strong direct signals found."
    rendered = []
    for item in items:
        evidence = item.get("evidence", [])
        tail = f" Evidence: \"{evidence[0]}\"" if evidence else ""
        rendered.append(f"{item['label']} (count {item['count']}).{tail}")
    return " ".join(rendered)


def emotional_weight(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    emotional = sum(1 for token in tokens if token in EMOTIONAL_WORDS)
    punctuation = min(text.count("!") + text.count("?"), 10) / 10
    return min(1.0, (emotional / len(tokens)) * 8 + punctuation * 0.25)


def detect_contradictions(texts: list[str], terms: list[str]) -> dict[str, Any]:
    positive_claims = []
    negative_claims = []
    other_claims = []
    negation_re = re.compile(r"\b(no|not|never|don't|dont|didn't|didnt|without|no longer|used to)\b", re.I)
    strong_positive_re = re.compile(r"\b(love|close|support|best|great|happy|glad|miss)\b", re.I)
    strong_negative_re = re.compile(r"\b(hate|avoid|argue|angry|upset|hard|difficult|not close|no contact)\b", re.I)

    for text in texts:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            if not any(re.search(rf"\b{re.escape(term)}\b", sentence, re.I) for term in terms):
                continue
            compact = sentence.strip()
            if negation_re.search(compact) or strong_negative_re.search(compact):
                negative_claims.append(compact)
            elif strong_positive_re.search(compact):
                positive_claims.append(compact)
            else:
                other_claims.append(compact)

    has_contradiction = bool(positive_claims and negative_claims)
    return {
        "has_contradiction": has_contradiction,
        "positive_or_supportive_claims": positive_claims[:5],
        "negative_or_negated_claims": negative_claims[:5],
        "neutral_claims": other_claims[:5],
        "note": "Contradiction is heuristic: opposite emotional/negated claims around the same query term are flagged for human-readable merging.",
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_index(args: argparse.Namespace) -> dict[str, Any]:
    csv_path = Path(args.csv).expanduser()
    messages = parse_csv(csv_path)
    DATA_DIR.mkdir(exist_ok=True)
    with (DATA_DIR / "messages.jsonl").open("w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(asdict(msg), ensure_ascii=False) + "\n")

    topics = build_topic_checkpoints(
        messages,
        threshold=args.topic_threshold,
        min_topic_messages=args.min_topic_messages,
        max_topic_messages=args.max_topic_messages,
    )
    hundred = build_hundred_checkpoints(messages)
    chunks = build_chunks(messages)
    persona = extract_persona(messages)
    persona_drift = build_persona_drift(messages)
    intent_model = train_intent_classifier(messages)
    index = {
        "source_csv": str(csv_path),
        "message_count": len(messages),
        "topic_count": len(topics),
        "chunk_count": len(chunks),
        "topic_checkpoints": topics,
        "hundred_checkpoints": hundred,
        "chunks": chunks,
        "persona": persona,
        "persona_drift": persona_drift,
        "intent_model": intent_model,
    }
    write_json(DATA_DIR / "topic_checkpoints.json", topics)
    write_json(DATA_DIR / "hundred_checkpoints.json", hundred)
    write_json(DATA_DIR / "chunks.json", chunks)
    write_json(DATA_DIR / "persona.json", persona)
    write_json(DATA_DIR / "persona_drift.json", persona_drift)
    write_json(DATA_DIR / "intent_model.json", intent_model)
    write_json(DEFAULT_INDEX, index)
    return index


def load_index() -> dict[str, Any]:
    if not DEFAULT_INDEX.exists():
        raise SystemExit("No data/index.json found. Run with --build first.")
    return json.loads(DEFAULT_INDEX.read_text(encoding="utf-8"))


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Conversation Persona RAG</title>
  <style>
    :root { color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; }
    body { margin: 0; background: #f7f6f2; color: #1d2730; }
    main { max-width: 1040px; margin: 0 auto; padding: 28px 18px 40px; }
    header { display: flex; justify-content: space-between; gap: 16px; align-items: end; margin-bottom: 18px; }
    h1 { font-size: 28px; margin: 0 0 6px; letter-spacing: 0; }
    p { line-height: 1.5; }
    .meta { color: #52616b; margin: 0; }
    .panel { background: #fff; border: 1px solid #ddd8cc; border-radius: 8px; padding: 16px; }
    .ask { display: grid; grid-template-columns: 1fr auto; gap: 10px; margin-bottom: 14px; }
    input { font: inherit; padding: 12px 14px; border: 1px solid #b9c1c8; border-radius: 6px; }
    button { font: inherit; padding: 12px 16px; border: 0; border-radius: 6px; background: #22577a; color: #fff; cursor: pointer; }
    button:hover { background: #17425e; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
    .chips button { background: #e9f1f5; color: #1d394c; padding: 8px 10px; }
    #answer { white-space: pre-wrap; line-height: 1.55; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px; }
    h2 { font-size: 16px; margin: 0 0 10px; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f3f5f6; padding: 12px; border-radius: 6px; max-height: 360px; overflow: auto; }
    @media (max-width: 760px) { header, .ask, .grid { grid-template-columns: 1fr; display: grid; } }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Conversation Persona RAG</h1>
      <p class="meta" id="stats">Loading index stats...</p>
    </div>
  </header>
  <section class="panel">
    <div class="chips">
      <button data-q="What kind of person is this user?">What kind of person?</button>
      <button data-q="What are their habits?">Habits</button>
      <button data-q="How do they talk?">Communication style</button>
      <button data-q="Show persona drift timeline">Drift timeline</button>
      <button data-q="Did I mention anything about my sister?">Conflict RAG</button>
      <button data-q="classify: remind me to call my sister tomorrow">Intent</button>
    </div>
    <form class="ask" id="form">
      <input id="query" value="What kind of person is this user?" autocomplete="off">
      <button>Ask</button>
    </form>
    <div id="answer"></div>
  </section>
  <section class="grid">
    <div class="panel">
      <h2>Retrieved Topics</h2>
      <pre id="topics">Ask a question to see topic checkpoints.</pre>
    </div>
    <div class="panel">
      <h2>Retrieved Chunks</h2>
      <pre id="chunks">Ask a question to see message chunks.</pre>
    </div>
  </section>
</main>
<script>
async function ask(q) {
  document.querySelector('#answer').textContent = 'Thinking...';
  const res = await fetch('/api/ask?q=' + encodeURIComponent(q));
  const data = await res.json();
  document.querySelector('#answer').textContent = data.answer;
  document.querySelector('#topics').textContent = JSON.stringify(data.retrieval.topics, null, 2);
  document.querySelector('#chunks').textContent = JSON.stringify(data.retrieval.chunks, null, 2);
}
document.querySelector('#form').addEventListener('submit', ev => {
  ev.preventDefault();
  ask(document.querySelector('#query').value);
});
document.querySelectorAll('[data-q]').forEach(btn => btn.addEventListener('click', () => {
  document.querySelector('#query').value = btn.dataset.q;
  ask(btn.dataset.q);
}));
fetch('/api/stats').then(r => r.json()).then(s => {
  document.querySelector('#stats').textContent =
    `${s.message_count.toLocaleString()} messages | ${s.topic_count.toLocaleString()} topic checkpoints | ${s.chunk_count.toLocaleString()} chunks`;
});
ask(document.querySelector('#query').value);
</script>
</body>
</html>"""


def make_handler(engine: RagEngine) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, payload: Any) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                raw = HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif parsed.path == "/api/stats":
                self.send_json(
                    {
                        "message_count": engine.index["message_count"],
                        "topic_count": engine.index["topic_count"],
                        "chunk_count": engine.index["chunk_count"],
                        "drift_days": len(engine.index.get("persona_drift", [])),
                        "intent_labels": engine.index.get("intent_model", {}).get("labels", INTENT_LABELS),
                    }
                )
            elif parsed.path == "/api/ask":
                q = parse_qs(parsed.query).get("q", [""])[0]
                self.send_json(engine.answer(q))
            elif parsed.path == "/api/persona":
                self.send_json(engine.index["persona"])
            elif parsed.path == "/api/drift":
                self.send_json(engine.index.get("persona_drift", []))
            elif parsed.path == "/api/intent":
                text = parse_qs(parsed.query).get("text", [""])[0]
                self.send_json(engine.intent_classifier.predict(text))
            else:
                self.send_error(404, html.escape(parsed.path))

        def log_message(self, fmt: str, *args: Any) -> None:
            print("%s - %s" % (self.address_string(), fmt % args))

    return Handler


def serve(args: argparse.Namespace) -> None:
    engine = RagEngine(load_index())
    port = int(os.environ.get("PORT", args.port))
    server = ThreadingHTTPServer((args.host, port), make_handler(engine))
    print(f"Serving on http://{args.host}:{port}")
    server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local conversation RAG and persona chatbot.")
    parser.add_argument("--csv", default=str(ROOT / "conversations.csv"), help="Path to conversations.csv")
    parser.add_argument("--build", action="store_true", help="Build data/index.json from the CSV")
    parser.add_argument("--serve", action="store_true", help="Start the chatbot web server")
    parser.add_argument("--ask", help="Ask a question from the command line")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--topic-threshold", type=float, default=0.07)
    parser.add_argument("--min-topic-messages", type=int, default=10)
    parser.add_argument("--max-topic-messages", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.build:
        index = build_index(args)
        print(
            f"Built index: {index['message_count']} messages, "
            f"{index['topic_count']} topic checkpoints, {index['chunk_count']} chunks."
        )
    if args.ask:
        engine = RagEngine(load_index())
        print(engine.answer(args.ask)["answer"])
    if args.serve:
        serve(args)
    if not (args.build or args.ask or args.serve):
        print("Nothing to do. Use --build, --ask, or --serve.")


if __name__ == "__main__":
    main()
