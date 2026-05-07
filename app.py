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


def render_persona_items(items: list[dict[str, Any]]) -> str:
    if not items:
        return "No strong direct signals found."
    rendered = []
    for item in items:
        evidence = item.get("evidence", [])
        tail = f" Evidence: \"{evidence[0]}\"" if evidence else ""
        rendered.append(f"{item['label']} (count {item['count']}).{tail}")
    return " ".join(rendered)


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
    index = {
        "source_csv": str(csv_path),
        "message_count": len(messages),
        "topic_count": len(topics),
        "chunk_count": len(chunks),
        "topic_checkpoints": topics,
        "hundred_checkpoints": hundred,
        "chunks": chunks,
        "persona": persona,
    }
    write_json(DATA_DIR / "topic_checkpoints.json", topics)
    write_json(DATA_DIR / "hundred_checkpoints.json", hundred)
    write_json(DATA_DIR / "chunks.json", chunks)
    write_json(DATA_DIR / "persona.json", persona)
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
                    }
                )
            elif parsed.path == "/api/ask":
                q = parse_qs(parsed.query).get("q", [""])[0]
                self.send_json(engine.answer(q))
            elif parsed.path == "/api/persona":
                self.send_json(engine.index["persona"])
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
