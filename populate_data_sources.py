import os
import json
import time
import sqlite3
import hashlib
from datetime import datetime, timezone
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# --- Setup connections to both memory stores ---
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_memory")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
review_memory_collection = chroma_client.get_or_create_collection(
    name="review_memory",
    embedding_function=embedding_fn
)

DB_PATH = "agent_memory.db"
PR_DATA_PATH = "pr_data_scikit-learn.json"
PROGRESS_FILE = "seed_progress.json"


def load_progress():
    """Returns the set of PR numbers already fully processed."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return set(json.load(f))
    return set()


def save_progress(processed_pr_numbers: set):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(list(processed_pr_numbers), f)


def make_stable_doc_id(pr_number: int, path: str, comment_body: str) -> str:
    """
    Creates a deterministic ID for a review comment, so the same comment
    always maps to the same Chroma ID, regardless of run order or reprocessing.
    """
    content_hash = hashlib.sha256(comment_body.encode("utf-8")).hexdigest()[:12]
    safe_path = path.replace("/", "_").replace(".", "_")[:50]
    return f"pr{pr_number}_{safe_path}_{content_hash}"


def extract_convention_from_comment(comment_body: str, diff_hunk: str, pr_title: str, max_retries=5) -> dict:
    """
    Asks Gemini if this review comment reflects a reusable convention.
    Retries with exponential backoff on transient server errors.
    """
    prompt = f"""You are analyzing a code review comment to determine if it reflects
a REUSABLE CODING CONVENTION (something that would apply to future, different code)
or a ONE-OFF NOTE (specific only to this exact change, not generalizable).

PR Title: {pr_title}

Code context (diff):
{diff_hunk[:500]}

Review comment:
{comment_body[:800]}

Respond ONLY with valid JSON, no markdown formatting, no backticks, in this exact format:
{{
  "is_convention": true or false,
  "convention_text": "a short, general, reusable rule IF is_convention is true, else empty string",
  "category": "one of: code-style, testing, performance, compatibility, documentation, error-handling, other"
}}"""

    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`")
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            return json.loads(raw_text)

        except genai_errors.ServerError as e:
            wait = min(2 ** attempt * 5, 60)  # 5s, 10s, 20s, 40s, 60s cap
            print(f"  ⏳ Gemini server busy (attempt {attempt+1}/{max_retries}), waiting {wait}s...")
            time.sleep(wait)

        except json.JSONDecodeError:
            print(f"  ⚠️  Failed to parse Gemini response, skipping this comment.")
            return {"is_convention": False, "convention_text": "", "category": "other"}

    print(f"  ❌ Gave up after {max_retries} retries, skipping this comment.")
    return {"is_convention": False, "convention_text": "", "category": "other"}

def is_same_convention_llm(text_a: str, text_b: str, max_retries=3) -> bool:
    """Ask Gemini to judge if two convention statements express the same underlying rule."""
    prompt = f"""Do these two statements express the SAME underlying coding convention or rule,
even if worded differently or at different levels of detail? Answer with only "yes" or "no".

Statement A: {text_a}
Statement B: {text_b}"""

    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text.strip().lower().startswith("yes")
        except genai_errors.ServerError:
            wait = min(2 ** attempt * 5, 30)
            print(f"    ⏳ Gemini busy during similarity check, waiting {wait}s...")
            time.sleep(wait)

    print("    ⚠️  Similarity check failed after retries, treating as NOT a match.")
    return False

def find_similar_convention(cursor, new_convention_text: str, top_k: int = 3, embedding_floor: float = 0.60):
    """
    Two-stage matching:
    1. Embedding similarity narrows down to the top_k closest existing conventions
       (fast, cheap pre-filter — embedding_floor is deliberately loose here).
    2. Gemini confirms whether any of those candidates actually express the
       SAME underlying rule (accurate, handles differing specificity/wording).
    Returns (existing_id, existing_text) if a true match is confirmed, else None.
    """
    cursor.execute("SELECT id, convention_text FROM conventions WHERE active = 1")
    existing_rows = cursor.fetchall()

    if not existing_rows:
        return None

    import numpy as np

    existing_texts = [row[1] for row in existing_rows]
    new_embedding = np.array(embedding_fn([new_convention_text])[0])
    existing_embeddings = embedding_fn(existing_texts)

    scored = []
    for (conv_id, text), emb in zip(existing_rows, existing_embeddings):
        existing_vec = np.array(emb)
        score = np.dot(new_embedding, existing_vec) / (
            np.linalg.norm(new_embedding) * np.linalg.norm(existing_vec)
        )
        if score >= embedding_floor:
            scored.append((score, conv_id, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    candidates = scored[:top_k]

    for score, conv_id, text in candidates:
        if is_same_convention_llm(new_convention_text, text):
            print(f"    🔎 LLM confirmed match (embedding score {score:.2f})")
            return (conv_id, text)

    return None


def save_convention(cursor, convention_text: str, pr_number: int, category: str):
    """Writes a new convention, or reinforces a semantically similar existing one."""
    now = datetime.now(timezone.utc).isoformat()

    similar = find_similar_convention(cursor, convention_text)

    if similar:
        conv_id, existing_text = similar
        cursor.execute(
            "SELECT times_confirmed FROM conventions WHERE id = ?", (conv_id,)
        )
        times_confirmed = cursor.fetchone()[0]
        cursor.execute(
            "UPDATE conventions SET times_confirmed = ?, last_confirmed_at = ? WHERE id = ?",
            (times_confirmed + 1, now, conv_id)
        )
        print(f"    🔗 Reinforced existing convention: \"{existing_text[:60]}...\"")
    else:
        cursor.execute(
            """INSERT INTO conventions 
               (convention_text, source_pr_number, category, created_at, last_confirmed_at) 
               VALUES (?, ?, ?, ?, ?)""",
            (convention_text, pr_number, category, now, now)
        )


def seed_memory(limit_prs: int = 30):
    """
    Main seeding loop. Processes PRs from pr_data_scikit-learn.json,
    writing raw comments to Chroma and extracted conventions to SQLite.
    Resumable and duplicate-safe across reruns.
    """
    with open(PR_DATA_PATH, "r", encoding="utf-8") as f:
        pr_data = json.load(f)

    already_processed = load_progress()
    print(f"Resuming: {len(already_processed)} PRs already processed, skipping those.\n")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    processed_comments = 0
    conventions_found = 0
    skipped_duplicates = 0

    for pr in pr_data[:limit_prs]:
        pr_number = pr["pr_number"]
        pr_title = pr["title"]

        if pr_number in already_processed:
            continue  # skip PRs we already fully completed

        for comment in pr["review_comments"]:
            comment_body = comment.get("body", "")
            diff_hunk = comment.get("diff_hunk", "") or ""
            path = comment.get("path", "")

            if not comment_body.strip():
                continue

            doc_id = make_stable_doc_id(pr_number, path, comment_body)

            # Check if this exact comment is already embedded — only affects
            # whether we add it to Chroma, NOT whether we extract a convention
            already_embedded = review_memory_collection.get(ids=[doc_id])
            if already_embedded["ids"]:
                skipped_duplicates += 1
            else:
                review_memory_collection.add(
                    documents=[comment_body],
                    metadatas=[{
                        "pr_number": pr_number,
                        "pr_title": pr_title,
                        "path": path,
                        "author": comment.get("author", "unknown")
                    }],
                    ids=[doc_id]
                )
                processed_comments += 1

            # Convention extraction always runs, regardless of Chroma dedup status
            result = extract_convention_from_comment(comment_body, diff_hunk, pr_title)

            if result.get("is_convention") and result.get("convention_text"):
                save_convention(
                    cursor,
                    result["convention_text"],
                    pr_number,
                    result.get("category", "other")
                )
                conventions_found += 1
                print(f"  📌 Convention found (PR #{pr_number}): {result['convention_text'][:80]}")

            conn.commit()
            time.sleep(0.5)  # gentle pacing to avoid hitting API rate limits

        already_processed.add(pr_number)
        save_progress(already_processed)  # save after EVERY PR, not just at the end
        print(f"Processed PR #{pr_number} ({pr_title[:50]})")

    conn.close()
    print(f"\n✅ Done. {processed_comments} new comments embedded, "
          f"{skipped_duplicates} duplicates skipped, {conventions_found} conventions extracted.")

if __name__ == "__main__":
    seed_memory(limit_prs=30)