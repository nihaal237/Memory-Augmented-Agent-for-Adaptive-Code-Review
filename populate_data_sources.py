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
from memory_utils import (
    gemini_client, chroma_client, embedding_fn, review_memory_collection,
    DB_PATH, make_stable_doc_id, save_convention
)

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