import os
import time
import sqlite3
import hashlib
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# --- Shared connections ---
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


def make_stable_doc_id(pr_number, path: str, comment_body: str) -> str:
    """Creates a deterministic ID for a review comment based on its content."""
    content_hash = hashlib.sha256(comment_body.encode("utf-8")).hexdigest()[:12]
    safe_path = (path or "").replace("/", "_").replace(".", "_")[:50]
    pr_part = str(pr_number) if pr_number is not None else "live"
    return f"pr{pr_part}_{safe_path}_{content_hash}"


def is_same_convention_llm(text_a: str, text_b: str, max_retries: int = 3) -> bool:
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
    Two-stage matching: embedding similarity narrows candidates, then Gemini
    confirms whether any actually express the same underlying rule.
    Returns (existing_id, existing_text) if confirmed, else None.
    """
    cursor.execute("SELECT id, convention_text FROM conventions WHERE active = 1")
    existing_rows = cursor.fetchall()

    if not existing_rows:
        return None

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


def save_convention(cursor, convention_text: str, pr_number, category: str):
    """Writes a new convention, or reinforces a semantically similar existing one."""
    now = datetime.now(timezone.utc).isoformat()

    similar = find_similar_convention(cursor, convention_text)

    if similar:
        conv_id, existing_text = similar
        cursor.execute("SELECT times_confirmed FROM conventions WHERE id = ?", (conv_id,))
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