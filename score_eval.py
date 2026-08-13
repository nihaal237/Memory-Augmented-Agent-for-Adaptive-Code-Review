import json
import time
from google import genai
from google.genai import errors as genai_errors
import os
from dotenv import load_dotenv

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

RESULTS_PATH = "evaluation_results.json"
SCORES_PATH = "evaluation_scores.json"


def score_review_pair(human_review: str, review_a: str, review_b: str, pr_title: str, max_retries=5) -> dict:
    """
    Score two AI reviews (A and B) against the same human review IN ONE CALL,
    forcing consistent calibration between them. A/B labels are used instead of
    'memory-on'/'memory-off' to avoid biasing the judge.
    """
    prompt = f"""You are evaluating two AI-generated code reviews (A and B) against what
real human reviewers actually said about the same pull request. Score them using the
SAME standards, directly comparing them to each other as well as to the human review.

PR Title: {pr_title}

Real human review comments (ground truth):
{human_review[:2000] if human_review.strip() else "(No substantive human comments available)"}

AI Review A:
{review_a[:2000]}

AI Review B:
{review_b[:2000]}

For EACH review (A and B), score 0-10 on:
1. "overlap": How well does it capture the SAME substantive issues the human reviewers raised?
2. "false_positives": Inverted score — 10 = no irrelevant/incorrect flags, 0 = mostly noise.
3. "actionability": 10 = specific and actionable, 0 = vague/generic.

Be consistent: if both reviews make a similar point, they should receive similar overlap
scores for that point, even if worded differently.

Respond ONLY with valid JSON, no markdown, no backticks:
{{
  "review_a": {{"overlap": <int 0-10>, "false_positives": <int 0-10>, "actionability": <int 0-10>}},
  "review_b": {{"overlap": <int 0-10>, "false_positives": <int 0-10>, "actionability": <int 0-10>}},
  "reasoning": "one or two sentences comparing A and B directly"
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

        except genai_errors.ServerError:
            wait = min(2 ** attempt * 5, 60)
            print(f"    ⏳ Gemini busy, waiting {wait}s...")
            time.sleep(wait)
        except json.JSONDecodeError:
            print(f"    ⚠️  Failed to parse score response, retrying...")
            time.sleep(2)

    print("    ❌ Gave up scoring after retries.")
    return {
        "review_a": {"overlap": None, "false_positives": None, "actionability": None},
        "review_b": {"overlap": None, "false_positives": None, "actionability": None},
        "reasoning": "SCORING_FAILED"
    }


def run_scoring():
    import random
    random.seed(42)

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    scores = []

    for i, item in enumerate(results):
        pr_number = item["pr_number"]
        print(f"[{i+1}/{len(results)}] Scoring PR #{pr_number}: {item['title'][:50]}")

        human_review = item["human_review"]

        # Randomize A/B assignment per PR to avoid any positional bias in the judge
        flip = random.choice([True, False])
        if flip:
            review_a, review_b = item["memory_on_review"], item["memory_off_review"]
            a_label, b_label = "memory_on", "memory_off"
        else:
            review_a, review_b = item["memory_off_review"], item["memory_on_review"]
            a_label, b_label = "memory_off", "memory_on"

        result = score_review_pair(human_review, review_a, review_b, item["title"])
        time.sleep(1)

        # Map back to memory_on / memory_off regardless of A/B order
        scores.append({
            "pr_number": pr_number,
            "title": item["title"],
            "memory_on_score": result["review_a"] if a_label == "memory_on" else result["review_b"],
            "memory_off_score": result["review_a"] if a_label == "memory_off" else result["review_b"],
            "reasoning": result.get("reasoning", "")
        })

        with open(SCORES_PATH, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Scoring complete. Saved to {SCORES_PATH}")
    return scores


def summarize_scores(scores):
    dims = ["overlap", "false_positives", "actionability"]
    on_totals = {d: [] for d in dims}
    off_totals = {d: [] for d in dims}

    for item in scores:
        on = item["memory_on_score"]
        off = item["memory_off_score"]
        for d in dims:
            if on.get(d) is not None:
                on_totals[d].append(on[d])
            if off.get(d) is not None:
                off_totals[d].append(off[d])

    print("\n=== SUMMARY: Memory-ON vs Memory-OFF (avg scores, 0-10) ===\n")
    print(f"{'Dimension':<18} {'Memory-ON':<12} {'Memory-OFF':<12} {'Difference':<10}")
    print("-" * 55)
    for d in dims:
        on_avg = sum(on_totals[d]) / len(on_totals[d]) if on_totals[d] else 0
        off_avg = sum(off_totals[d]) / len(off_totals[d]) if off_totals[d] else 0
        diff = on_avg - off_avg
        sign = "+" if diff >= 0 else ""
        print(f"{d:<18} {on_avg:<12.2f} {off_avg:<12.2f} {sign}{diff:.2f}")


if __name__ == "__main__":
    scores = run_scoring()
    summarize_scores(scores)