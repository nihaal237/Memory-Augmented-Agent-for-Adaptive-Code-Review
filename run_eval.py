import json
import time
from agent_review_graph import review_pr
from agent_baseline import review_pr_baseline

TEST_SET_PATH = "test_set.json"
RESULTS_PATH = "evaluation_results.json"
PROGRESS_PATH = "eval_progress.json"


def load_eval_progress():
    import os
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r") as f:
            return set(json.load(f))
    return set()


def save_eval_progress(processed):
    with open(PROGRESS_PATH, "w") as f:
        json.dump(list(processed), f)


def load_existing_results():
    import os
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            return json.load(f)
    return []


def run_evaluation():
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        test_prs = json.load(f)

    already_done = load_eval_progress()
    results = load_existing_results()

    print(f"Resuming: {len(already_done)} PRs already evaluated.\n")

    for pr in test_prs:
        pr_number = pr["pr_number"]

        if pr_number in already_done:
            continue

        print(f"Evaluating PR #{pr_number}: {pr['title'][:60]}")

        # Build a compact ground-truth string from real human review comments
        human_review_text = "\n".join(
            f"- ({c.get('path', '')}): {c.get('body', '')}"
            for c in pr.get("review_comments", [])
        )

        # --- Memory-ON agent ---
        print("  Running memory-ON agent...")
        try:
            memory_on_review = review_pr(
            pr_title=pr["title"],
            pr_body=pr["body"],
            files_changed=pr["files_changed"],
            enable_reflection=False)
            
        except Exception as e:
            print(f"  ⚠️  Memory-ON failed: {e}")
            memory_on_review = f"[ERROR: {e}]"

        # --- Memory-OFF agent ---
        print("  Running memory-OFF agent...")
        try:
            memory_off_review = review_pr_baseline(
                pr_title=pr["title"],
                pr_body=pr["body"],
                files_changed=pr["files_changed"]
            )
        except Exception as e:
            print(f"  ⚠️  Memory-OFF failed: {e}")
            memory_off_review = f"[ERROR: {e}]"

        results.append({
            "pr_number": pr_number,
            "title": pr["title"],
            "human_review": human_review_text,
            "memory_on_review": memory_on_review,
            "memory_off_review": memory_off_review
        })

        already_done.add(pr_number)
        save_eval_progress(already_done)

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"  ✅ Saved. ({len(already_done)}/{len(test_prs)} done)\n")
        time.sleep(1)  # gentle pacing across both agents' API calls

    print(f"\n🏁 Evaluation run complete. {len(results)} PRs processed.")


if __name__ == "__main__":
    run_evaluation()