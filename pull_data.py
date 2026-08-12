import os
import time
import json
from dotenv import load_dotenv
from github import Github

load_dotenv()

github_client = Github(os.getenv("GITHUB_TOKEN"))

REPO_NAME = "scikit-learn/scikit-learn"
NUM_PRS_TO_FETCH = 150          # target: PRs that HAVE review comments
CHECKPOINT_EVERY = 10           # save progress every N successful PRs
OUTPUT_FILE = "pr_data_scikit-learn.json"

repo = github_client.get_repo(REPO_NAME)


def check_rate_limit(client, min_remaining=50):
    """Sleep if we're close to hitting the GitHub API rate limit."""
    rate = client.get_rate_limit().core
    if rate.remaining < min_remaining:
        reset_time = rate.reset.timestamp()
        sleep_for = max(reset_time - time.time() + 5, 0)
        print(f"Rate limit low ({rate.remaining} left). Sleeping {sleep_for:.0f}s...")
        time.sleep(sleep_for)


def fetch_closed_prs_with_reviews(repo, limit=150):
    """
    Pulls merged PRs that have at least one review comment.
    Skips PRs with zero review comments before doing the expensive
    get_files() call, to save API requests.
    """
    pulls = repo.get_pulls(state="closed", sort="created", direction="desc")

    collected = []
    scanned = 0

    for pr in pulls:
        if len(collected) >= limit:
            break

        scanned += 1

        if scanned % 25 == 0:
            check_rate_limit(github_client)

        if not pr.merged:
            continue

        # Cheap check first: does this PR even have review comments?
        review_comments_paginated = pr.get_review_comments()
        if review_comments_paginated.totalCount == 0:
            continue

        review_comments = []
        for comment in review_comments_paginated:
            review_comments.append({
                "body": comment.body,
                "path": comment.path,
                "author": comment.user.login if comment.user else "unknown",
                "line": getattr(comment, "line", None),
                "diff_hunk": getattr(comment, "diff_hunk", ""),
            })

        # Only now pull the (more expensive) file diffs
        files_changed = []
        for f in pr.get_files():
            files_changed.append({
                "filename": f.filename,
                "patch": f.patch if f.patch else "",
                "additions": f.additions,
                "deletions": f.deletions
            })

        collected.append({
            "pr_number": pr.number,
            "title": pr.title,
            "body": pr.body if pr.body else "",
            "files_changed": files_changed,
            "review_comments": review_comments
        })

        print(f"[{len(collected)}/{limit}] Fetched PR #{pr.number} "
              f"({len(review_comments)} review comments): {pr.title[:60]}")

        if len(collected) % CHECKPOINT_EVERY == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(collected, f, indent=2, ensure_ascii=False)
            print(f"  -> checkpoint saved ({len(collected)} PRs)")

    print(f"\nScanned {scanned} PRs total to find {len(collected)} with review comments.")
    return collected


if __name__ == "__main__":
    check_rate_limit(github_client)

    data = fetch_closed_prs_with_reviews(repo, limit=NUM_PRS_TO_FETCH)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(data)} PRs to {OUTPUT_FILE}")