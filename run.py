import json

PR_DATA_PATH = "pr_data_scikit-learn.json"

with open(PR_DATA_PATH, "r", encoding="utf-8") as f:
    pr_data = json.load(f)

# Exclude the first 30 (already used for seeding memory)
candidate_pool = pr_data[30:]

print(f"Total candidate PRs (excluding seeded 30): {len(candidate_pool)}\n")

thresholds = [1, 2, 3, 5, 10]
for t in thresholds:
    count = sum(1 for pr in candidate_pool if len(pr.get("review_comments", [])) >= t)
    print(f"PRs with >= {t} review comments: {count}")

# Show the actual distribution of comment counts, sorted
counts = sorted(
    [(pr["pr_number"], len(pr.get("review_comments", []))) for pr in candidate_pool],
    key=lambda x: x[1],
    reverse=True
)

print("\nTop 15 PRs by comment count (in this pool):")
for pr_number, count in counts[:15]:
    print(f"  PR #{pr_number}: {count} comments")

print("\nBottom 10 PRs by comment count (in this pool):")
for pr_number, count in counts[-10:]:
    print(f"  PR #{pr_number}: {count} comments")