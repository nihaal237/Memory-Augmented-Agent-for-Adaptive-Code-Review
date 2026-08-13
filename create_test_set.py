import json
import random

PR_DATA_PATH = "pr_data_scikit-learn.json"
random.seed(42)  # reproducible sample

with open(PR_DATA_PATH, "r", encoding="utf-8") as f:
    pr_data = json.load(f)

candidate_pool = pr_data[30:]

# Filter: 3-15 review comments (substantial but not unwieldy)
filtered = [
    pr for pr in candidate_pool
    if 3 <= len(pr.get("review_comments", [])) <= 15
]

print(f"PRs in usable range (3-15 comments): {len(filtered)}")

SAMPLE_SIZE = 20
test_sample = random.sample(filtered, min(SAMPLE_SIZE, len(filtered)))

with open("test_set.json", "w", encoding="utf-8") as f:
    json.dump(test_sample, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(test_sample)} PRs to test_set.json")
print("\nSelected PRs:")
for pr in test_sample:
    print(f"  PR #{pr['pr_number']} ({len(pr['review_comments'])} comments): {pr['title'][:60]}")