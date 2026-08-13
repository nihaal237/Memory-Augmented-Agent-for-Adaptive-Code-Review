import json

RESULTS_PATH = "evaluation_results.json"
SCORES_PATH = "evaluation_scores.json"

# Keywords that suggest a PR touches themes your memory was actually seeded on
CONVENTION_KEYWORDS = [
    "error", "exception", "try", "except", "file", "path", "resource",
    "docstring", "documentation", "naming", "deprecat", "warning",
    "array api", "type hint", "test", "validation"
]

# Themes where human feedback was purely process/meta, not code substance
META_KEYWORDS = [
    "close this pr", "merge conflict", "redundant", "duplicate pr",
    "should we merge", "wait for", "another pr"
]


def tag_pr(title: str, human_review: str) -> str:
    text = (title + " " + human_review).lower()

    if any(kw in text for kw in META_KEYWORDS):
        return "meta/process (not a real review test)"
    if any(kw in text for kw in CONVENTION_KEYWORDS):
        return "convention-relevant"
    return "pure logic/wording (not memory-relevant)"


def tag_all():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = {r["pr_number"]: r for r in json.load(f)}

    with open(SCORES_PATH, "r", encoding="utf-8") as f:
        scores = json.load(f)

    tagged = []
    for s in scores:
        pr_number = s["pr_number"]
        result = results.get(pr_number, {})
        tag = tag_pr(s["title"], result.get("human_review", ""))
        tagged.append({**s, "tag": tag})

    with open("tagged_scores.json", "w", encoding="utf-8") as f:
        json.dump(tagged, f, indent=2, ensure_ascii=False)

    # Print grouped summary
    from collections import defaultdict
    groups = defaultdict(list)
    for t in tagged:
        groups[t["tag"]].append(t)

    for tag, items in groups.items():
        on_overlaps = [i["memory_on_score"]["overlap"] for i in items if i["memory_on_score"]["overlap"] is not None]
        off_overlaps = [i["memory_off_score"]["overlap"] for i in items if i["memory_off_score"]["overlap"] is not None]
        on_avg = sum(on_overlaps) / len(on_overlaps) if on_overlaps else 0
        off_avg = sum(off_overlaps) / len(off_overlaps) if off_overlaps else 0
        print(f"\n[{tag}] ({len(items)} PRs)")
        print(f"  Memory-ON overlap avg:  {on_avg:.2f}")
        print(f"  Memory-OFF overlap avg: {off_avg:.2f}")


if __name__ == "__main__":
    tag_all()