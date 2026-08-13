from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

llm_baseline = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

BASELINE_SYSTEM_PROMPT = """You are an experienced code reviewer. Review the following
pull request and provide a clear, concise list of specific issues or observations.
If the code looks fine, say so briefly."""


def review_pr_baseline(pr_title: str, pr_body: str, files_changed: list) -> str:
    """A plain LLM reviewer with NO memory access — the control group."""
    diffs_text = "\n\n".join(
        f"File: {f['filename']}\n{f['patch'][:1000]}"
        for f in files_changed
    )

    messages = [
        SystemMessage(content=BASELINE_SYSTEM_PROMPT),
        HumanMessage(content=f"""Review this pull request:

Title: {pr_title}
Description: {pr_body[:500]}

Code changes:
{diffs_text[:3000]}""")
    ]

    response = llm_baseline.invoke(messages)
    content = response.content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content