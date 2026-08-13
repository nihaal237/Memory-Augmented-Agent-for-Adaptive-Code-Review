import os
import json
import sqlite3
import numpy as np
from dotenv import load_dotenv
from typing import Annotated, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from memory_utils import (
    embedding_fn, review_memory_collection, DB_PATH, save_convention
)

load_dotenv()

class ReviewState(TypedDict):
    messages: Annotated[list, add_messages]
    enable_reflection: bool

# --- Tools, defined using LangChain's @tool decorator ---
@tool
def search_similar_reviews(query: str) -> str:
    """Search past code review comments for ones semantically similar to a piece of
    code or concern. Use this when the current code reminds you of a pattern, issue,
    or style choice that might have been discussed before."""
    results = review_memory_collection.query(query_texts=[query], n_results=3)

    if not results["documents"] or not results["documents"][0]:
        return "No similar past review comments found."

    formatted = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        formatted.append(f"- (from PR #{meta['pr_number']}, {meta['path']}): {doc}")
    return "\n".join(formatted)


@tool
def search_conventions(query: str) -> str:
    """Search known, established coding conventions for this repository relevant to
    a topic. Use this to check if there's a known team standard about something
    (e.g., error handling style, naming, documentation format)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT convention_text, category, times_confirmed 
        FROM conventions WHERE active = 1 ORDER BY times_confirmed DESC
    """)
    all_conventions = cursor.fetchall()
    conn.close()

    if not all_conventions:
        return "No conventions stored yet."

    query_emb = np.array(embedding_fn([query])[0])
    texts = [row[0] for row in all_conventions]
    conv_embeddings = embedding_fn(texts)

    scored = []
    for (text, category, count), emb in zip(all_conventions, conv_embeddings):
        vec = np.array(emb)
        score = np.dot(query_emb, vec) / (np.linalg.norm(query_emb) * np.linalg.norm(vec))
        scored.append((score, text, category, count))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:5]
    return "\n".join(f"- [{cat}, confirmed {count}x] {text}" for _, text, cat, count in top)


tools = [search_similar_reviews, search_conventions]
tool_node = ToolNode(tools)

# --- LLM setup, bound with tools ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)
llm_with_tools = llm.bind_tools(tools)


# --- Graph state definition ---
class ReviewState(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = """You are an experienced code reviewer for this repository.
You have access to two tools that let you recall past review history and known conventions:
- search_similar_reviews: find past review comments similar to a concern
- search_conventions: find established team conventions on a topic

Before finalizing your review, you should proactively check search_conventions at least
once for any code pattern involving error handling, resource access (files, network,
database), input validation, or naming — these are areas where established team
conventions are common and directly affect review quality. Use search_similar_reviews
when the code reminds you of a specific pattern or issue you suspect has come up before.

It's fine if a search returns nothing relevant — but skipping the search entirely on
code that touches these areas means you might miss established team standards.

Once you're confident, provide your final review as a clear, concise list of specific
issues or observations. If the code looks fine, say so briefly."""


def extract_text(content) -> str:
    """
    Gemini via LangChain sometimes returns content as a plain string,
    sometimes as a list of content blocks (dicts with a 'text' key).
    This normalizes both cases into plain text.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts)

    return str(content)


# --- Graph nodes ---
def reasoning_node(state: ReviewState):
    """The agent's main reasoning step: decide to call a tool, or give a final answer."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)

    # Print tool call requests live, at the moment they're decided
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        for tc in tool_calls:
            print(f"  🔧 Agent called tool: {tc['name']}({tc['args']})")

    return {"messages": [response]}


def should_continue(state: ReviewState) -> str:
    """Routing logic: if the last message requested tool calls, go to tools;
    otherwise route to reflection (if enabled) or end directly."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"

    if state.get("enable_reflection", True):
        return "reflection"
    return END


def reflection_node(state: ReviewState):
    """
    After the review is finalized, decide if anything from this review
    reflects a reusable convention worth saving to memory.
    """
    messages = state["messages"]
    final_review = extract_text(messages[-1].content)

    # Find the original PR description from the first human message
    original_request = next(
        (m.content for m in messages if isinstance(m, HumanMessage)), ""
    )

    reflection_prompt = f"""You just completed this code review:

{final_review}

Original context reviewed:
{original_request[:1000]}

Does this review contain any REUSABLE CODING CONVENTION worth remembering for future
reviews (something that would apply to different code, not just this specific PR)?
If the review already confirmed an existing known convention (e.g., cited "team convention"),
respond with is_convention: false, since it's already stored — only flag genuinely NEW
observations not already established.

Respond ONLY with valid JSON, no markdown, no backticks:
{{
  "is_convention": true or false,
  "convention_text": "short, general, reusable rule IF true, else empty string",
  "category": "one of: code-style, testing, performance, compatibility, documentation, error-handling, other"
}}"""

    response = llm.invoke([HumanMessage(content=reflection_prompt)])
    raw_text = extract_text(response.content).strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        print("  ⚠️  Reflection: failed to parse response, skipping.")
        return state

    if result.get("is_convention") and result.get("convention_text"):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        save_convention(cursor, result["convention_text"], pr_number=None, category=result.get("category", "other"))
        conn.commit()
        conn.close()
        print(f"  🧠 Reflection: saved new convention -> {result['convention_text'][:80]}")
    else:
        print("  🧠 Reflection: nothing new worth saving.")

    return state


# --- Build the graph ---
graph_builder = StateGraph(ReviewState)
graph_builder.add_node("reasoning", reasoning_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("reflection", reflection_node)

graph_builder.set_entry_point("reasoning")
graph_builder.add_conditional_edges(
    "reasoning", 
    should_continue, 
    {"tools": "tools", "reflection": "reflection", END: END}
)
graph_builder.add_edge("tools", "reasoning")
graph_builder.add_edge("reflection", END)

review_agent = graph_builder.compile()


def review_pr(pr_title: str, pr_body: str, files_changed: list, enable_reflection: bool = True) -> str:
    """Runs the LangGraph agent on a PR and returns the final review text.
    
    Set enable_reflection=False during evaluation runs to prevent the agent's
    memory from being modified by test-set reviews, keeping evaluation clean.
    """
    diffs_text = "\n\n".join(
        f"File: {f['filename']}\n{f['patch'][:1000]}"
        for f in files_changed
    )

    initial_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""Review this pull request:

Title: {pr_title}
Description: {pr_body[:500]}

Code changes:
{diffs_text[:3000]}""")
    ]

    final_state = review_agent.invoke({
        "messages": initial_messages,
        "enable_reflection": enable_reflection
    })
    return extract_text(final_state["messages"][-1].content)

if __name__ == "__main__":
    test_files = [{
        "filename": "utils.py",
        "patch": """+def load_config(path):
+    if os.path.exists(path):
+        with open(path) as f:
+            return json.load(f)
+    return {}"""
    }]

    review = review_pr(
        pr_title="Add config loader utility",
        pr_body="Adds a helper to load JSON config files.",
        files_changed=test_files
    )

    print("\n=== FINAL REVIEW ===")
    print(review)