"""Tab 6: Context Engineering -- tool loadout, quarantine, pruning, summarization, offloading."""

import streamlit as st
import json
import time
import numpy as np
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from utils import get_llm, call_llm, require_api_key


def _init_state():
    defaults = {
        "ce_loadout_output": None,
        "ce_quarantine_output": None,
        "ce_pruning_output": None,
        "ce_summarization_output": None,
        "ce_offloading_output": None,
        "ce_offloading_thoughts": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render():
    _init_state()
    st.header("Context Engineering")

    st.markdown("""
Context engineering is about **controlling what the LLM sees** so it
produces better outputs. Five core techniques:

| Technique | Core Idea |
|-----------|-----------|
| **Tool Loadout** | Show only relevant tools (Less-is-More) |
| **Context Quarantine** | Isolate sub-tasks in separate contexts |
| **Context Pruning** | Remove irrelevant sentences before generation |
| **Context Summarization** | Condense growing context into summaries |
| **Context Offloading** | Give the agent a think/scratchpad tool |
    """)

    if not require_api_key():
        return

    subtab = st.radio(
        "Technique",
        ["Tool Loadout", "Context Quarantine", "Context Pruning", "Context Summarization", "Context Offloading"],
        horizontal=True,
        key="ce_subtab",
    )

    st.markdown("---")

    if subtab == "Tool Loadout":
        _render_tool_loadout()
    elif subtab == "Context Quarantine":
        _render_quarantine()
    elif subtab == "Context Pruning":
        _render_pruning()
    elif subtab == "Context Summarization":
        _render_summarization()
    elif subtab == "Context Offloading":
        _render_offloading()


# ── 1. Tool Loadout ──

TOOL_CATALOG = [
    {"name": "get_weather", "desc": "Get the current weather for a specific location"},
    {"name": "search_web", "desc": "Search the web for information about a topic"},
    {"name": "calculate_math", "desc": "Evaluate a mathematical expression"},
    {"name": "translate_text", "desc": "Translate text to a target language"},
    {"name": "send_email", "desc": "Send an email to a recipient with subject and body"},
    {"name": "get_stock_price", "desc": "Get the current stock price for a ticker symbol"},
    {"name": "create_calendar_event", "desc": "Create a calendar event with title, date, and time"},
    {"name": "get_news", "desc": "Get latest news articles for a specific category"},
    {"name": "convert_currency", "desc": "Convert an amount from one currency to another"},
    {"name": "get_directions", "desc": "Get driving directions from origin to destination"},
]


def _render_tool_loadout():
    st.subheader("Tool Loadout (Less-is-More)")
    st.markdown("""
Instead of showing all tools to the LLM, we:
1. Ask the LLM to **describe** what tools it needs (without seeing any)
2. Match those descriptions to real tools using the LLM
3. Show only the matched tools

This reduces confusion, hallucinated parameters, and improves accuracy.
    """)

    st.markdown("**Available tools (10 total):**")
    for t in TOOL_CATALOG:
        st.text(f"  - {t['name']}: {t['desc']}")

    query = st.text_input(
        "User query:",
        value="What is the weather in Tokyo and how much is 100 USD in JPY?",
        key="ce_loadout_input",
    )

    top_k = st.slider("Tools to select", 2, 5, 3, key="ce_loadout_k")

    if st.button("Run Tool Selection", key="ce_loadout_btn"):
        with st.spinner("Selecting tools..."):
            try:
                output = _run_tool_loadout(query, top_k)
                st.session_state["ce_loadout_output"] = output
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state["ce_loadout_output"]:
        out = st.session_state["ce_loadout_output"]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**LLM described needing:**")
            for r in out.get("recommendations", []):
                st.text(f"  - {r[:100]}")
        with col2:
            st.markdown(f"**Selected {len(out.get('selected', []))} tools:**")
            for t in out.get("selected", []):
                st.text(f"  - {t}")

        reduction = out.get("reduction", 0)
        st.metric("Context Reduction", f"{reduction}%", delta=f"-{reduction}% tokens")


def _run_tool_loadout(query: str, top_k: int):
    # Step 1: LLM recommends tools without seeing catalog
    rec_raw = call_llm(
        f"Given this query, describe 1-3 tool functionalities needed. "
        f"Return ONLY a JSON list of short descriptions.\n\nQuery: {query}",
        temperature=0,
    )
    try:
        clean = rec_raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        recommendations = json.loads(clean)
    except Exception:
        recommendations = [rec_raw]

    # Step 2: Match using LLM (simpler than embeddings for the playground)
    catalog_str = "\n".join(f"{i}. {t['name']}: {t['desc']}" for i, t in enumerate(TOOL_CATALOG))
    match_raw = call_llm(
        f"Match these needed capabilities to the tool catalog. "
        f"Return the top {top_k} tool indices as a JSON array of integers.\n\n"
        f"Needed: {json.dumps(recommendations)}\n\nCatalog:\n{catalog_str}\n\nIndices:",
        temperature=0,
    )
    try:
        clean = match_raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        indices = json.loads(clean)
        selected = [TOOL_CATALOG[i]["name"] for i in indices if i < len(TOOL_CATALOG)]
    except Exception:
        selected = [TOOL_CATALOG[0]["name"]]

    return {
        "recommendations": recommendations,
        "selected": selected[:top_k],
        "reduction": int((1 - top_k / len(TOOL_CATALOG)) * 100),
    }


# ── 2. Context Quarantine ──

def _render_quarantine():
    st.subheader("Context Quarantine")
    st.markdown("""
We compare two approaches:
- **Single agent** -- handles all sub-tasks in one shared context
- **Quarantined agents** -- each sub-task gets its own agent with isolated context

The quarantined version produces more focused outputs because each
agent cannot be distracted or poisoned by the others.
    """)

    query = st.text_input(
        "Research topic:",
        value="Cloud-native application development",
        key="ce_quar_input",
    )

    aspects = st.text_input(
        "Aspects (comma-separated):",
        value="market trends, key technologies, best practices",
        key="ce_quar_aspects",
    )

    if st.button("Run Comparison", key="ce_quar_btn"):
        with st.spinner("Running single vs quarantined agents..."):
            try:
                output = _run_quarantine(query, [a.strip() for a in aspects.split(",")])
                st.session_state["ce_quarantine_output"] = output
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state["ce_quarantine_output"]:
        out = st.session_state["ce_quarantine_output"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Single Agent (shared context)**")
            st.text_area("Output", out.get("single", ""), height=300, key="ce_q_single", disabled=True)
            st.caption(f"Context size: {out.get('single_ctx', 0):,} chars")

        with col2:
            st.markdown("**Quarantined Agents (isolated)**")
            st.text_area("Output", out.get("quarantined", ""), height=300, key="ce_q_quar", disabled=True)
            st.caption(f"Max per-agent context: {out.get('quarantined_ctx', 0):,} chars")


def _run_quarantine(topic: str, aspects: list):
    llm = get_llm(temperature=0.7)

    # Single agent
    single_ctx = []
    for aspect in aspects:
        msgs = [SystemMessage(content="We are a research assistant.")] + single_ctx + [
            HumanMessage(content=f"Research {aspect} for: {topic}. Be concise (3-4 sentences).")
        ]
        resp = llm.invoke(msgs)
        single_ctx.append(HumanMessage(content=f"Research {aspect}"))
        single_ctx.append(AIMessage(content=resp.content))
    single_text = "\n\n".join(f"[{aspects[i//2]}]\n{single_ctx[i+1].content}" for i in range(0, len(single_ctx), 2))
    single_size = sum(len(m.content) for m in single_ctx)

    # Quarantined agents
    quarantined_results = []
    max_ctx = 0
    for aspect in aspects:
        resp = llm.invoke([
            SystemMessage(content=f"We are a specialist in {aspect}. Focus ONLY on this. Be concise (3-4 sentences)."),
            HumanMessage(content=f"Research {aspect} for: {topic}"),
        ])
        quarantined_results.append(f"[{aspect}]\n{resp.content}")
        ctx_size = len(resp.content)
        max_ctx = max(max_ctx, ctx_size)

    # Synthesize quarantined
    synth = llm.invoke([
        SystemMessage(content="Combine these findings into a brief integrated summary."),
        HumanMessage(content="\n\n".join(quarantined_results)),
    ])
    quarantined_text = "\n\n".join(quarantined_results) + f"\n\n[Synthesis]\n{synth.content}"

    return {
        "single": single_text,
        "single_ctx": single_size,
        "quarantined": quarantined_text,
        "quarantined_ctx": max_ctx,
    }


# ── 3. Context Pruning ──

SAMPLE_DOC = """Amazon Web Services (AWS) is a subsidiary of Amazon providing cloud computing platforms. AWS was launched in 2006.

The company headquarters are in Seattle, Washington. AWS employs over 100,000 people worldwide.

Amazon Bedrock is a fully managed service offering foundation models from AI21 Labs, Anthropic, Cohere, Meta, Mistral AI, Stability AI, and Amazon through a single API.

The AWS re:Invent conference is held annually in Las Vegas. It typically attracts over 50,000 attendees.

With Amazon Bedrock, we can experiment with and evaluate top FMs, customize them with fine-tuning and RAG, and build agents that execute tasks using enterprise data.

Bedrock supports Claude (Anthropic), Titan (Amazon), Llama (Meta), Command (Cohere), and Jurassic (AI21 Labs). The service is serverless.

Amazon founder Jeff Bezos started the company in 1994 as an online bookstore.

Key Bedrock features include model customization, knowledge bases for RAG, agents for automation, guardrails, and model evaluation.

The Amazon rainforest covers most of the Amazon basin of South America. This is unrelated to Amazon the company."""


def _render_pruning():
    st.subheader("Context Pruning")
    st.markdown("""
We remove irrelevant sentences **before** passing context to the LLM.
The pruner identifies which sentences are relevant to the question
and discards the rest, saving tokens and reducing distraction.
    """)

    st.text_area("Document (editable):", value=SAMPLE_DOC, height=200, key="ce_prune_doc")
    question = st.text_input(
        "Question:",
        value="What is Amazon Bedrock and what models does it support?",
        key="ce_prune_q",
    )

    if st.button("Prune and Compare", key="ce_prune_btn"):
        with st.spinner("Pruning context..."):
            try:
                doc = st.session_state.get("ce_prune_doc", SAMPLE_DOC)
                output = _run_pruning(question, doc)
                st.session_state["ce_pruning_output"] = output
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state["ce_pruning_output"]:
        out = st.session_state["ce_pruning_output"]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original", f"{out['original_len']} chars")
        with col2:
            st.metric("After Pruning", f"{out['pruned_len']} chars",
                      delta=f"-{out['reduction']}")

        st.markdown("**Kept sentences:**")
        st.text_area("Pruned context", out.get("pruned_text", ""), height=150, key="ce_prune_result", disabled=True)

        st.markdown("**Answer comparison:**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("*With full context:*")
            st.write(out.get("answer_full", ""))
        with col2:
            st.markdown("*With pruned context:*")
            st.write(out.get("answer_pruned", ""))


def _run_pruning(question: str, document: str):
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    sentences = nltk.sent_tokenize(document)
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))

    raw = call_llm(
        f"Return ONLY the sentence indices relevant to the question as a JSON array.\n\n"
        f"Question: {question}\n\nSentences:\n{numbered}\n\nRelevant indices:",
        temperature=0,
    )
    try:
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        indices = json.loads(clean)
        pruned = " ".join(sentences[i] for i in indices if i < len(sentences))
    except Exception:
        pruned = document
        indices = list(range(len(sentences)))

    answer_full = call_llm(f"Context:\n{document}\n\nQuestion: {question}\nAnswer:", temperature=0)
    answer_pruned = call_llm(f"Context:\n{pruned}\n\nQuestion: {question}\nAnswer:", temperature=0)

    return {
        "original_len": len(document),
        "pruned_len": len(pruned),
        "reduction": f"{(1 - len(pruned)/max(len(document),1))*100:.0f}%",
        "pruned_text": pruned,
        "kept": len(indices),
        "total": len(sentences),
        "answer_full": answer_full,
        "answer_pruned": answer_pruned,
    }


# ── 4. Context Summarization ──

def _render_summarization():
    st.subheader("Context Summarization")
    st.markdown("""
As conversation context grows, we summarize older messages to keep
the context window manageable. This preserves key facts and decisions
while discarding noise.

Below we simulate a multi-turn conversation and show when and how
summarization triggers.
    """)

    max_msgs = st.slider("Summarize after N messages", 4, 12, 6, key="ce_sum_max")
    questions = st.text_area(
        "Conversation messages (one per line):",
        value="What are the best practices for REST API design?\n"
              "How should we handle authentication?\n"
              "What about rate limiting?\n"
              "Should we use GraphQL instead?\n"
              "What database would you recommend?\n"
              "How do we handle database migrations?",
        height=150,
        key="ce_sum_msgs",
    )

    if st.button("Run Summarization Demo", key="ce_sum_btn"):
        msgs = [q.strip() for q in questions.strip().split("\n") if q.strip()]
        with st.spinner("Running conversation with auto-summarization..."):
            try:
                output = _run_summarization(msgs, max_msgs)
                st.session_state["ce_summarization_output"] = output
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state["ce_summarization_output"]:
        out = st.session_state["ce_summarization_output"]
        for step in out.get("steps", []):
            with st.expander(f"Turn {step['turn']}: \"{step['query'][:60]}\"", expanded=False):
                st.markdown(f"**Messages in buffer:** {step['buffer_size']}")
                st.markdown(f"**Summarized?** {'Yes -- compressed' if step['was_summarized'] else 'No'}")
                if step.get("summary"):
                    st.markdown("**Running summary:**")
                    st.text(step["summary"][:300])
                st.markdown("**Response:**")
                st.write(step["response"][:300])


def _run_summarization(messages: list, max_msgs: int):
    llm = get_llm(temperature=0.7)
    buffer = []
    summary = None
    steps = []

    for i, msg in enumerate(messages):
        was_summarized = False

        # Check if we need to summarize
        if len(buffer) >= max_msgs:
            history = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content[:150]}"
                for m in buffer
            )
            summary = call_llm(
                f"Summarize this conversation, preserving key decisions and facts:\n{history}",
                temperature=0.3,
            )
            buffer = buffer[-2:]
            was_summarized = True

        # Build context
        sys = "We are a helpful assistant. Be concise."
        if summary:
            sys += f"\n\nConversation summary:\n{summary}"
        buffer.append(HumanMessage(content=msg))
        resp = llm.invoke([SystemMessage(content=sys)] + buffer)
        buffer.append(AIMessage(content=resp.content))

        steps.append({
            "turn": i + 1,
            "query": msg,
            "response": resp.content,
            "buffer_size": len(buffer),
            "was_summarized": was_summarized,
            "summary": summary,
        })

    return {"steps": steps}


# ── 5. Context Offloading ──

@tool
def think(thought: str) -> str:
    """Use this tool to think through a problem step by step. It does not
    obtain new information -- it records our reasoning for later reference."""
    return "Thought recorded."

@tool
def get_order(order_id: str) -> str:
    """Look up an order by its ID."""
    orders = {
        "ORD-123": json.dumps({"status": "delivered", "total": 150.00, "items": ["Widget A", "Widget B"],
                                "delivery_date": "2025-11-25", "customer_tier": "gold"}),
        "ORD-456": json.dumps({"status": "shipped", "total": 89.99, "items": ["Gadget X"],
                                "delivery_date": None, "customer_tier": "standard"}),
    }
    return orders.get(order_id, '{"error": "Order not found"}')

@tool
def process_refund(order_id: str, amount: float, reason: str) -> str:
    """Process a refund for a given order."""
    return json.dumps({"status": "refund_processed", "confirmation": f"REF-{order_id}"})

REFUND_POLICY = """REFUND POLICY:
- Full refund within 30 days of delivery
- 50% refund between 30-60 days
- No refund after 60 days
- Gold/Platinum customers get extended 90-day full refund window
- Damaged items always get full refund regardless of time"""


def _render_offloading():
    st.subheader("Context Offloading (Think Tool)")
    st.markdown(f"""
The "think" tool lets the agent reason in a scratchpad **outside** the
main conversation. This is especially powerful in policy-heavy
environments where the agent must verify rules before acting.

Anthropic reported a **54% improvement** on airline customer service
tasks using this pattern.

**Policy for this demo:**
```
{REFUND_POLICY}
```
    """)

    query = st.text_input(
        "Customer request:",
        value="I want a refund for order ORD-123. The product was not what I expected.",
        key="ce_offload_input",
    )

    col1, col2 = st.columns(2)
    with col1:
        run_with = st.button("Run WITH think tool", key="ce_off_with")
    with col2:
        run_without = st.button("Run WITHOUT think tool", key="ce_off_without")

    if run_with:
        with st.spinner("Agent reasoning with think tool..."):
            try:
                output, thoughts = _run_with_think(query)
                st.session_state["ce_offloading_output"] = {"with": output, "thoughts": thoughts}
                st.session_state["ce_offloading_thoughts"] = thoughts
            except Exception as e:
                st.error(f"Error: {e}")

    if run_without:
        with st.spinner("Agent responding directly..."):
            try:
                output = _run_without_think(query)
                prev = st.session_state.get("ce_offloading_output") or {}
                prev["without"] = output
                st.session_state["ce_offloading_output"] = prev
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state["ce_offloading_output"]:
        out = st.session_state["ce_offloading_output"]

        if "thoughts" in out and out["thoughts"]:
            with st.expander(f"Think Tool Log ({len(out['thoughts'])} thoughts)", expanded=True):
                for i, t in enumerate(out["thoughts"]):
                    st.markdown(f"**Thought {i+1}:**")
                    st.text(t[:300])

        if "with" in out:
            st.markdown("**WITH think tool:**")
            st.info(out["with"][:500])

        if "without" in out:
            st.markdown("**WITHOUT think tool:**")
            st.info(out["without"][:500])

        if "with" in out and "without" in out:
            st.markdown("---")
            st.markdown(
                "**Observation:** The think-tool version explicitly reasons through "
                "each policy rule before deciding. This is auditable and less prone "
                "to errors. The direct version may skip steps or misapply rules."
            )


def _run_with_think(query: str):
    llm = get_llm(temperature=0.3)
    tools = [think, get_order, process_refund]
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)
    thoughts = []

    messages = [
        SystemMessage(content=(
            f"We are a customer service agent. Before processing ANY refund, "
            f"we MUST use the think tool to verify policy compliance.\n\n{REFUND_POLICY}\n\n"
            f"Steps: 1) think about which rules apply, 2) look up the order, "
            f"3) think about the correct refund, 4) process if compliant."
        )),
        HumanMessage(content=query),
    ]

    for _ in range(10):
        resp = llm_with_tools.invoke(messages)
        messages.append(resp)

        if not resp.tool_calls:
            return resp.content, thoughts

        for tc in resp.tool_calls:
            name, args, tid = tc["name"], tc["args"], tc["id"]
            if name == "think":
                thoughts.append(args.get("thought", ""))
            result = tool_map[name].invoke(args)
            messages.append(ToolMessage(content=str(result), tool_call_id=tid))

    return "Agent reached max turns.", thoughts


def _run_without_think(query: str):
    return call_llm(
        f"{REFUND_POLICY}\n\n"
        f"Order ORD-123: delivered 2025-11-25, total $150, gold customer, not damaged.\n\n"
        f"Customer request: {query}\n\nProvide the refund decision.",
        temperature=0,
    )
