"""Tab 3: Scratchpad -- visualize how agent working memory updates step by step."""

import streamlit as st
import operator
import json
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from utils import get_llm, require_api_key


def _init_state():
    defaults = {
        "sp_run_output": None,
        "sp_step_snapshots": [],
        "sp_custom_task": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


class ScratchpadState(TypedDict):
    task: str
    scratchpad: Annotated[list, operator.add]
    final_answer: str


def render():
    _init_state()
    st.header("Scratchpad (Working Memory)")

    st.markdown("""
A **scratchpad** is a list in the graph state that nodes **append** to using
`Annotated[list, operator.add]`. This means every node's return value is
concatenated to the existing list -- nothing is overwritten.

This is the agent's **working memory**: each step can see everything
previous steps wrote, and adds its own contribution.

```
State["scratchpad"] = []

Node 1 (think) returns: ["Thought: approach X"]
--> scratchpad = ["Thought: approach X"]

Node 2 (research) returns: ["Research: found Y"]
--> scratchpad = ["Thought: approach X", "Research: found Y"]

Node 3 (draft) returns: ["Draft: combining X and Y..."]
--> scratchpad = ["Thought: ...", "Research: ...", "Draft: ..."]
```

Below we run a 4-node graph and show the scratchpad contents **after
each node**, so we can see exactly how working memory accumulates.
    """)

    if not require_api_key():
        return

    task = st.text_input(
        "Task for the scratchpad agent:",
        value="Design a rate limiting strategy for a public REST API",
        key="sp_task_input",
    )

    if st.button("Run Scratchpad Graph", key="sp_run_btn"):
        with st.spinner("Running 4-node graph..."):
            try:
                output, snapshots = _run_scratchpad_graph(task)
                st.session_state["sp_run_output"] = output
                st.session_state["sp_step_snapshots"] = snapshots
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state["sp_run_output"]:
        out = st.session_state["sp_run_output"]
        snapshots = st.session_state["sp_step_snapshots"]

        # Show step-by-step scratchpad evolution
        st.subheader("Scratchpad Evolution (step by step)")

        for i, snap in enumerate(snapshots):
            node_name = snap["node"]
            items_before = snap["before"]
            items_after = snap["after"]
            new_item = snap["added"]

            with st.expander(f"Node {i+1}: {node_name} -- scratchpad has {len(items_after)} item(s)", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Scratchpad BEFORE this node:**")
                    if items_before:
                        for j, item in enumerate(items_before):
                            st.text(f"  [{j}] {item[:120]}")
                    else:
                        st.text("  (empty)")

                with col2:
                    st.markdown("**Scratchpad AFTER this node:**")
                    for j, item in enumerate(items_after):
                        marker = " [NEW]" if j == len(items_after) - 1 else ""
                        st.text(f"  [{j}] {item[:120]}{marker}")

                st.markdown(f"**This node added:** `{new_item[:200]}`")

        # Final answer
        st.markdown("---")
        st.subheader("Final Answer")
        st.info(out.get("final_answer", ""))

        # Full scratchpad dump
        with st.expander("Full Scratchpad Contents (raw)"):
            for i, item in enumerate(out.get("scratchpad", [])):
                st.text(f"[{i}] {item}")


def _run_scratchpad_graph(task: str):
    llm = get_llm(temperature=0.7)
    snapshots = []

    def _make_node(node_name, instruction):
        def node_fn(state: ScratchpadState):
            before = list(state["scratchpad"])  # copy before

            history = "\n".join(state["scratchpad"]) if state["scratchpad"] else "None yet"
            resp = llm.invoke([
                SystemMessage(content=instruction),
                HumanMessage(content=f"Task: {state['task']}\n\nPrevious working memory:\n{history}\n\nProvide your contribution in 2-3 sentences."),
            ])
            new_entry = f"{node_name}: {resp.content.strip()}"
            after = before + [new_entry]

            snapshots.append({
                "node": node_name,
                "before": before,
                "after": after,
                "added": new_entry,
            })
            return {"scratchpad": [new_entry]}
        return node_fn

    def finalize(state: ScratchpadState):
        history = "\n".join(state["scratchpad"])
        resp = llm.invoke([
            SystemMessage(content="We are a senior engineer. Read the working memory and produce a concise final recommendation."),
            HumanMessage(content=f"Task: {state['task']}\n\nWorking memory:\n{history}\n\nFinal recommendation:"),
        ])
        return {"final_answer": resp.content.strip()}

    g = StateGraph(ScratchpadState)
    g.add_node("think", _make_node("Think", "We are a thinker. Analyze the problem and identify the key challenge."))
    g.add_node("research", _make_node("Research", "We are a researcher. Given the thinking so far, provide relevant technical context."))
    g.add_node("plan", _make_node("Plan", "We are a planner. Based on thinking and research, outline the approach."))
    g.add_node("finalize", finalize)

    g.add_edge(START, "think")
    g.add_edge("think", "research")
    g.add_edge("research", "plan")
    g.add_edge("plan", "finalize")
    g.add_edge("finalize", END)

    app = g.compile()
    result = app.invoke({"task": task, "scratchpad": [], "final_answer": ""})

    return result, snapshots
