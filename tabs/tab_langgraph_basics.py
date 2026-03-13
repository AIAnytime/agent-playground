"""Tab 1: LangGraph Basics -- graph construction, nodes, edges, state."""

import streamlit as st
import json
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from utils import get_llm, call_llm, require_api_key


def _init_state():
    defaults = {
        "lg_basics_graph_output": None,
        "lg_basics_conditional_output": None,
        "lg_basics_multi_output": None,
        "lg_basics_custom_query": "",
        "lg_basics_custom_output": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render():
    _init_state()
    st.header("LangGraph Basics")

    st.markdown("""
LangGraph models computation as a **directed graph** where:
- **Nodes** are Python functions that transform state
- **Edges** define the execution order
- **State** is a shared TypedDict that flows through all nodes

This is a fundamentally different pattern from simple chain-of-calls.
The graph is compiled once, then invoked with an initial state.
    """)

    if not require_api_key():
        return

    # ── Section 1: Simple Linear Graph ──
    st.subheader("1. Simple Linear Graph")
    st.markdown("""
We build a two-node graph: **analyze** (break down a topic) then
**summarize** (condense the analysis). State flows linearly.

```
START --> analyze --> summarize --> END
```
    """)

    if st.button("Run Linear Graph", key="lg_linear_btn"):
        with st.spinner("Running graph..."):
            try:
                output = _run_linear_graph()
                st.session_state["lg_basics_graph_output"] = output
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state["lg_basics_graph_output"]:
        out = st.session_state["lg_basics_graph_output"]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Node 1: Analyze**")
            st.text_area("Analysis", out.get("analysis", ""), height=200, key="lg_analysis_ta", disabled=True)
        with col2:
            st.markdown("**Node 2: Summarize**")
            st.text_area("Summary", out.get("summary", ""), height=200, key="lg_summary_ta", disabled=True)

    # ── Section 2: Conditional Branching ──
    st.markdown("---")
    st.subheader("2. Conditional Branching")
    st.markdown("""
Edges can be **conditional** -- a router function inspects the state and
decides which node to visit next. This is how agents implement
decision-making within a graph.

```
START --> classify --[technical]--> tech_response --> END
                  --[general]----> general_response --> END
```
    """)

    cond_query = st.text_input(
        "Enter a question (technical or general):",
        value="How does gradient descent work in neural networks?",
        key="lg_cond_input",
    )

    if st.button("Run Conditional Graph", key="lg_cond_btn"):
        with st.spinner("Running..."):
            try:
                output = _run_conditional_graph(cond_query)
                st.session_state["lg_basics_conditional_output"] = output
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state["lg_basics_conditional_output"]:
        out = st.session_state["lg_basics_conditional_output"]
        st.markdown(f"**Route taken:** `{out.get('category', 'unknown')}`")
        st.text_area("Response", out.get("response", ""), height=200, key="lg_cond_resp_ta", disabled=True)

    # ── Section 3: Multi-Step Pipeline ──
    st.markdown("---")
    st.subheader("3. Multi-Step Pipeline with State Accumulation")
    st.markdown("""
Each node reads from and writes to the shared state. We can trace
exactly what each node contributed -- full auditability.

```
START --> research --> critique --> refine --> END
```
    """)

    pipeline_topic = st.text_input(
        "Topic to research:",
        value="Benefits and risks of AI code generation tools",
        key="lg_pipeline_input",
    )

    if st.button("Run Pipeline", key="lg_pipeline_btn"):
        with st.spinner("Running 3-step pipeline..."):
            try:
                output = _run_pipeline(pipeline_topic)
                st.session_state["lg_basics_multi_output"] = output
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state["lg_basics_multi_output"]:
        out = st.session_state["lg_basics_multi_output"]
        for step_name in ["research", "critique", "refined"]:
            with st.expander(f"Step: {step_name}", expanded=(step_name == "refined")):
                st.write(out.get(step_name, ""))


# ── Graph implementations ──

def _run_linear_graph():
    class LinearState(TypedDict):
        topic: str
        analysis: str
        summary: str

    llm = get_llm(temperature=0.7)

    def analyze(state: LinearState):
        resp = llm.invoke([
            SystemMessage(content="We are a research analyst. Provide a detailed analysis."),
            HumanMessage(content=f"Analyze this topic: {state['topic']}"),
        ])
        return {"analysis": resp.content}

    def summarize(state: LinearState):
        resp = llm.invoke([
            SystemMessage(content="We are a summarizer. Condense the analysis into 3-4 key points."),
            HumanMessage(content=f"Summarize:\n{state['analysis']}"),
        ])
        return {"summary": resp.content}

    g = StateGraph(LinearState)
    g.add_node("analyze", analyze)
    g.add_node("summarize", summarize)
    g.add_edge(START, "analyze")
    g.add_edge("analyze", "summarize")
    g.add_edge("summarize", END)

    app = g.compile()
    return app.invoke({"topic": "The impact of large language models on software engineering"})


def _run_conditional_graph(query: str):
    class CondState(TypedDict):
        query: str
        category: str
        response: str

    llm = get_llm(temperature=0.3)

    def classify(state: CondState):
        resp = llm.invoke([
            SystemMessage(content="Classify the query as 'technical' or 'general'. Reply with one word only."),
            HumanMessage(content=state["query"]),
        ])
        cat = "technical" if "technical" in resp.content.lower() else "general"
        return {"category": cat}

    def tech_response(state: CondState):
        resp = llm.invoke([
            SystemMessage(content="We are a technical expert. Provide a detailed technical answer."),
            HumanMessage(content=state["query"]),
        ])
        return {"response": resp.content}

    def general_response(state: CondState):
        resp = llm.invoke([
            SystemMessage(content="We are a helpful assistant. Provide a clear, accessible answer."),
            HumanMessage(content=state["query"]),
        ])
        return {"response": resp.content}

    def route(state: CondState):
        return state["category"]

    g = StateGraph(CondState)
    g.add_node("classify", classify)
    g.add_node("technical", tech_response)
    g.add_node("general", general_response)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route, {"technical": "technical", "general": "general"})
    g.add_edge("technical", END)
    g.add_edge("general", END)

    app = g.compile()
    return app.invoke({"query": query})


def _run_pipeline(topic: str):
    class PipeState(TypedDict):
        topic: str
        research: str
        critique: str
        refined: str

    llm = get_llm(temperature=0.7)

    def research(state: PipeState):
        resp = llm.invoke([
            SystemMessage(content="We are a researcher. Provide a thorough analysis of the topic."),
            HumanMessage(content=f"Research: {state['topic']}"),
        ])
        return {"research": resp.content}

    def critique(state: PipeState):
        resp = llm.invoke([
            SystemMessage(content="We are a critical reviewer. Identify weaknesses, gaps, and biases."),
            HumanMessage(content=f"Critique this research:\n{state['research']}"),
        ])
        return {"critique": resp.content}

    def refine(state: PipeState):
        resp = llm.invoke([
            SystemMessage(content="We are an editor. Incorporate the critique to produce a refined version."),
            HumanMessage(content=f"Original:\n{state['research']}\n\nCritique:\n{state['critique']}\n\nRefine:"),
        ])
        return {"refined": resp.content}

    g = StateGraph(PipeState)
    g.add_node("research", research)
    g.add_node("critique", critique)
    g.add_node("refine", refine)
    g.add_edge(START, "research")
    g.add_edge("research", "critique")
    g.add_edge("critique", "refine")
    g.add_edge("refine", END)

    app = g.compile()
    return app.invoke({"topic": topic})
