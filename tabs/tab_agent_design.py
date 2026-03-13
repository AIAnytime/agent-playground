"""Tab 2: Agent Design -- ReAct agents, tool calling, multi-agent patterns."""

import streamlit as st
import json
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from utils import get_llm, call_llm, require_api_key


def _init_state():
    defaults = {
        "agent_react_output": None,
        "agent_react_steps": [],
        "agent_custom_output": None,
        "agent_custom_steps": [],
        "agent_multi_output": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# -- Mock tools --

@tool
def search_knowledge_base(query: str) -> str:
    """Search our internal knowledge base for relevant information."""
    kb = {
        "pricing": "Standard plan: $29/mo, Pro: $99/mo, Enterprise: custom pricing. All plans include 14-day trial.",
        "refund": "Full refund within 30 days. Pro-rata refund within 60 days. No refund after 60 days.",
        "features": "Core features: dashboards, API access, team collaboration. Pro adds: custom integrations, priority support.",
        "security": "SOC2 Type II certified. Data encrypted at rest (AES-256) and in transit (TLS 1.3). GDPR compliant.",
    }
    for key, val in kb.items():
        if key in query.lower():
            return val
    return f"No specific results for '{query}'. Available topics: pricing, refund, features, security."

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

@tool
def get_current_date() -> str:
    """Get the current date."""
    from datetime import date
    return f"Current date: {date.today().isoformat()}"

TOOLS = [search_knowledge_base, calculate, get_current_date]


def render():
    _init_state()
    st.header("Agent Design")

    st.markdown("""
An **agent** is a graph where the LLM decides which tools to call and when
to stop. The most common pattern is **ReAct** (Reason + Act): the LLM
reasons about what to do, calls a tool, observes the result, and repeats
until it has enough information to answer.

**Key components:**
- **Tools** -- functions the agent can call (search, calculate, APIs)
- **Tool binding** -- the LLM is told about available tools via `.bind_tools()`
- **Agent loop** -- call LLM, execute tool calls, feed results back, repeat
    """)

    if not require_api_key():
        return

    # ── Section 1: ReAct Agent (prebuilt) ──
    st.subheader("1. ReAct Agent (LangGraph Prebuilt)")
    st.markdown("""
`create_react_agent` is LangGraph's built-in ReAct implementation. It
handles the tool-call loop automatically. We provide the LLM and tools.
    """)

    react_query = st.text_input(
        "Ask the agent something:",
        value="What is our Pro plan pricing and what would 12 months cost?",
        key="agent_react_input",
    )

    if st.button("Run ReAct Agent", key="agent_react_btn"):
        with st.spinner("Agent is working..."):
            try:
                output, steps = _run_react_agent(react_query)
                st.session_state["agent_react_output"] = output
                st.session_state["agent_react_steps"] = steps
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state["agent_react_output"]:
        st.markdown("**Final Answer:**")
        st.info(st.session_state["agent_react_output"])

        if st.session_state["agent_react_steps"]:
            with st.expander("Agent Steps (tool calls and reasoning)", expanded=True):
                for i, step in enumerate(st.session_state["agent_react_steps"]):
                    st.markdown(f"**Step {i+1}:** `{step['type']}`")
                    st.code(step["content"][:500], language="text")

    # ── Section 2: Custom Agent with Manual Loop ──
    st.markdown("---")
    st.subheader("2. Custom Agent (Manual Tool Loop)")
    st.markdown("""
For full control, we can build the agent loop ourselves. This lets us
add custom logic -- logging, retries, guardrails -- at each step.

```
while True:
    response = llm_with_tools.invoke(messages)
    if response.tool_calls:
        for tc in response.tool_calls:
            result = execute_tool(tc)
            messages.append(ToolMessage(...))
    else:
        break  # agent is done
```
    """)

    custom_query = st.text_input(
        "Ask the custom agent:",
        value="What security certifications do we have? Also what is 256 * 365?",
        key="agent_custom_input",
    )

    if st.button("Run Custom Agent", key="agent_custom_btn"):
        with st.spinner("Running custom loop..."):
            try:
                output, steps = _run_custom_agent(custom_query)
                st.session_state["agent_custom_output"] = output
                st.session_state["agent_custom_steps"] = steps
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state["agent_custom_output"]:
        st.markdown("**Final Answer:**")
        st.info(st.session_state["agent_custom_output"])

        if st.session_state["agent_custom_steps"]:
            with st.expander("Execution Trace", expanded=True):
                for i, step in enumerate(st.session_state["agent_custom_steps"]):
                    st.markdown(f"**Turn {i+1}:** `{step['action']}`")
                    st.code(step["detail"][:500], language="text")

    # ── Section 3: Multi-Agent (Supervisor) ──
    st.markdown("---")
    st.subheader("3. Multi-Agent: Supervisor Pattern")
    st.markdown("""
A **supervisor** agent delegates sub-tasks to specialist agents, each
with their own system prompt and tools. The supervisor reads their
outputs and synthesizes a final answer.

```
User --> Supervisor --> [Researcher, Analyst, Writer] --> Supervisor --> Answer
```
    """)

    multi_query = st.text_input(
        "Complex question for multi-agent:",
        value="Research our pricing, calculate annual cost for Pro plan with 20% discount, and write a recommendation.",
        key="agent_multi_input",
    )

    if st.button("Run Multi-Agent", key="agent_multi_btn"):
        with st.spinner("Supervisor delegating to specialists..."):
            try:
                output = _run_multi_agent(multi_query)
                st.session_state["agent_multi_output"] = output
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state["agent_multi_output"]:
        out = st.session_state["agent_multi_output"]
        for role, content in out.items():
            with st.expander(f"Agent: {role}", expanded=(role == "supervisor")):
                st.write(content)


# ── Implementations ──

def _run_react_agent(query: str):
    llm = get_llm(temperature=0.3)
    agent = create_react_agent(llm, TOOLS)
    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    steps = []
    final_answer = ""
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    steps.append({"type": f"tool_call: {tc['name']}", "content": json.dumps(tc["args"], indent=2)})
            elif msg.content:
                final_answer = msg.content
        elif isinstance(msg, ToolMessage):
            steps.append({"type": "tool_result", "content": msg.content})

    return final_answer, steps


def _run_custom_agent(query: str):
    llm = get_llm(temperature=0.3)
    llm_with_tools = llm.bind_tools(TOOLS)
    tool_map = {t.name: t for t in TOOLS}

    messages = [
        SystemMessage(content="We are a helpful assistant. Use tools when needed to answer accurately."),
        HumanMessage(content=query),
    ]
    steps = []

    for turn in range(8):
        resp = llm_with_tools.invoke(messages)
        messages.append(resp)

        if not resp.tool_calls:
            steps.append({"action": "final_answer", "detail": resp.content})
            return resp.content, steps

        for tc in resp.tool_calls:
            name, args, tid = tc["name"], tc["args"], tc["id"]
            steps.append({"action": f"call {name}", "detail": json.dumps(args)})
            result = tool_map[name].invoke(args)
            steps.append({"action": f"result from {name}", "detail": str(result)})
            messages.append(ToolMessage(content=str(result), tool_call_id=tid))

    return "Agent reached max turns.", steps


def _run_multi_agent(query: str):
    llm = get_llm(temperature=0.7)
    results = {}

    # Researcher
    researcher = create_react_agent(llm, [search_knowledge_base])
    res = researcher.invoke({"messages": [HumanMessage(content=f"Research: {query}")]})
    research_text = _extract_final(res)
    results["researcher"] = research_text

    # Analyst
    analyst = create_react_agent(llm, [calculate])
    res = analyst.invoke({"messages": [HumanMessage(content=f"Analyze with calculations: {query}")]})
    analysis_text = _extract_final(res)
    results["analyst"] = analysis_text

    # Supervisor synthesizes
    synthesis = llm.invoke([
        SystemMessage(content="We are a supervisor. Synthesize the inputs into a clear recommendation."),
        HumanMessage(content=f"Research:\n{research_text}\n\nAnalysis:\n{analysis_text}\n\nProvide a final recommendation."),
    ])
    results["supervisor"] = synthesis.content

    return results


def _extract_final(result):
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return msg.content
    return "No output."
