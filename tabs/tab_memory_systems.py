"""Tab 5: Memory Systems -- buffer, working, episodic, semantic, procedural."""

import streamlit as st
import json
import hashlib
from typing import TypedDict, Annotated, List, Dict
from datetime import datetime
from collections import deque
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from utils import get_llm, call_llm, require_api_key


def _init_state():
    defaults = {
        "mem_buffer_history": [],
        "mem_buffer_output": None,
        "mem_working": [],
        "mem_working_output": None,
        "mem_episodic_store": [],
        "mem_episodic_output": None,
        "mem_semantic_store": {},
        "mem_semantic_output": None,
        "mem_procedural_store": {},
        "mem_procedural_output": None,
        "mem_unified_output": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render():
    _init_state()
    st.header("Memory Systems")

    st.markdown("""
Agent memory mirrors how human memory works -- different stores for
different purposes. Each type has its own retention policy, access
pattern, and use case.

| Type | Human Analogy | Agent Implementation | Retention |
|------|--------------|---------------------|-----------|
| **Buffer/Sensory** | Sensory register | Raw message queue (last N messages) | Very short |
| **Working** | Active thought | Scratchpad, current reasoning state | Current task |
| **Episodic** | Personal experiences | Timestamped interaction logs | Long-term |
| **Semantic** | Facts and knowledge | Extracted facts, entity store | Permanent |
| **Procedural** | Skills, how-to | Learned patterns, successful strategies | Permanent |
    """)

    if not require_api_key():
        return

    subtab = st.radio(
        "Memory Type",
        ["Buffer (Sensory)", "Working Memory", "Episodic", "Semantic", "Procedural", "Unified Agent"],
        horizontal=True,
        key="mem_subtab",
    )

    st.markdown("---")

    if subtab == "Buffer (Sensory)":
        _render_buffer()
    elif subtab == "Working Memory":
        _render_working()
    elif subtab == "Episodic":
        _render_episodic()
    elif subtab == "Semantic":
        _render_semantic()
    elif subtab == "Procedural":
        _render_procedural()
    elif subtab == "Unified Agent":
        _render_unified()


def _render_buffer():
    st.subheader("Buffer / Sensory Memory")
    st.markdown("""
The simplest memory: a **fixed-size queue** of the last N messages.
Old messages fall off the end. This is what most chatbots use by default.

- **Capacity:** fixed (e.g., last 6 messages)
- **Retention:** very short -- oldest messages are discarded
- **Use case:** basic conversation continuity
    """)

    buffer_size = st.slider("Buffer size (messages)", 2, 12, 6, key="mem_buf_size")
    user_msg = st.text_input("Send a message:", key="mem_buf_input")

    if st.button("Send", key="mem_buf_send") and user_msg:
        with st.spinner("Responding..."):
            st.session_state["mem_buffer_history"].append({"role": "human", "content": user_msg})

            # Build context from buffer
            recent = st.session_state["mem_buffer_history"][-buffer_size:]
            msgs = [SystemMessage(content="We are a helpful assistant. Be concise.")]
            for m in recent:
                if m["role"] == "human":
                    msgs.append(HumanMessage(content=m["content"]))
                else:
                    msgs.append(AIMessage(content=m["content"]))

            llm = get_llm(temperature=0.7)
            resp = llm.invoke(msgs)
            st.session_state["mem_buffer_history"].append({"role": "ai", "content": resp.content})
            st.rerun()

    if st.session_state["mem_buffer_history"]:
        st.markdown("**Conversation:**")
        for m in st.session_state["mem_buffer_history"]:
            prefix = "User" if m["role"] == "human" else "Agent"
            st.text(f"  {prefix}: {m['content'][:200]}")

        total = len(st.session_state["mem_buffer_history"])
        visible = min(total, buffer_size)
        st.caption(f"Total messages: {total} | In buffer (visible to LLM): {visible}")

        if st.button("Clear Buffer", key="mem_buf_clear"):
            st.session_state["mem_buffer_history"] = []
            st.rerun()


def _render_working():
    st.subheader("Working Memory")
    st.markdown("""
Working memory holds the **current reasoning state** -- facts gathered,
hypotheses formed, and decisions made during the current task. It is
the scratchpad we saw in the previous tab, but framed as a memory type.

- **Capacity:** task-scoped
- **Retention:** cleared when the task is done
- **Use case:** multi-step reasoning, complex analysis
    """)

    task = st.text_input(
        "Task to reason about:",
        value="Should we migrate our monolithic app to microservices?",
        key="mem_work_input",
    )

    if st.button("Run Working Memory Agent", key="mem_work_btn"):
        with st.spinner("Agent reasoning through steps..."):
            try:
                output, steps = _run_working_memory(task)
                st.session_state["mem_working"] = steps
                st.session_state["mem_working_output"] = output
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state["mem_working"]:
        st.markdown("**Working memory contents at each step:**")
        for i, step in enumerate(st.session_state["mem_working"]):
            with st.expander(f"Step {i+1}: {step['label']}", expanded=(i == len(st.session_state['mem_working'])-1)):
                st.markdown("Working memory state:")
                for j, item in enumerate(step["memory"]):
                    st.text(f"  [{j}] {item[:150]}")

    if st.session_state["mem_working_output"]:
        st.markdown("**Final Output:**")
        st.info(st.session_state["mem_working_output"])


def _render_episodic():
    st.subheader("Episodic Memory")
    st.markdown("""
Episodic memory stores **past interactions as timestamped episodes**.
When the agent encounters a similar situation, it retrieves relevant
episodes to inform its response -- like recalling a past experience.

- **Capacity:** unbounded (stored externally)
- **Retention:** permanent
- **Use case:** learning from past interactions, personalization
    """)

    episode_input = st.text_input(
        "Ask something (episodes are remembered across questions):",
        key="mem_ep_input",
    )

    if st.button("Send", key="mem_ep_send") and episode_input:
        with st.spinner("Checking episodic memory..."):
            try:
                output, store = _run_episodic(episode_input)
                st.session_state["mem_episodic_output"] = output
                st.session_state["mem_episodic_store"] = store
            except Exception as e:
                st.error(f"Error: {e}")
        st.rerun()

    if st.session_state["mem_episodic_store"]:
        with st.expander(f"Episodic Store ({len(st.session_state['mem_episodic_store'])} episodes)", expanded=False):
            for ep in st.session_state["mem_episodic_store"]:
                st.text(f"  [{ep['timestamp']}] Q: {ep['query'][:80]} -> {ep['response'][:80]}")

    if st.session_state["mem_episodic_output"]:
        st.markdown("**Response (with episodic context):**")
        st.info(st.session_state["mem_episodic_output"])

    if st.button("Clear Episodes", key="mem_ep_clear"):
        st.session_state["mem_episodic_store"] = []
        st.session_state["mem_episodic_output"] = None
        st.rerun()


def _render_semantic():
    st.subheader("Semantic Memory")
    st.markdown("""
Semantic memory stores **extracted facts and knowledge** -- entity
relationships, user preferences, domain knowledge. Unlike episodic
memory (which stores raw interactions), semantic memory stores
**distilled, structured information**.

- **Capacity:** unbounded
- **Retention:** permanent, updated over time
- **Use case:** knowledge base, user preferences, entity tracking
    """)

    sem_input = st.text_input(
        "Tell the agent something or ask a question:",
        value="My name is Alex. I prefer Python over JavaScript. I work at Acme Corp.",
        key="mem_sem_input",
    )

    if st.button("Process", key="mem_sem_send") and sem_input:
        with st.spinner("Extracting and storing facts..."):
            try:
                output, store = _run_semantic(sem_input)
                st.session_state["mem_semantic_output"] = output
                st.session_state["mem_semantic_store"] = store
            except Exception as e:
                st.error(f"Error: {e}")
        st.rerun()

    if st.session_state["mem_semantic_store"]:
        st.markdown("**Semantic Store (extracted facts):**")
        for entity, facts in st.session_state["mem_semantic_store"].items():
            with st.expander(f"Entity: {entity}"):
                for f in facts:
                    st.text(f"  - {f}")

    if st.session_state["mem_semantic_output"]:
        st.markdown("**Response:**")
        st.info(st.session_state["mem_semantic_output"])

    if st.button("Clear Semantic Store", key="mem_sem_clear"):
        st.session_state["mem_semantic_store"] = {}
        st.session_state["mem_semantic_output"] = None
        st.rerun()


def _render_procedural():
    st.subheader("Procedural Memory")
    st.markdown("""
Procedural memory stores **learned strategies and patterns** -- how
to solve specific types of problems. When the agent encounters a
similar problem, it retrieves the relevant procedure.

- **Capacity:** grows over time
- **Retention:** permanent
- **Use case:** consistent behavior, learned optimizations
    """)

    proc_input = st.text_input(
        "Describe a problem to solve:",
        value="How should we handle database connection timeouts in a web API?",
        key="mem_proc_input",
    )

    if st.button("Solve (and learn procedure)", key="mem_proc_send") and proc_input:
        with st.spinner("Solving and extracting procedure..."):
            try:
                output, store = _run_procedural(proc_input)
                st.session_state["mem_procedural_output"] = output
                st.session_state["mem_procedural_store"] = store
            except Exception as e:
                st.error(f"Error: {e}")
        st.rerun()

    if st.session_state["mem_procedural_store"]:
        st.markdown("**Procedural Store (learned strategies):**")
        for category, procedures in st.session_state["mem_procedural_store"].items():
            with st.expander(f"Category: {category}"):
                for p in procedures:
                    st.text(f"  - {p[:150]}")

    if st.session_state["mem_procedural_output"]:
        st.markdown("**Solution:**")
        st.info(st.session_state["mem_procedural_output"])


def _render_unified():
    st.subheader("Unified Memory Agent")
    st.markdown("""
In practice, an agent uses **all memory types together**:
- Buffer for immediate context
- Working memory for current reasoning
- Episodic for relevant past interactions
- Semantic for known facts
- Procedural for learned strategies

We demonstrate this by running a single query through a unified
agent that consults all stores.
    """)

    unified_input = st.text_input(
        "Ask the unified agent:",
        value="Based on everything we have discussed, what do you know about me and what would you recommend?",
        key="mem_unified_input",
    )

    if st.button("Run Unified Agent", key="mem_unified_btn"):
        with st.spinner("Consulting all memory stores..."):
            try:
                output = _run_unified(unified_input)
                st.session_state["mem_unified_output"] = output
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state["mem_unified_output"]:
        out = st.session_state["mem_unified_output"]
        for store_name, content in out.items():
            with st.expander(f"Memory consulted: {store_name}", expanded=(store_name == "final_response")):
                st.write(content)


# ── Implementations ──

def _run_working_memory(task: str):
    llm = get_llm(temperature=0.7)
    memory = []
    steps = []

    for label, instruction in [
        ("Gather facts", "List the key facts and constraints relevant to this decision."),
        ("Analyze tradeoffs", "Given the facts, analyze the pros and cons."),
        ("Form recommendation", "Given facts and analysis, make a clear recommendation."),
    ]:
        context = "\n".join(memory) if memory else "None yet"
        resp = llm.invoke([
            SystemMessage(content=f"We are a senior architect. {instruction}"),
            HumanMessage(content=f"Task: {task}\n\nWorking memory:\n{context}\n\nProvide 2-3 concise points."),
        ])
        entry = f"{label}: {resp.content.strip()}"
        memory.append(entry)
        steps.append({"label": label, "memory": list(memory)})

    return memory[-1], steps


def _run_episodic(query: str):
    store = st.session_state["mem_episodic_store"]

    # Build context from past episodes
    episode_context = ""
    if store:
        episode_context = "Past interactions:\n" + "\n".join(
            f"- [{ep['timestamp']}] User asked: {ep['query'][:100]} | Agent said: {ep['response'][:100]}"
            for ep in store[-5:]
        )

    resp = call_llm(
        f"{episode_context}\n\nCurrent query: {query}\n\nRespond helpfully. If past interactions are relevant, reference them.",
        system="We are a helpful assistant with memory of past interactions.",
        temperature=0.7,
    )

    store.append({
        "query": query,
        "response": resp,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })

    return resp, store


def _run_semantic(text: str):
    store = st.session_state["mem_semantic_store"]

    # Extract facts using LLM
    extraction = call_llm(
        f"Extract structured facts from this text. Return JSON: "
        f'{{"entities": {{"entity_name": ["fact1", "fact2"]}}}}\n\nText: {text}',
        temperature=0,
    )

    try:
        clean = extraction.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(clean)
        entities = parsed.get("entities", {})
        for entity, facts in entities.items():
            if entity not in store:
                store[entity] = []
            for f in facts:
                if f not in store[entity]:
                    store[entity].append(f)
    except Exception:
        pass

    # Generate response using semantic store
    facts_str = "\n".join(f"{e}: {', '.join(fs)}" for e, fs in store.items())
    resp = call_llm(
        f"Known facts:\n{facts_str}\n\nUser said: {text}\n\nRespond acknowledging what we know.",
        temperature=0.7,
    )

    return resp, store


def _run_procedural(problem: str):
    store = st.session_state["mem_procedural_store"]

    # Check for existing procedures
    existing = "\n".join(
        f"Category {cat}: {', '.join(p[:80] for p in procs)}"
        for cat, procs in store.items()
    ) if store else "None yet"

    # Solve the problem
    solution = call_llm(
        f"Known procedures:\n{existing}\n\nProblem: {problem}\n\n"
        f"Solve this problem. Then extract a reusable procedure from the solution.",
        system="We are a senior engineer. Provide concrete, actionable solutions.",
        temperature=0.7,
    )

    # Extract procedure
    proc_extraction = call_llm(
        f"From this solution, extract a reusable procedure. Return JSON: "
        f'{{"category": "short_category_name", "steps": ["step1", "step2"]}}\n\nSolution: {solution}',
        temperature=0,
    )

    try:
        clean = proc_extraction.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(clean)
        cat = parsed.get("category", "general")
        steps = parsed.get("steps", [])
        if cat not in store:
            store[cat] = []
        for s in steps:
            if s not in store[cat]:
                store[cat].append(s)
    except Exception:
        pass

    return solution, store


def _run_unified(query: str):
    results = {}

    # Buffer
    buf = st.session_state.get("mem_buffer_history", [])
    results["buffer"] = f"{len(buf)} messages in buffer. Last: {buf[-1]['content'][:100] if buf else 'empty'}"

    # Episodic
    eps = st.session_state.get("mem_episodic_store", [])
    results["episodic"] = f"{len(eps)} episodes stored." + (
        f" Most recent: {eps[-1]['query'][:80]}" if eps else ""
    )

    # Semantic
    sem = st.session_state.get("mem_semantic_store", {})
    facts_str = "; ".join(f"{e}: {', '.join(fs)}" for e, fs in sem.items()) if sem else "No facts stored."
    results["semantic"] = facts_str

    # Procedural
    proc = st.session_state.get("mem_procedural_store", {})
    proc_str = "; ".join(f"{c}: {len(ps)} steps" for c, ps in proc.items()) if proc else "No procedures stored."
    results["procedural"] = proc_str

    # Unified response
    context = "\n".join(f"[{k}] {v}" for k, v in results.items())
    resp = call_llm(
        f"Memory state:\n{context}\n\nQuery: {query}\n\n"
        f"Respond using all available memory. Reference specific facts if available.",
        system="We are an agent with unified memory. Use all memory stores to give the best response.",
        temperature=0.7,
    )
    results["final_response"] = resp

    return results
