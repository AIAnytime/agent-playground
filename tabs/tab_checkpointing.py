"""Tab 4: Checkpointing & MemorySaver -- persistence, replay, thread management."""

import streamlit as st
import json
import uuid
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from utils import get_llm, call_llm, require_api_key


def _init_state():
    defaults = {
        "ckpt_memory_saver": None,
        "ckpt_thread_id": None,
        "ckpt_conversation": [],
        "ckpt_checkpoints_log": [],
        "ckpt_thread_list": [],
        "ckpt_replay_output": None,
        "ckpt_fork_output": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


class ConvState(TypedDict):
    messages: Annotated[list, lambda a, b: a + b]
    summary: str
    turn_count: int


def render():
    _init_state()
    st.header("Checkpointing and MemorySaver")

    st.markdown("""
**Checkpointing** saves the full graph state after every node execution.
This enables:

- **Persistence** -- conversations survive app restarts (with a persistent backend)
- **Replay** -- re-run from any previous state
- **Forking** -- branch a conversation to explore alternatives
- **Time travel** -- inspect what the state looked like at any past step

LangGraph provides `MemorySaver` (in-memory) for development and
`SqliteSaver`/`PostgresSaver` for production.

```
Step 1       Step 2       Step 3
  |            |            |
  v            v            v
[ckpt-1] -> [ckpt-2] -> [ckpt-3]   <-- each is a full state snapshot
                            |
                       [ckpt-3b]    <-- forked branch
```
    """)

    if not require_api_key():
        return

    # ── Section 1: Persistent Conversation ──
    st.subheader("1. Conversation with Checkpointing")
    st.markdown("""
Each message exchange is checkpointed. We can send multiple messages
and the agent remembers the full conversation -- because the graph
state (including all messages) is saved after each invocation.
    """)

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("New Thread", key="ckpt_new_thread"):
            tid = str(uuid.uuid4())[:8]
            st.session_state["ckpt_thread_id"] = tid
            st.session_state["ckpt_conversation"] = []
            st.session_state["ckpt_checkpoints_log"] = []
            st.session_state["ckpt_memory_saver"] = MemorySaver()
            if tid not in [t["id"] for t in st.session_state["ckpt_thread_list"]]:
                st.session_state["ckpt_thread_list"].append({"id": tid, "messages": 0})
            st.rerun()

    if not st.session_state["ckpt_thread_id"]:
        st.info("Click 'New Thread' to start a checkpointed conversation.")
        return

    st.caption(f"Thread ID: `{st.session_state['ckpt_thread_id']}`")

    # Show conversation history
    for msg in st.session_state["ckpt_conversation"]:
        role = msg["role"]
        with st.chat_message("user" if role == "human" else "assistant"):
            st.write(msg["content"])

    user_input = st.chat_input("Send a message...", key="ckpt_chat_input")
    if user_input:
        st.session_state["ckpt_conversation"].append({"role": "human", "content": user_input})

        with st.spinner("Agent responding..."):
            try:
                response, ckpt_info = _send_message(user_input)
                st.session_state["ckpt_conversation"].append({"role": "ai", "content": response})
                st.session_state["ckpt_checkpoints_log"].append(ckpt_info)

                # Update thread list
                for t in st.session_state["ckpt_thread_list"]:
                    if t["id"] == st.session_state["ckpt_thread_id"]:
                        t["messages"] = len(st.session_state["ckpt_conversation"])
            except Exception as e:
                st.error(f"Error: {e}")
        st.rerun()

    # ── Section 2: Checkpoint Inspector ──
    st.markdown("---")
    st.subheader("2. Checkpoint Inspector")
    st.markdown("Each checkpoint stores the full state. We can inspect what was saved.")

    if st.session_state["ckpt_checkpoints_log"]:
        for i, ckpt in enumerate(st.session_state["ckpt_checkpoints_log"]):
            with st.expander(f"Checkpoint {i+1} -- Turn {ckpt.get('turn', i+1)}"):
                st.json({
                    "thread_id": ckpt.get("thread_id", ""),
                    "turn_count": ckpt.get("turn", 0),
                    "message_count": ckpt.get("msg_count", 0),
                    "has_summary": ckpt.get("has_summary", False),
                    "state_keys": ckpt.get("state_keys", []),
                })
    else:
        st.caption("No checkpoints yet. Send a message to create one.")

    # ── Section 3: Thread Management ──
    st.markdown("---")
    st.subheader("3. Thread Management")
    st.markdown("""
Each `thread_id` is an independent conversation. Switching threads loads
a completely different state -- this is how production apps handle
multiple users or conversation sessions.
    """)

    if st.session_state["ckpt_thread_list"]:
        for t in st.session_state["ckpt_thread_list"]:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                active = " (active)" if t["id"] == st.session_state["ckpt_thread_id"] else ""
                st.text(f"Thread {t['id']}{active} -- {t['messages']} messages")
            with col_b:
                if t["id"] != st.session_state["ckpt_thread_id"]:
                    if st.button("Switch", key=f"ckpt_switch_{t['id']}"):
                        st.session_state["ckpt_thread_id"] = t["id"]
                        st.session_state["ckpt_conversation"] = []
                        st.session_state["ckpt_checkpoints_log"] = []
                        st.rerun()
    else:
        st.caption("No threads created yet.")

    # ── Section 4: Concepts ──
    st.markdown("---")
    st.subheader("4. How It Works")
    st.markdown("""
| Concept | Description |
|---------|-------------|
| **MemorySaver** | In-memory checkpoint backend. State is lost on restart. Good for development. |
| **SqliteSaver** | Persistent checkpoint backend using SQLite. Survives restarts. |
| **thread_id** | Identifies a conversation. Same graph, different threads = independent states. |
| **checkpoint_id** | Identifies a specific state snapshot within a thread. |
| **Replay** | Re-invoke the graph from a past checkpoint to reproduce or debug behavior. |
| **Fork** | Create a new thread starting from an existing checkpoint -- branch the conversation. |

**In production**, we would use `PostgresSaver` or `SqliteSaver` with a
`thread_id` per user session. The graph itself is stateless -- all state
lives in the checkpoint store.
    """)


def _get_or_create_graph():
    """Build and compile the conversational graph with checkpointing."""
    if st.session_state["ckpt_memory_saver"] is None:
        st.session_state["ckpt_memory_saver"] = MemorySaver()

    memory = st.session_state["ckpt_memory_saver"]
    llm = get_llm(temperature=0.7)

    def chat_node(state: ConvState):
        turn = state.get("turn_count", 0) + 1
        sys_content = "We are a helpful assistant. Be concise."
        if state.get("summary"):
            sys_content += f"\n\nConversation summary so far:\n{state['summary']}"

        msgs = [SystemMessage(content=sys_content)] + state["messages"][-10:]
        resp = llm.invoke(msgs)

        return {
            "messages": [resp],
            "turn_count": turn,
        }

    def maybe_summarize(state: ConvState):
        if len(state["messages"]) > 12:
            history = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content[:150]}"
                for m in state["messages"][-12:]
            )
            summary = call_llm(
                f"Summarize this conversation briefly:\n{history}",
                temperature=0.3,
            )
            return {"summary": summary, "messages": state["messages"][-4:]}
        return {}

    g = StateGraph(ConvState)
    g.add_node("chat", chat_node)
    g.add_node("summarize", maybe_summarize)
    g.add_edge(START, "chat")
    g.add_edge("chat", "summarize")
    g.add_edge("summarize", END)

    return g.compile(checkpointer=memory)


def _send_message(user_input: str):
    app = _get_or_create_graph()
    config = {"configurable": {"thread_id": st.session_state["ckpt_thread_id"]}}

    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
    )

    # Extract response
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    response = ai_msgs[-1].content if ai_msgs else "No response."

    ckpt_info = {
        "thread_id": st.session_state["ckpt_thread_id"],
        "turn": result.get("turn_count", 0),
        "msg_count": len(result.get("messages", [])),
        "has_summary": bool(result.get("summary")),
        "state_keys": list(result.keys()),
    }

    return response, ckpt_info
