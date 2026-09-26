# Agent Playground

Interactive Streamlit app for exploring LangGraph, agent design, memory systems, and context engineering.

## Features

| Tab | What It Covers |
|-----|---------------|
| **LangGraph Basics** | Linear graphs, conditional branching, multi-step pipelines with state |
| **Agent Design** | ReAct agent, custom tool loop, multi-agent supervisor pattern |
| **Scratchpad** | Working memory visualization -- see how the scratchpad updates at each node |
| **Checkpointing** | MemorySaver, persistent conversations, checkpoint inspector, thread management |
| **Memory Systems** | Buffer, working, episodic, semantic, procedural memory + unified agent |
| **Context Engineering** | Tool loadout, quarantine, pruning, summarization, offloading (think tool) |

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Configuration

- Open the app in your browser
- In the sidebar, choose **Gemini** or **OpenAI** as the LLM provider
- Enter your API key and click **Connect**
- All tabs become functional once connected

### Supported Models

| Provider | Model |
|----------|-------|
| Gemini | gemini-2.0-flash |
| OpenAI | gpt-4o |

## Project Structure

```
app.py              -- Main Streamlit entry point
utils.py            -- Shared LLM setup, session state helpers
tabs/
  tab_langgraph_basics.py
  tab_agent_design.py
  tab_scratchpad.py
  tab_checkpointing.py
  tab_memory_systems.py
  tab_context_engineering.py
```

## License

Proprietary — all rights reserved. This code is published for viewing and evaluation only; no use, copying, modification, redistribution, commercial use, or use as AI/ML training data without written permission. See [LICENSE](LICENSE). Commercial licensing: aianytime07@gmail.com · sonu@aianytime.net.
