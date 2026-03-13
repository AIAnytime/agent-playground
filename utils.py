"""Shared utilities for the LangGraph Playground app."""

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage


def init_session_defaults():
    """Initialize all session state defaults once."""
    defaults = {
        "api_provider": "Gemini",
        "api_key": "",
        "llm_ready": False,
        # Tab-specific state keys are initialized in each tab
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_sidebar():
    """Render the sidebar with API key input and model selection."""
    with st.sidebar:
        st.title("LangGraph Playground")
        st.markdown("---")
        st.subheader("Configuration")

        provider = st.radio(
            "LLM Provider",
            ["Gemini", "OpenAI"],
            index=0 if st.session_state.get("api_provider", "Gemini") == "Gemini" else 1,
            key="provider_radio",
        )
        st.session_state["api_provider"] = provider

        if provider == "Gemini":
            model_name = "gemini-2.0-flash"
            st.caption(f"Model: {model_name}")
        else:
            model_name = "gpt-4o"
            st.caption(f"Model: {model_name}")

        api_key = st.text_input(
            f"{provider} API Key",
            type="password",
            value=st.session_state.get("api_key", ""),
            key="api_key_input",
        )

        if api_key and api_key != st.session_state.get("api_key", ""):
            st.session_state["api_key"] = api_key
            st.session_state["llm_ready"] = False

        if api_key:
            if st.button("Connect", use_container_width=True):
                with st.spinner("Testing connection..."):
                    try:
                        test_llm = get_llm(api_key, provider, temperature=0)
                        resp = test_llm.invoke("Reply with exactly: OK")
                        st.session_state["llm_ready"] = True
                        st.session_state["api_key"] = api_key
                        st.success("Connected.")
                    except Exception as e:
                        st.session_state["llm_ready"] = False
                        st.error(f"Connection failed: {e}")
        else:
            st.info(f"Enter your {provider} API key to begin.")

        if st.session_state.get("llm_ready"):
            st.markdown("---")
            st.caption(f"Provider: {provider} | Model: {model_name}")
            st.caption("Status: Connected")

        st.markdown("---")
        st.markdown(
            "**Tabs**\n\n"
            "1. LangGraph Basics\n"
            "2. Agent Design\n"
            "3. Scratchpad\n"
            "4. Checkpointing\n"
            "5. Memory Systems\n"
            "6. Context Engineering"
        )


def get_llm(api_key=None, provider=None, temperature=0.7):
    """Return a configured LLM instance."""
    api_key = api_key or st.session_state.get("api_key", "")
    provider = provider or st.session_state.get("api_provider", "Gemini")

    if not api_key:
        raise ValueError("No API key configured.")

    if provider == "Gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=temperature,
        )
    else:
        return ChatOpenAI(
            model="gpt-4o",
            api_key=api_key,
            temperature=temperature,
        )


def call_llm(prompt: str, system: str = "", temperature: float = 0.7) -> str:
    """Convenience: send a prompt, get a string back."""
    msgs = []
    if system:
        msgs.append(SystemMessage(content=system))
    msgs.append(HumanMessage(content=prompt))
    llm = get_llm(temperature=temperature)
    return llm.invoke(msgs).content.strip()


def require_api_key():
    """Check that API key is set. Returns True if ready, False otherwise."""
    if not st.session_state.get("llm_ready"):
        st.warning("Please enter your API key in the sidebar and click Connect.")
        return False
    return True
