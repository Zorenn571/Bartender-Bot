import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from bartenderbot import app  # importing the compiled LangGraph app from core file

st.set_page_config(page_title="Bartender Bot", page_icon="🍸")

st.title("Bartender Bot")
st.caption("Ask for cocktail ideas and what to buy from the store.")

# Thread/session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = os.getenv("THREAD_ID", "streamlit-thread-1")

if "chat" not in st.session_state:
    st.session_state.chat = []

# Render history
for role, content in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(content)

# Chat input
user_text = st.chat_input("Ask about a drink, ingredients, or store items…")
if user_text:
    # Show user message immediately
    st.session_state.chat.append(("user", user_text))
    with st.chat_message("user"):
        st.markdown(user_text)

    # Call LangGraph app
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    result = app.invoke({"messages": [HumanMessage(user_text)]}, config)

    bot_reply = result["messages"][-1].content if result["messages"] else "…"
    st.session_state.chat.append(("assistant", bot_reply))

    with st.chat_message("assistant"):
        st.markdown(bot_reply)

if st.button("Clear chat"):
    st.session_state.chat = []
    st.rerun()

