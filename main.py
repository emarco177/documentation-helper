import streamlit as st
from backend.core import run_llm


def show_messages(text):
    messages_str = [
        f"{_['role']}: {_['content']}" for _ in st.session_state["messages"][1:]
    ]
    text.text_area("Messages", value=str("\n".join(messages_str)), height=400)


BASE_PROMPT = [{"role": "system", "content": "You are a helpful assistant."}]

if __name__ == "__main__":
    if "messages" not in st.session_state:
        st.session_state["messages"] = BASE_PROMPT

    st.header("LangChain Udemy Course- Helper Bot")
    text = st.empty()
    show_messages(text)

    prompt = st.text_input("Prompt", value="Enter your message here...")

    if st.button("Send"):
        with st.spinner("Generating response..."):
            st.session_state["messages"] += [{"role": "user", "content": prompt}]
            message_response = run_llm(query=prompt)
            formatted_response = message_response["result"]
            st.session_state["messages"] += [
                {"role": "system", "content": formatted_response}
            ]
            show_messages(text)

    if st.button("Clear"):
        st.session_state["messages"] = BASE_PROMPT
        show_messages(text)
