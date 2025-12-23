import streamlit as st
import tempfile
import os

from test import chatbot   # your backend file

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Agentic Multi-Modal Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Agentic Multi-Modal Assistant")
st.caption("Text • PDF • Image • Audio • YouTube")

st.divider()

# --------------------------------------------------
# INPUTS
# --------------------------------------------------
user_text = st.text_area(
    "Enter your query",
    placeholder="Ask a question, request a summary, paste a YouTube link…",
    height=120
)

uploaded_file = st.file_uploader(
    "Upload a file (PDF / Image / Audio)",
    type=["pdf", "png", "jpg", "jpeg", "wav", "mp3"]
)

run_btn = st.button("🚀 Run Agent")

# --------------------------------------------------
# EXECUTION
# --------------------------------------------------
if run_btn:

    if not user_text and not uploaded_file:
        st.warning("Please enter text or upload a file.")
        st.stop()

    payload = {
        "input": user_text or ""
    }

    if uploaded_file:
        suffix = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        payload["file_path"] = file_path

    with st.spinner("Agent is thinking..."):
        try:
            result = chatbot.invoke(payload)

            st.subheader("✅ Output")
            st.markdown(result.get("output", "No output generated."))

        except Exception as e:
            st.error("❌ Error while running agent")
            st.exception(e)
