import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/query"

# -------------------------
# INITIAL STATE
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

st.set_page_config(page_title="Cricket Chatbot", layout="centered")

# -------------------------
# HEADER / HEADING
# -------------------------
st.markdown(
    """
    <h1 style='text-align:center; margin-bottom:0;'>🏏 Cricket Rules Assistant</h1>
    <p style='text-align:center; color:gray; margin-top:5px; font-size:18px;'>
        Ask any question about Cricket Laws — Powered by RAG + FastAPI
    </p>
    <hr style='margin-top:10px;'>
    """,
    unsafe_allow_html=True
)


# -------------------------
# CHAT BUBBLE RENDERING
# -------------------------
def render_user_bubble(text):
    st.markdown(
        f"""
        <div style='background:#DCF8C6;padding:10px;border-radius:10px;margin:8px 0;text-align:right;'>
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )

def render_assistant_bubble(text):
    st.markdown(
        f"""
        <div style='background:#F1F0F0;padding:10px;border-radius:10px;margin:8px 0;text-align:left;'>
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )

def render_citations(citations):
    if not citations:
        return

    with st.expander("📚 Sources & Citations"):
        for c in citations:
            source = c.get("source", "Unknown Source")
            page = c.get("page", "N/A")
            score = c.get("relevance_score", None)

            st.markdown(
                f"""
                <div style='padding:8px;margin:5px 0;border-radius:8px;
                background:#FFFFFF;border:1px solid #DDD;'>
                    <b>📄 Source:</b> {source}<br>
                    <b>📘 Page:</b> {page}<br>
                    <b>⭐ Relevance:</b> {score}
                </div>
                """,
                unsafe_allow_html=True
            )

# -------------------------
# DISPLAY CHAT HISTORY
# -------------------------
for message in st.session_state.messages:
    if message["role"] == "user":
        render_user_bubble(message["content"])

    else:
        render_assistant_bubble(message["content"])
        render_citations(message.get("citations", []))

# -------------------------
# SEND MESSAGE LOGIC
# -------------------------
def send_message():
    user_input = st.session_state.input_text.strip()
    if not user_input:
        return

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    payload = {
        "question": user_input,
        "session_id": st.session_state.session_id
    }

    try:
        response = requests.post(API_URL, json=payload).json()

        answer = response.get("answer", "⚠ No answer returned")
        citations = response.get("citations", [])
        session_id = response.get("session_id", None)

        if session_id:
            st.session_state.session_id = session_id

        # Add assistant message with citations
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "citations": citations
        })

    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠ API Error: {e}",
            "citations": []
        })

    # Clear input field
    st.session_state.input_text = ""

# -------------------------
# INPUT BOX (bottom)
# -------------------------
st.text_input(
    "Type your message...",
    key="input_text",
    placeholder="Ask something...",
    on_change=send_message
)
