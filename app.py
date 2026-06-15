import streamlit as st
import PyPDF2
from transformers import pipeline
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #0f1117;
    }

    .stApp {
        background: linear-gradient(135deg, #0f1117 0%, #1a1d27 100%);
    }

    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
    }

    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }

    .hero span.accent {
        color: #7c6dfa;
    }

    .hero p {
        color: #8b8fa8;
        font-size: 1rem;
        margin-top: 0.3rem;
    }

    .chat-bubble-user {
        background: #7c6dfa;
        color: white;
        padding: 0.75rem 1.1rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0 0.5rem auto;
        max-width: 80%;
        width: fit-content;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .chat-bubble-bot {
        background: #1e2130;
        color: #d4d6e4;
        padding: 0.75rem 1.1rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem auto 0.5rem 0;
        max-width: 80%;
        width: fit-content;
        font-size: 0.95rem;
        line-height: 1.5;
        border: 1px solid #2a2d3e;
    }

    .status-box {
        background: #1e2130;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        color: #8b8fa8;
        font-size: 0.88rem;
    }

    .status-box strong {
        color: #7c6dfa;
    }

    div[data-testid="stFileUploader"] {
        background: #1e2130;
        border: 2px dashed #2a2d3e;
        border-radius: 12px;
        padding: 1rem;
    }

    div[data-testid="stFileUploader"]:hover {
        border-color: #7c6dfa;
    }

    .stTextInput > div > div > input {
        background: #1e2130 !important;
        border: 1px solid #2a2d3e !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        padding: 0.7rem 1rem !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #7c6dfa !important;
        box-shadow: 0 0 0 2px rgba(124,109,250,0.2) !important;
    }

    .stButton > button {
        background: #7c6dfa !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.4rem !important;
        font-weight: 600 !important;
        width: 100%;
        transition: opacity 0.2s;
    }

    .stButton > button:hover {
        opacity: 0.88 !important;
    }

    .stSpinner > div {
        border-top-color: #7c6dfa !important;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Load QA model (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_qa_model():
    return pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2"
    )


# ── Extract text from PDF ─────────────────────────────────────────────────────
def extract_pdf_text(uploaded_file):
    reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


# ── Answer question ───────────────────────────────────────────────────────────
def get_answer(qa_model, context, question):
    # roberta has a 512-token limit; chunk context if too long
    max_chars = 3000
    if len(context) > max_chars:
        # slide a window, pick best-scoring answer
        best = None
        for i in range(0, len(context), max_chars - 200):
            chunk = context[i: i + max_chars]
            try:
                result = qa_model(question=question, context=chunk)
                if best is None or result["score"] > best["score"]:
                    best = result
            except Exception:
                continue
        return best
    else:
        return qa_model(question=question, context=context)


# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📄 PDF <span class="accent">Chatbot</span></h1>
    <p>Upload any PDF and ask questions — get instant answers from your document.</p>
</div>
""", unsafe_allow_html=True)

# Upload section
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"], label_visibility="collapsed")

if uploaded_file:
    if uploaded_file.name != st.session_state.pdf_name:
        with st.spinner("Reading PDF..."):
            text = extract_pdf_text(uploaded_file)
        if text:
            st.session_state.pdf_text = text
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.chat_history = []
            st.success(f"✅ **{uploaded_file.name}** loaded — {len(text):,} characters extracted.")
        else:
            st.error("Could not extract text. Try a different PDF.")

    if st.session_state.pdf_text:
        st.markdown(f"""
        <div class="status-box">
            📂 Active document: <strong>{st.session_state.pdf_name}</strong>
            &nbsp;·&nbsp; {len(st.session_state.pdf_text):,} chars
        </div>
        """, unsafe_allow_html=True)

        # Chat history
        if st.session_state.chat_history:
            st.markdown("---")
            for turn in st.session_state.chat_history:
                st.markdown(f'<div class="chat-bubble-user">🧑 {turn["question"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-bubble-bot">🤖 {turn["answer"]}</div>', unsafe_allow_html=True)
            st.markdown("---")

        # Input
        with st.form("qa_form", clear_on_submit=True):
            question = st.text_input(
                "Ask a question",
                placeholder="e.g. What is the main topic of this document?",
                label_visibility="collapsed"
            )
            submitted = st.form_submit_button("Ask →")

        if submitted and question.strip():
            with st.spinner("Finding answer..."):
                qa_model = load_qa_model()
                result = get_answer(qa_model, st.session_state.pdf_text, question.strip())

            if result and result.get("score", 0) > 0.05:
                answer = result["answer"]
                confidence = round(result["score"] * 100, 1)
                display_answer = f"{answer}  *(confidence: {confidence}%)*"
            else:
                display_answer = "I couldn't find a clear answer in the document. Try rephrasing your question."

            st.session_state.chat_history.append({
                "question": question.strip(),
                "answer": display_answer
            })
            st.rerun()

        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

else:
    st.markdown("""
    <div class="status-box" style="text-align:center; padding: 2rem;">
        ⬆️ Upload a PDF above to get started
    </div>
    """, unsafe_allow_html=True)
