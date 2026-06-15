import streamlit as st
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image to Article",
    page_icon="✍️",
    layout="centered"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #f7f4ef; }

.hero {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    color: #1a1a2e;
    margin-bottom: 0.3rem;
}
.hero h1 span { color: #c0392b; }
.hero p { color: #6b6b80; font-size: 1rem; }

.step-badge {
    display: inline-block;
    background: #1a1a2e;
    color: #f7f4ef;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 0.25rem 0.7rem;
    border-radius: 4px;
    margin-bottom: 0.5rem;
}

.caption-box {
    background: #1a1a2e;
    color: #f7f4ef;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    font-size: 0.97rem;
    margin: 1rem 0;
    line-height: 1.6;
}
.caption-box span { color: #e74c3c; font-weight: 600; }

.article-box {
    background: #ffffff;
    border-radius: 12px;
    padding: 2rem 2.2rem;
    margin-top: 1.5rem;
    border-left: 4px solid #c0392b;
    box-shadow: 0 2px 16px rgba(0,0,0,0.07);
    color: #1a1a2e;
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    line-height: 1.9;
}

div[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 2px dashed #c0392b;
    border-radius: 12px;
    padding: 1rem;
}

.stButton > button {
    background: #1a1a2e !important;
    color: #f7f4ef !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.6rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover { background: #c0392b !important; }

.stSelectbox > div > div {
    background: #ffffff !important;
    border-radius: 8px !important;
    border: 1px solid #ddd !important;
}

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_captioning_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource(show_spinner=False)
def load_text_generator():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )


# ── Caption image ─────────────────────────────────────────────────────────────
def caption_image(image: Image.Image, processor, model) -> str:
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60)
    return processor.decode(out[0], skip_special_tokens=True)


# ── Generate article ──────────────────────────────────────────────────────────
def generate_article(generator, caption: str, style: str) -> str:
    style_prompts = {
        "Blog Post": f"Write a detailed and engaging blog post about: {caption}. Include an introduction, main points, and conclusion.",
        "News Article": f"Write a professional news article about: {caption}. Use journalistic style with facts.",
        "Social Media": f"Write creative and catchy social media captions about: {caption}. Make it trendy and engaging.",
        "Academic": f"Write a formal academic article discussing: {caption}. Include analysis and structured paragraphs.",
    }
    prompt = style_prompts.get(style, style_prompts["Blog Post"])
    result = generator(
        prompt,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.85,
        repetition_penalty=1.3
    )
    return result[0]["generated_text"]


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Image to <span>Article</span></h1>
    <p>Upload any image — get a full article written automatically.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="step-badge">Step 1 — Upload Image</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

if uploaded:
    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    st.image(image, use_column_width=True, caption="Your uploaded image")

    st.markdown('<br><div class="step-badge">Step 2 — Choose Article Style</div>', unsafe_allow_html=True)
    style = st.selectbox(
        "Style",
        ["Blog Post", "News Article", "Social Media", "Academic"],
        label_visibility="collapsed"
    )

    st.markdown('<br><div class="step-badge">Step 3 — Generate</div>', unsafe_allow_html=True)
    if st.button("✍️ Write Article"):

        with st.spinner("Understanding your image..."):
            processor, caption_model = load_captioning_model()
            caption = caption_image(image, processor, caption_model)

        st.markdown(f"""
        <div class="caption-box">
            🔍 <span>Image understood as:</span> {caption}
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Writing your article..."):
            generator = load_text_generator()
            article = generate_article(generator, caption, style)

        st.markdown(f"""
        <div class="article-box">
            {article}
        </div>
        """, unsafe_allow_html=True)

        st.download_button(
            label="⬇️ Download Article",
            data=article,
            file_name="article.txt",
            mime="text/plain"
        )

else:
    st.markdown("""
    <div style="text-align:center; padding: 2.5rem; background:#fff; border-radius:12px; color:#aaa; margin-top:1rem;">
        ⬆️ Upload an image above to get started
    </div>
    """, unsafe_allow_html=True)
