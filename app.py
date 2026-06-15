import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np

st.set_page_config(
    page_title="AI Sentiment Detector",
    page_icon="😊",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_name
    )

    return tokenizer, model

tokenizer, model = load_model()

labels = [
    "negative",
    "neutral",
    "positive"
]

st.title("😊 AI Sentiment Detection App")
st.write("Enter any text and detect whether the sentiment is Positive, Neutral, or Negative.")

user_text = st.text_area(
    "Enter Text",
    height=150,
    placeholder="Type something here..."
)

if st.button("Analyze Sentiment"):

    if user_text.strip() == "":
        st.warning("Please enter some text.")

    else:

        inputs = tokenizer(
            user_text,
            return_tensors="tf",
            truncation=True,
            padding=True,
            max_length=128
        )

        outputs = model(**inputs)

        probabilities = tf.nn.softmax(
            outputs.logits,
            axis=-1
        ).numpy()[0]

        predicted_index = np.argmax(probabilities)

        predicted_label = labels[predicted_index]

        confidence = probabilities[predicted_index] * 100

        st.success(
            f"Detected Sentiment: {predicted_label.upper()}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Sentiment Probabilities")

        chart_data = {
            labels[i]: float(probabilities[i])
            for i in range(len(labels))
        }

        st.bar_chart(chart_data)

        st.subheader("Detailed Scores")

        for label, score in chart_data.items():
            st.write(
                f"**{label.capitalize()}** : {score*100:.2f}%"
            )
