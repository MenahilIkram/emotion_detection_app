import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np

st.set_page_config(
    page_title="AI Emotion Detector",
    page_icon="😊",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_name = "bhadresh-savani/distilbert-base-uncased-emotion"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_name,
        from_pt=True
    )

    return tokenizer, model

tokenizer, model = load_model()

emotion_labels = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]

st.title("😊 AI Emotion Detection App")
st.write("Enter any text and detect the emotion using a Hugging Face TensorFlow model.")

user_text = st.text_area(
    "Enter Text",
    height=150,
    placeholder="Type something here..."
)

if st.button("Detect Emotion"):

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

        predicted_emotion = emotion_labels[predicted_index]
        confidence = probabilities[predicted_index] * 100

        st.success(
            f"Detected Emotion: {predicted_emotion.upper()}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Emotion Probabilities")

        chart_data = {
            emotion_labels[i]: float(probabilities[i])
            for i in range(len(emotion_labels))
        }

        st.bar_chart(chart_data)

        st.subheader("Detailed Scores")

        for emotion, score in chart_data.items():
            st.write(
                f"**{emotion.capitalize()}** : {score*100:.2f}%"
            )
