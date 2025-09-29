"""Streamlit app for serving the `my_model.h5` image classifier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

MODEL_PATH = Path("my_model.h5")
DEFAULT_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
IMAGE_SIZE = (32, 32)


def _lazy_load_keras():
    """Delay the TensorFlow import so the page can render helpful guidance if missing."""
    try:
        from tensorflow import keras  # type: ignore
    except ImportError as exc:  # pragma: no cover - handled interactively in Streamlit
        raise RuntimeError(
            "TensorFlow is required to load `my_model.h5`. Please ensure it is installed "
            "(for Streamlit Cloud, list `tensorflow` in `requirements.txt`)."
        ) from exc
    return keras


@st.cache_resource(show_spinner=True)
def load_model(model_path: Path):
    keras = _lazy_load_keras()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find model file at {model_path.resolve()}. Make sure `my_model.h5` is deployed with the app."
        )
    return keras.models.load_model(model_path)


def parse_class_names(raw_value: str | None, fallback: Iterable[str]) -> List[str]:
    if not raw_value:
        return list(fallback)
    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            return parsed  # type: ignore[return-value]
    except json.JSONDecodeError:
        pass

    # Fallback to comma-separated parsing if JSON fails.
    tokens = [token.strip() for token in raw_value.split(",") if token.strip()]
    return tokens or list(fallback)


def preprocess(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    resized = image.resize(IMAGE_SIZE)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def format_predictions(predictions: np.ndarray, class_names: List[str], top_k: int = 3):
    if predictions.ndim != 2 or predictions.shape[0] != 1:
        raise ValueError("Predictions are expected with shape (1, num_classes).")

    probabilities = predictions[0]
    indices = np.argsort(probabilities)[::-1][:top_k]
    return [
        {
            "label": class_names[idx] if idx < len(class_names) else f"Class {idx}",
            "score": float(probabilities[idx]),
        }
        for idx in indices
    ]


def main():
    st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon="🖼️", layout="wide")
    st.title("🖼️ CIFAR-10 Image Classifier")
    st.caption(
        "Upload an image and the Streamlit app will resize it to 32×32 and run it through the saved Keras model."
    )

    with st.sidebar:
        st.header("Configuration")
        st.markdown(
            """
            - The bundled model was trained on **32×32 RGB images**.
            - Default class names follow the CIFAR-10 dataset.
            - Override the class labels by pasting a JSON array or comma-separated list.
            """
        )

        class_text = st.text_area(
            "Custom class labels",
            value=json.dumps(DEFAULT_CLASS_NAMES, indent=2),
            height=220,
            help="Optional: Paste a JSON array (e.g., [\"cat\", \"dog\"]) or a comma-separated list to override the labels.",
        )
        class_names = parse_class_names(class_text, DEFAULT_CLASS_NAMES)

        st.markdown(
            """
            **Tips**

            - For best results, upload clear pictures of a single object.
            - Try JPG or PNG images up to ~5 MB.
            - Predictions include a probability bar chart and top-k table.
            """
        )

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded is None:
        st.info("👆 Upload a JPG or PNG image to get started.")
        return

    image = Image.open(uploaded)
    st.subheader("Preview")
    st.image(image, caption="Original Upload", use_column_width=False, width=256)

    model = None
    error_placeholder = st.empty()
    with st.spinner("Loading model and running inference..."):
        try:
            model = load_model(MODEL_PATH)
            batch = preprocess(image)
            predictions = model.predict(batch)
        except Exception as exc:  # pragma: no cover - surfaced to user
            error_placeholder.error(f"⚠️ Unable to generate predictions: {exc}")
            st.stop()

    top_predictions = format_predictions(predictions, class_names, top_k=5)

    st.subheader("Top Predictions")
    st.table(
        [
            {"Class": item["label"], "Confidence": f"{item['score']*100:.2f}%"}
            for item in top_predictions
        ]
    )

    st.subheader("Probability Distribution")
    chart_data = pd.DataFrame(
        {
            "label": [
                class_names[idx] if idx < len(class_names) else f"Class {idx}"
                for idx in range(len(predictions[0]))
            ],
            "confidence": predictions[0],
        }
    ).set_index("label")
    st.bar_chart(chart_data)

    st.success("✅ Inference complete! Adjust the labels in the sidebar or upload another image to compare results.")


if __name__ == "__main__":  # pragma: no cover
    main()
