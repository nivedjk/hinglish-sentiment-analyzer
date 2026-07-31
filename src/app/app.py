import os
import sys

import streamlit as st
import pandas as pd
sys.path.append(os.path.abspath("."))

from src.app.predictor import predict_sentiment
from src.app.batch_predictor import predict_csv


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Hinglish Sentiment Analyzer",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# Load CSS
# ==========================================

CSS_PATH = os.path.join(
    os.path.dirname(__file__),
    "assets",
    "style.css",
)

with open(CSS_PATH, "r", encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True,
    )


# ==========================================
# Session State
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

if "example_text" not in st.session_state:
    st.session_state.example_text = ""


# ==========================================
# Sidebar
# ==========================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🤖 Model Information</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Model**")
    st.write("Google MuRIL")

    st.markdown("**Task**")
    st.write("Hinglish Sentiment Classification")

    st.markdown("**Classes**")
    st.write("Negative • Neutral • Positive")

    st.markdown("**Training Samples**")
    st.write("13,599")

    st.markdown("**Validation Samples**")
    st.write("1,700")

    st.markdown("**Validation Accuracy**")
    st.write("64.65%")

    st.markdown("**Validation F1**")
    st.write("64.70%")

    st.divider()

    st.caption(
        "Fine-tuned Transformer model for "
        "Hinglish social-media sentiment."
    )


# ==========================================
# Hero Section
# ==========================================

st.markdown(
    """
    <div class="hero">
        <h1 class="hero-title">😊 Hinglish Sentiment Analyzer</h1>
        <p class="hero-subtitle">
            Analyze sentiment in Hinglish text using a
            fine-tuned Google MuRIL Transformer model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
# ==========================================
# Batch Prediction
# ==========================================

with st.expander("📁 Batch Prediction — Upload CSV"):

    st.write(
        "Upload a CSV containing a column named "
        "`text` to analyze multiple sentences."
    )

    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"],
    )

    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(
                uploaded_file
            )

            st.write(
                f"Loaded {len(batch_df)} rows."
            )

            st.dataframe(
                batch_df.head(),
                use_container_width=True,
            )

            if "text" not in batch_df.columns:

                st.error(
                    "CSV must contain a column named 'text'."
                )

            else:

                if st.button(
                    "🚀 Analyze CSV",
                    key="batch_predict",
                ):

                    with st.spinner(
                        "Analyzing CSV..."
                    ):

                        result_df = predict_csv(
                            batch_df,
                            "text",
                        )

                    st.success(
                        f"Completed {len(result_df)} predictions."
                    )

                    st.dataframe(
                        result_df,
                        use_container_width=True,
                    )

                    csv_data = result_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="⬇️ Download Predictions",
                        data=csv_data,
                        file_name="sentiment_predictions.csv",
                        mime="text/csv",
                    )

        except Exception as e:

            st.error(
                f"Error processing CSV: {e}"
            )
# ==========================================
# Input Section
# ==========================================

st.markdown(
    """
    <div class="card">
        <div class="card-title">
            Enter your text
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


text = st.text_area(
    "Text",
    value=st.session_state.example_text,
    placeholder="Example: Ye movie bahut mast thi...",
    height=140,
    label_visibility="collapsed",
)


# ==========================================
# Example Text
# ==========================================

st.markdown("**Try an example:**")

example_columns = st.columns(3)

examples = [
    "Ye movie bahut mast thi",
    "Aaj weather theek hai",
    "Worst service ever",
]

for column, example in zip(example_columns, examples):

    with column:

        if st.button(
            example,
            use_container_width=True,
        ):

            st.session_state.example_text = example
            st.rerun()


# ==========================================
# Analyze Sentiment
# ==========================================

if st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True,
):

    if not text or not text.strip():

        st.warning("Please enter some text first.")

    else:

        with st.spinner("Analyzing sentiment..."):

            result = predict_sentiment(text)

        # ==================================
        # Save History
        # ==================================

        st.session_state.history.insert(
            0,
            {
                "text": text,
                "result": result,
            },
        )

        # ==================================
        # Prediction Result
        # ==================================

        st.markdown(
            """
            <div class="prediction-card">
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="prediction-label">
                {result["label"]}
            </div>

            <div class="confidence">
                Confidence: {result["confidence"] * 100:.2f}%
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        # ==================================
        # Additional Metrics
        # ==================================

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:

            st.metric(
                "Confidence Level",
                result["confidence_level"],
            )

        with metric_col2:

            st.metric(
                "Inference Time",
                f"{result['inference_time_ms']:.1f} ms",
            )

        # ==================================
        # Probability Distribution
        # ==================================

        st.subheader("Probability Distribution")

        probabilities = result["probabilities"]

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "😡 Negative",
                f"{probabilities['negative'] * 100:.2f}%",
            )

            st.progress(
                probabilities["negative"]
            )

        with col2:

            st.metric(
                "😐 Neutral",
                f"{probabilities['neutral'] * 100:.2f}%",
            )

            st.progress(
                probabilities["neutral"]
            )

        with col3:

            st.metric(
                "😊 Positive",
                f"{probabilities['positive'] * 100:.2f}%",
            )

            st.progress(
                probabilities["positive"]
            )


# ==========================================
# Prediction History
# ==========================================

if st.session_state.history:

    st.divider()

    st.subheader("Prediction History")

    for item in st.session_state.history[:10]:

        result = item["result"]

        with st.container():

            st.markdown(
                f"**{item['text']}**"
            )

            st.write(
                f"{result['label']} • "
                f"{result['confidence'] * 100:.2f}% confidence"
            )

            st.divider()


# ==========================================
# Footer
# ==========================================

st.markdown(
    """
    <div class="footer">
        Hinglish Sentiment Analyzer • Google MuRIL • PyTorch
    </div>
    """,
    unsafe_allow_html=True,
)