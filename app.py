import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

from model import (
    PRIMARY_MODEL_PATH,
    EXPERIMENT_METRICS,
     get_class_names,
    load_trained_model,
    validate_and_open_image,
    predict_disease,
)
from gradcam import generate_gradcam_overlay

st.set_page_config(
    page_title="Automated Fish Disease Detection | Deep Learning",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #0a2540 0%, #0d6efd 100%);
        color: white;
        padding: 1.6rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: white !important;
        font-size: 1.85rem;
        margin-bottom: 0.35rem;
    }
    .main-header p {
        color: #e2e8f0;
        font-size: 1rem;
        margin: 0;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #0d6efd;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .metric-card h4 {
        margin: 0 0 0.3rem 0;
        color: #475569;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-card h2 {
        margin: 0;
        color: #0f172a !important;
        font-size: 1.65rem;
    }
    .disclaimer-box {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 4px solid #f59e0b;
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        color: #92400e;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

SAMPLE_IMAGES = {
    "Healthy Fish (Sample)": os.path.join("assets", "sample_healthy_fish.png"),
    "Bacterial Red Disease (Sample)": os.path.join("assets", "sample_bacterial_red_disease.png"),
    "Bacterial Diseases – Aeromoniasis (Sample)": os.path.join("assets", "sample_aeromoniasis.png"),
    "Bacterial Gill Disease (Sample)": os.path.join("assets", "sample_bacterial_gill_disease.png"),
    "Saprolegniasis (Sample)": os.path.join("assets", "sample_saprolegniasis.png"),
    "Parasitic Diseases (Sample)": os.path.join("assets", "sample_parasitic_disease.png"),
    "White Tail Disease (Sample)": os.path.join("assets", "sample_white_tail_disease.png"),
}

with st.sidebar:
    st.markdown("## 🐟 Aquaculture AI")
    st.caption("B.Tech Final Year Project")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home & Disease Predictor",
            "📊 Model Performance",
            "ℹ️ About Project",
        ],
    )

    st.markdown("---")
    st.markdown("### System Status")
    st.success("Active: `MobileNetV2 (160×160)`")
    st.markdown(
        """
        **Primary Model Specs:**
        - **Architecture:** MobileNetV2 (Transfer Learning)
        - **Input Resolution:** 160 × 160 RGB
        - **Preprocessing:** Internal Rescaling Layer
        - **Test Accuracy:** 86.80%
        - **Classes:** 7
        """
    )


def render_probability_chart(probabilities: dict, predicted_class: str):
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    classes = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    colors = ["#0d6efd" if c == predicted_class else "#94a3b8" for c in classes]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.barh(classes[::-1], scores[::-1], color=colors[::-1], height=0.6)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Probability (%)")
    ax.set_title("Class Probability Distribution (All 7 Classes)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    for bar, score in zip(bars, scores[::-1]):
        ax.text(
            bar.get_width() + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}%",
            va="center",
            fontsize=9.5,
            fontweight="bold" if score == max(scores) else "normal",
        )

    plt.tight_layout()
    return fig


if page == "🏠 Home & Disease Predictor":
    st.markdown(
        """
        <div class="main-header">
            <h1>A Framework for Automated Fish Disease Detection in Aquaculture Using Deep Learning</h1>
            <p>Upload a freshwater fish image or select a built-in test specimen to detect diseases across 7 classes with MobileNetV2 transfer learning and Grad-CAM visual explainability.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Primary Architecture", "MobileNetV2", "Transfer Learning")
    m2.metric("Test Accuracy", "86.80%", "+41.03% vs Basic CNN")
    m3.metric("Disease Classes", "7 Classes", "South Asia Dataset")
    m4.metric("Explainability", "Grad-CAM", "Spatial Heatmap")

    st.markdown("---")

    st.subheader("1. Select or Upload a Fish Image for Diagnosis")
    input_mode = st.radio(
        "Choose Input Source:",
        ["📤 Upload Fish Image (JPG, JPEG, PNG)", "🧪 Use Built-in Sample Fish Images (Instant Demo)"],
        horizontal=True,
    )

    pil_image = None
    image_label = None

    if input_mode == "📤 Upload Fish Image (JPG, JPEG, PNG)":
        uploaded_file = st.file_uploader(
            "Choose a fish image file (Supported formats: JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file is not None:
            try:
                pil_image = validate_and_open_image(uploaded_file)
                image_label = uploaded_file.name
            except ValueError as err:
                st.error(str(err))
                st.stop()
    else:
        selected_sample = st.selectbox(
            "Select a sample fish image across the 7 dataset classes:",
            list(SAMPLE_IMAGES.keys()),
        )
        sample_path = SAMPLE_IMAGES[selected_sample]
        if os.path.exists(sample_path):
            pil_image = Image.open(sample_path).convert("RGB")
            image_label = os.path.basename(sample_path)

    if pil_image is not None:
        col_preview, col_action = st.columns([1.15, 1.85], gap="large")

        with col_preview:
            st.markdown("#### Image Preview")
            st.image(
                pil_image,
                caption=f"{image_label} ({pil_image.size[0]} × {pil_image.size[1]} px)",
                use_container_width=True,
            )
            predict_clicked = st.button(
                "🔍 Predict Disease",
                type="primary",
                use_container_width=True,
            )

        with col_action:
            if not predict_clicked:
                st.info(
                    "Click **Predict Disease** on the left to run the MobileNetV2 model, "
                    "compute class probabilities for all seven categories, and generate the Grad-CAM heatmap."
                )
            else:
                with st.spinner("Running MobileNetV2 inference and computing Grad-CAM heatmap..."):
                    try:
                        model, is_auto_init = load_trained_model(PRIMARY_MODEL_PATH)
                        result = predict_disease(model, pil_image, is_auto_init=is_auto_init)
                        overlay_img, heatmap_img = generate_gradcam_overlay(
                            model=model,
                            original_pil_image=pil_image,
                            img_tensor=result["input_tensor"],
                            class_index=result["predicted_index"],
                        )
                    except Exception as exc:
                        st.error(f"Error during prediction or Grad-CAM computation: {exc}")
                        st.stop()

                pred_class = result["predicted_class"]
                confidence = result["confidence"]

                st.session_state["prediction_history"].insert(
                    0,
                    {
                        "Timestamp": datetime.now().strftime("%H:%M:%S"),
                        "File Name": image_label,
                        "Predicted Class": pred_class,
                        "Confidence (%)": f"{confidence:.2f}%",
                    },
                )

                c_res1, c_res2 = st.columns(2)
                with c_res1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h4>Predicted Classification</h4>
                            <h2>{pred_class}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c_res2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h4>Prediction Confidence</h4>
                            <h2>{confidence:.2f}%</h2>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("#### Prediction Explanation")
                st.write(result["explanation"])

        if predict_clicked:
            st.markdown("---")
            st.subheader("2. Explainable AI (Grad-CAM) & Probability Breakdown")

            g_col1, g_col2 = st.columns([1.2, 1.3], gap="large")

            with g_col1:
                st.markdown("#### Grad-CAM Visual Explainability")
                tab_overlay, tab_raw = st.tabs(["Superimposed Grad-CAM Overlay", "Standalone Activation Heatmap"])
                with tab_overlay:
                    st.image(
                        overlay_img,
                        caption=f"Grad-CAM Overlay — Regions influencing '{pred_class}'",
                        use_container_width=True,
                    )
                with tab_raw:
                    st.image(
                        heatmap_img,
                        caption="Raw Grad-CAM Activation Map (Red = High Influence, Blue = Low Influence)",
                        use_container_width=True,
                    )
                st.caption(
                    "Warm colors (red/yellow) indicate image regions with the highest positive gradient activation "
                    "for the predicted class inside the final convolutional feature map."
                )

            with g_col2:
                st.markdown("#### Probability Scores Across All 7 Classes")
                fig_prob = render_probability_chart(result["probabilities"], pred_class)
                st.pyplot(fig_prob, use_container_width=True)

                prob_df = pd.DataFrame(
                    [
                        {"Disease Class": k, "Probability (%)": f"{v:.2f}%"}
                        for k, v in sorted(
                            result["probabilities"].items(), key=lambda item: item[1], reverse=True
                        )
                    ]
                )
                with st.expander("View Exact Class Probability Table"):
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("3. Condition Overview & General Aquaculture Management")
            info = result["disease_info"]
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.markdown(f"**Condition Summary ({pred_class}):**")
                st.write(info["summary"])
                st.markdown("**Typical Visual Indicators:**")
                st.write(info["visual_cues"])
            with info_col2:
                st.markdown("**Basic Preventive & Management Suggestions:**")
                st.write(info["prevention"])

            st.markdown(
                """
                <div class="disclaimer-box">
                    <strong>Educational & Decision-Support Disclaimer:</strong> This deep learning system is an academic pattern-recognition tool developed for automated aquaculture screening. Image-based predictions depend on lighting, camera angle, and image clarity, and do not replace professional laboratory pathology or certified aquatic veterinary diagnosis.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("4. Session Prediction History")
    if len(st.session_state["prediction_history"]) == 0:
        st.caption("No predictions made during this session yet. Upload or select an image above to begin.")
    else:
        hist_df = pd.DataFrame(st.session_state["prediction_history"])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear Session History"):
            st.session_state["prediction_history"] = []
            st.rerun()


elif page == "📊 Model Performance":
    st.markdown(
        """
        <div class="main-header">
            <h1>Model Evaluation & Experimental Performance</h1>
            <p>Verified results on the Freshwater Fish Disease Aquaculture in South Asia test set (697 test images across 7 classes).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("1. Primary Model Metrics (MobileNetV2 Transfer Learning)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Accuracy", f"{EXPERIMENT_METRICS['mobilenetv2']['test_accuracy']:.2f}%")
    c2.metric("Weighted Precision", f"{EXPERIMENT_METRICS['mobilenetv2']['weighted_precision']:.2f}%")
    c3.metric("Weighted Recall", f"{EXPERIMENT_METRICS['mobilenetv2']['weighted_recall']:.2f}%")
    c4.metric("Weighted F1-Score", f"{EXPERIMENT_METRICS['mobilenetv2']['weighted_f1']:.2f}%")

    st.markdown("---")
    st.subheader("2. Architecture Comparison: Basic CNN vs. MobileNetV2")

    comp_col1, comp_col2 = st.columns([1.15, 1.35], gap="large")
    with comp_col1:
        comp_df = pd.DataFrame(
            [
                {
                    "Model": "Model 1: Basic CNN",
                    "Saved Artifact": "fish_disease_cnn.keras",
                    "Input Size": "160 × 160",
                    "Test Accuracy (%)": "45.77%",
                    "Role": "Baseline Model",
                },
                {
                    "Model": "Model 2: MobileNetV2 (Transfer Learning)",
                    "Saved Artifact": "fish_disease_mobilenetv2.keras",
                    "Input Size": "160 × 160",
                    "Test Accuracy (%)": "86.80%",
                    "Role": "Primary Production Model",
                },
            ]
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.markdown(
            """
            **Key Experimental Findings:**
            - Transfer learning with ImageNet-pretrained **MobileNetV2** improved test accuracy from **45.77%** (Basic CNN) to **86.80%** (+41.03 percentage points) in **8 training epochs**.
            - Global Average Pooling and `Dropout(0.3)` regularized the classifier head effectively across the 1,398 training and 349 validation images.
            """
        )

    with comp_col2:
        comp_img_path = os.path.join("results", "model_comparison.png")
        if os.path.exists(comp_img_path):
            st.image(comp_img_path, caption="Model Test Accuracy Comparison", use_container_width=True)

    st.markdown("---")
    st.subheader("3. Confusion Matrix & Classification Report")

    eval_col1, eval_col2 = st.columns(2, gap="large")
    with eval_col1:
        st.markdown("#### Confusion Matrix")
        cm_path = os.path.join("results", "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix on Test Set (697 Images)", use_container_width=True)

    with eval_col2:
        st.markdown("#### Classification Report")
        cr_txt_path = os.path.join("results", "classification_report.txt")
        if os.path.exists(cr_txt_path):
            with open(cr_txt_path, "r", encoding="utf-8") as f:
                st.code(f.read(), language="text")

    st.markdown("---")
    st.subheader("4. Training Curves, Class Distribution & Grad-CAM Samples")
    vis_col1, vis_col2 = st.columns(2, gap="large")
    with vis_col1:
        hist_path = os.path.join("results", "training_history.png")
        if os.path.exists(hist_path):
            st.image(hist_path, caption="Training and Validation Accuracy / Loss Curves (8 Epochs)", use_container_width=True)
    with vis_col2:
        dist_path = os.path.join("results", "class_distribution.png")
        if os.path.exists(dist_path):
            st.image(dist_path, caption="Dataset Split Distribution (1,747 Train / 697 Test)", use_container_width=True)

    gc_path = os.path.join("results", "gradcam_samples.png")
    if os.path.exists(gc_path):
        st.image(gc_path, caption="Grad-CAM Explainability Pipeline", use_container_width=True)


elif page == "ℹ️ About Project":
    st.markdown(
        """
        <div class="main-header">
            <h1>About the Project & Technical Methodology</h1>
            <p>A Framework for Automated Fish Disease Detection in Aquaculture Using Deep Learning</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("1. Project Background & Objective")
    st.write(
        "Disease outbreaks in freshwater aquaculture cause substantial stock mortality and economic loss across "
        "South Asia. Manual visual inspection of fish requires domain expertise and often delays early intervention. "
        "This project provides an automated deep learning framework capable of classifying seven distinct freshwater "
        "fish health states from uploaded photographs while providing visual transparency through Gradient-weighted "
        "Class Activation Mapping (Grad-CAM)."
    )

    st.markdown("---")
    st.subheader("2. Dataset Specification")
    d_col1, d_col2 = st.columns(2, gap="large")
    with d_col1:
        st.markdown(
            """
            - **Dataset Source:** *Freshwater Fish Disease Aquaculture in South Asia* (Kaggle)
            - **Archive Name:** `archive (2).zip`
            - **Total Training Set Images:** `1,747`
              - **Training Split:** `1,398` images
              - **Validation Split:** `349` images
            - **Testing Set Images:** `697` images
            """
        )
    with d_col2:
        st.markdown("**Seven Target Classes:**")
        for idx, cls_name in enumerate(get_class_names(), start=1):
            st.markdown(f"{idx}. `{cls_name}`")

    st.markdown("---")
    st.subheader("3. Deep Learning Architecture & Preprocessing Pipeline")
    st.markdown(
        """
        1. **Input Preparation (`160 × 160 × 3`):**
           Uploaded images (JPG, JPEG, PNG) are validated, converted to RGB, and resized to `160 × 160` pixels.
        2. **Single-Pass Rescaling Guardrail:**
           `fish_disease_mobilenetv2.keras` embeds its own internal `Rescaling` layer inside the model graph.
           To prevent double-normalization distortion, the web inference pipeline passes raw `float32` pixel intensities
           in the `[0.0, 255.0]` range directly to the model.
        3. **Feature Extraction & Classification Head:**
           - **Backbone:** Pretrained ImageNet `MobileNetV2`
           - **Pooling:** `GlobalAveragePooling2D`
           - **Regularization:** `Dropout(0.3)`
           - **Output Layer:** `Dense(7, activation='softmax')` trained for `8` epochs.
        4. **Explainable AI (Grad-CAM):**
           Using `tf.GradientTape`, gradients of the winning class score are computed with respect to the final 4D
           convolutional feature maps. Channel-wise global average pooling weights those feature maps to produce a
           spatial heatmap highlighting the visual regions responsible for the prediction.
        """
    )
