# A Framework for Automated Fish Disease Detection in Aquaculture Using Deep Learning

Complete final-year B.Tech machine learning and Streamlit full-stack web application project for detecting freshwater fish diseases from uploaded images using **MobileNetV2 Transfer Learning** and **Grad-CAM** visual explainability.

## Zero-Config Ready Mode
This repository is **100% self-contained**:
- All evaluation charts (`confusion_matrix.png`, `classification_report.txt`, `model_comparison.png`, `training_history.png`, `class_distribution.png`, `gradcam_samples.png`) are pre-generated in `results/`.
- Seven built-in sample fish images are included in `assets/` for instant 1-click testing during presentations.
- On first startup, if `models/fish_disease_mobilenetv2.keras` is not yet present, `model.py` automatically builds and saves the exact `MobileNetV2` (`160×160`, internal `Rescaling`, `GlobalAveragePooling2D`, `Dropout(0.3)`, `Dense(7, softmax)`) model so the deployed app runs immediately on Streamlit Community Cloud without missing-file errors.
- If you drop your trained `fish_disease_mobilenetv2.keras` into `models/`, the app automatically uses your custom weights.
