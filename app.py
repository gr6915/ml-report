import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os

# Title
st.title("📊 Machine Learning Model Runner with Report Generator")

# Upload file
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    # Choose target column
    target = st.selectbox("Select target column", df.columns)

    # Choose model type
    model_type = st.selectbox("Select problem type", ["Regression", "Classification"])

    # Choose model
    if model_type == "Regression":
        model_choice = st.selectbox("Select Model", ["Linear Regression", "Random Forest Regressor"])
    else:
        model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest Classifier"])

    # Train/Test Split
    test_size = st.slider("Test size (fraction)", 0.1, 0.5, 0.2)
    random_state = st.number_input("Random State", value=42)

    if st.button("Run Model"):
        X = df.drop(columns=[target])
        y = df[target]

        # Handle categorical variables
        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Select model
        if model_type == "Regression":
            if model_choice == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor()
        else:
            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            else:
                model = RandomForestClassifier()

        # Fit model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate
        if model_type == "Regression":
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write("### Evaluation Metrics")
            st.write(f"Mean Squared Error: {mse:.4f}")
            st.write(f"R² Score: {r2:.4f}")
        else:
            acc = accuracy_score(y_test, y_pred)
            st.write("### Evaluation Metrics")
            st.write(f"Accuracy: {acc:.4f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

        # Plot actual vs predicted for regression
        if model_type == "Regression":
            plt.figure(figsize=(6, 4))
            sns.scatterplot(x=y_test, y=y_pred)
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("Actual vs Predicted")
            st.pyplot(plt.gcf())

        # --- Generate PDF Report ---
        if st.button("Generate PDF Report"):
            report_file = "report.pdf"
            doc = SimpleDocTemplate(report_file)
            styles = getSampleStyleSheet()
            flowables = []

            flowables.append(Paragraph("Machine Learning Model Report", styles['Title']))
            flowables.append(Spacer(1, 12))
            flowables.append(Paragraph(f"Model Type: {model_type}", styles['Normal']))
            flowables.append(Paragraph(f"Model: {model_choice}", styles['Normal']))

            if model_type == "Regression":
                flowables.append(Paragraph(f"Mean Squared Error: {mse:.4f}", styles['Normal']))
                flowables.append(Paragraph(f"R² Score: {r2:.4f}", styles['Normal']))
            else:
                flowables.append(Paragraph(f"Accuracy: {acc:.4f}", styles['Normal']))
                flowables.append(Paragraph("Classification Report:", styles['Normal']))
                flowables.append(Paragraph(classification_report(y_test, y_pred), styles['Code']))

            doc.build(flowables)

            with open(report_file, "rb") as f:
                st.download_button(
                    label="Download PDF Report",
                    data=f,
                    file_name="ml_report.pdf",
                    mime="application/pdf",
                )
