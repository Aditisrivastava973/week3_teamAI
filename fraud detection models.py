import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("🛡️ Fraud Detection on Cybersecurity Threats")

# Upload CSV
uploaded_file = st.file_uploader("Global_Cybersecurity_Threats_2015-2024.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Target creation
    df['is_fraud'] = ((df['Financial Loss (in Million $)'] > 60) & 
                      (df['Attack Source'] == 'Insider')).astype(int)

    st.subheader("📊 Raw Dataset Preview")
    st.dataframe(df.head())

    # Balance data
    df_majority = df[df.is_fraud == 0]
    df_minority = df[df.is_fraud == 1]
    df_majority_downsampled = resample(df_majority,
                                       replace=False,
                                       n_samples=len(df_minority),
                                       random_state=42)
    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    # Encode features
    X = df_balanced.drop(columns="is_fraud")
    y = df_balanced["is_fraud"]

    label_encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    # Evaluation
    report = classification_report(y_test, y_pred, output_dict=True)
    st.subheader("📈 Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Feature importance
    st.subheader("🔥 Feature Importance")
    importances = rf_model.feature_importances_
    feat_series = pd.Series(importances, index=X.columns).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=feat_series.values, y=feat_series.index, palette="viridis", ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    st.success("✅ Model trained on balanced data!")