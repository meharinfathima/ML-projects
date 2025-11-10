# app.py - Iris ML Web App with Model Comparison
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import joblib

# Streamlit page setup
st.set_page_config(page_title="Iris ML Mini Project", layout="centered")
st.title("🌸 Iris Classifier — ML Mini Project")
st.write("Compare Logistic Regression and Random Forest on the classic Iris dataset.")

# 1️⃣ Load dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Sidebar settings
st.sidebar.header("Settings")
feat_options = {f"{i}: {name}": i for i, name in enumerate(feature_names)}
feat_choice = st.sidebar.multiselect("Choose two features (pick 2)", list(feat_options.keys()),
                                     default=[f"2: {feature_names[2]}", f"3: {feature_names[3]}"])
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.3, step=0.05)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)

if len(feat_choice) != 2:
    st.warning("Pick exactly two features to visualize decision regions.")
    st.stop()

feat_idx = [feat_options[feat_choice[0]], feat_options[feat_choice[1]]]
X2 = X[:, feat_idx]
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=test_size, stratify=y, random_state=int(random_state))

# 2️⃣ Train models
pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=42))
pipe_rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))

pipe_lr.fit(X_train, y_train)
pipe_rf.fit(X_train, y_train)

# Predictions
y_pred_lr = pipe_lr.predict(X_test)
y_pred_rf = pipe_rf.predict(X_test)

# 3️⃣ Evaluation function
def summarize_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

metrics_lr = summarize_metrics(y_test, y_pred_lr)
metrics_rf = summarize_metrics(y_test, y_pred_rf)

# Cross-validation
cv = 5
cv_lr = cross_val_score(pipe_lr, X2, y, cv=cv, scoring='accuracy').mean()
cv_rf = cross_val_score(pipe_rf, X2, y, cv=cv, scoring='accuracy').mean()

# 4️⃣ Display results
st.subheader("📊 Model Comparison")

comp_df = pd.DataFrame([
    {"Model": "Logistic Regression", **metrics_lr, "CV Accuracy": cv_lr},
    {"Model": "Random Forest", **metrics_rf, "CV Accuracy": cv_rf}
]).set_index("Model")

st.dataframe(comp_df.round(3))

# Bar chart
fig, ax = plt.subplots(figsize=(6, 3))
bar_width = 0.35
index = np.arange(len(comp_df))
ax.bar(index, comp_df["accuracy"], bar_width, label="Test Accuracy")
ax.bar(index + bar_width, comp_df["CV Accuracy"], bar_width, label="CV Accuracy")
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(comp_df.index)
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.legend()
st.pyplot(fig)

# 5️⃣ Confusion matrix (for RandomForest)
st.subheader("Confusion Matrix (RandomForest)")
cm = confusion_matrix(y_test, y_pred_rf)
st.write(pd.DataFrame(cm, index=target_names, columns=target_names))

# 6️⃣ Interactive prediction
st.subheader("🌼 Try your own input")
with st.form("predict_form"):
    vals = []
    for i, fi in enumerate(feat_idx):
        v = st.number_input(f"{feature_names[fi]}", value=float(np.mean(X2[:, i])), format="%.2f")
        vals.append(v)
    submitted = st.form_submit_button("Predict")
if submitted:
    arr = np.array(vals).reshape(1, -1)
    pred = pipe_rf.predict(arr)[0]
    probs = pipe_rf.predict_proba(arr)[0]
    st.write(f"**Predicted class:** {target_names[pred]}")
    st.write("Class probabilities:")
    st.json({target_names[i]: float(f"{probs[i]:.3f}") for i in range(len(target_names))})

# 7️⃣ Save best model
best_model = pipe_rf if metrics_rf["f1"] >= metrics_lr["f1"] else pipe_lr
joblib.dump(best_model, "best_iris_model.joblib")
st.info(f"Saved best model: {'Random Forest' if best_model == pipe_rf else 'Logistic Regression'} ✅")
