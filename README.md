# 🎓 GPA Predictor — ML + SHAP + Streamlit

> **Predict your next semester GPA** with machine learning and understand *why* with SHAP explainability.

---

## ✨ Features

- 🔮 **GPA Prediction** using Linear Regression & Random Forest
- 🖥️ **Interactive UI** powered by Streamlit
- 📊 **Feature Importance** visualization
- 🔄 **What-If Analysis** — adjust inputs and see how GPA changes
- 🧠 **Explainable AI** via SHAP values

---

## 🧠 Models

| Model | Role |
|---|---|
| Linear Regression | Baseline |
| Random Forest Regressor | Final prediction (recommended) |

---

## 📥 Input Features

| Feature | Description |
|---|---|
| `prev_gpa` | Previous semester GPA |
| `credits` | Total enrolled credits |
| `is_math` | Taking a math course? (`0` = No, `1` = Yes) |
| `workload_credits` | Study hours per credit |

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) reveals exactly which factors drive your predicted GPA:

- 🔴 **Red** → pushes GPA **higher**
- 🔵 **Blue** → pushes GPA **lower**

---

## 📁 Dataset Format

Your `grades.csv` should follow this structure:

```csv
gpa,credits,is_math,workload_hours
3.2,15,1,20
3.5,18,0,25
```

> **Auto-generated features:**
> - `prev_gpa` — derived from previous row GPA
> - `workload_credits` — computed as `workload_hours / credits`

---

## ⚙️ Installation

```bash
pip install pandas numpy streamlit matplotlib scikit-learn shap
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 🖥️ App Sections

1. 📋 **Data Preview** — inspect your loaded dataset
2. 🎛️ **Input Controls** — enter your academic details
3. 📈 **Model Predictions** — see your predicted GPA
4. 📊 **Feature Importance** — understand what matters most
5. 🔍 **SHAP Explanation** — per-prediction explainability

---

## 🗺️ Roadmap

- [ ] Add more features (attendance, sleep hours, course difficulty)
- [ ] Allow CSV upload directly from the UI
- [ ] Add evaluation metrics (MAE, RMSE)

---

## 👩‍💻 Author

**Malaika Yousaf**

---

<p align="center">Made with ❤️ and a lot of studying</p>
