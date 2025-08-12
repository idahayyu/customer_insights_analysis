# CUSTOMER INSIGHTS: LIFETIME VALUE PREDICTION FOR RIDE-HAILING

## Project Overview
This project analyzes Sigma Cabs customer and trip data to uncover the key drivers of **Customer Lifetime Value (CLV)**. The ultimate goal is to provide actionable, data-driven business recommendations to increase customer retention and profitability.

---

## Dataset

- Includes features like **trip distance**, **customer tenure**, **lifestyle index**, **surge pricing**, and **cancellations**.
- Missing values are imputed using:
  - **Median** for numerical columns
  - **Mode** for categorical columns

---

## Key Features & Target

- **Input Features**:  
  Trip distance, customer tenure, lifestyle index, surge pricing, cancellation rate, etc.

- **Target Variable**:  
  **Customer Lifetime Value (CLV)** — a metric that reflects both customer loyalty and revenue potential.

---

## Data Processing Steps

- Handling missing values (median/mode imputation)
- Feature engineering to calculate CLV
- One-hot encoding of categorical features
- Feature scaling using `StandardScaler`

---

## Exploratory Data Analysis (EDA)

Key insights discovered:
- 🚫 Cancellation rates increase with higher surge pricing.
- 😠 Surge pricing negatively impacts customer satisfaction.
- 🚗 Longer trip distances slightly reduce customer ratings.

---

## Business Recommendations

- Adjust surge pricing to prevent churn of high-value customers.
- Implement loyalty rewards and targeted offers for long-tenured users.
- Design flexible cancellation policies for frequent cancellers.
- Use trip behavior data to personalize promotions and incentives.

---

## Live Demo

Try the interactive dashboard online (no installation needed):

🔗 [Customer_Lifetime_Value_Analysis Dashboard]([https://customer-lifetime-value-analysis.streamlit.app)

---

## Tech Stack

- Python (Pandas, Scikit-learn, Seaborn, Streamlit)
- Jupyter Notebook
- Streamlit for dashboard
- Git & GitHub for version control

---

## Acknowledgment

Developed for educational and analytical purposes.  
Dataset made available under a public domain license (CC0).

---
