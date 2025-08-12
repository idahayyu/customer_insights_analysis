# CUSTOMER INSIGHTS: LIFETIME VALUE PREDICTION FOR RIDE-HAILING

## Project Overview
This project analyzes Sigma Cabs customer and trip data to understand key factors influencing Customer Lifetime Value (CLV). The goal is to provide data-driven business recommendations to improve customer retention and profitability.

## Dataset
- Contains customer and trip details such as trip distance, customer tenure, lifestyle index, surge pricing, and cancellations.
- Missing values are handled with median imputation for numeric columns and mode imputation for categorical columns.

## Key Features & Target
- Features include trip distance, customer tenure, lifestyle index, cancellation rates, and more.
- Target variable: Customer Lifetime Value (CLV), a weighted metric reflecting customer loyalty and value.

## Data Processing
- Missing value imputation to maintain data quality.
- Feature engineering to compute CLV.
- One-hot encoding for categorical features.
- Numeric feature scaling using StandardScaler.

## Exploratory Data Analysis (EDA)
- Analyzes CLV relation with demographics and customer behavior.
- Key insights:
  - Cancellation rates increase with surge pricing.
  - Higher surge pricing negatively impacts customer satisfaction.
  - Trip distance slightly lowers customer ratings.

## Business Recommendations
- Refine surge pricing to reduce cancellations among high-value customers.
- Develop loyalty programs and flexible pricing for frequent cancellers.
- Reward long-term customers with incentives.
- Personalize offers based on trip behaviors.

## Live Demo
Experience the interactive dashboard online without any downloads:  
[Open Streamlit Dashboard](https://share.streamlit.io/your_github_username/your_repo_name/main/app.py)  
*(Replace the URL with your actual deployed app link)*

---

Made with care by Ida Hayyu