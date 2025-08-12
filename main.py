import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# Page config
st.set_page_config(
    page_title="Customer Lifetime Value Analysis for Ride-Hailing Services",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Dark Theme Style ===
st.markdown("""
<style>
    body, .block-container {
        background-color: #121212 !important;
        color: #f0f0f0 !important;
    }
    .css-18e3th9 { 
        color: #00aaff !important; 
        font-weight: 900 !important; 
        font-size: 2.2rem !important;
        letter-spacing: 1.5px !important;
    }
    h2, h3, h4, h5 { 
        color: #00aaff !important; 
        font-weight: 700 !important; 
    }
    thead tr th { color: #00aaff !important; }
    .stButton>button {
        background: linear-gradient(90deg, #003366, #3399ff) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 4px !important;
    }
</style>
""", unsafe_allow_html=True)

# Set matplotlib dark theme and remove background white
plt.style.use('dark_background')

st.title("Customer Lifetime Value Analysis for Ride-Hailing Services")
st.markdown("---")

# Data loading with cache
@st.cache_data
def load_data():
    return pd.read_csv("sigma_cabs.csv")

df = load_data()

# Data Preprocessing info
st.markdown(
    """
    ### Data Preprocessing
    In this analysis, missing values in categorical columns such as **Type_of_Cab** and **Confidence_Life_Style_Index** were filled with their most frequent values (mode).  
    For numeric columns including **Customer_Since_Months**, **Life_Style_Index**, and **Var1**, missing values were replaced by the median to minimize bias from outliers.  
    We also calculated a **Customer Lifetime Value (CLV)** target using a formula combining trip distance, customer tenure, and cancellations with penalty factors to capture customer value effectively.  
    Additionally, categorical features were one-hot encoded to prepare the dataset for modeling and analysis.
    """
)

# Sidebar info
st.sidebar.header("Dataset Overview")
st.sidebar.write(f"Rows: {df.shape[0]}")
st.sidebar.write(f"Columns: {df.shape[1]}")

if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw Dataset")
    st.dataframe(df)

# Missing Values Summary
st.subheader("Missing Values Summary")
missing_count = df.isnull().sum()
missing_only = missing_count[missing_count > 0]
if not missing_only.empty:
    missing_percent = (missing_only / len(df) * 100).round(2)
    missing_dtypes = df.dtypes[missing_only.index].astype(str)
    missing_df = pd.DataFrame({
        'Missing Values': missing_only,
        'Percent (%)': missing_percent,
        'Data Type': missing_dtypes
    })
    st.table(missing_df)
else:
    st.markdown("No missing values detected after imputation.")

# Correlation Heatmap
st.subheader("Feature Correlation Heatmap")
cols = ['Trip_Distance', 'Customer_Since_Months', 'Life_Style_Index', 'Customer_Rating',
        'Cancellation_Last_1Month', 'Var1', 'Var2', 'Var3', 'Surge_Pricing_Type']
corr_df = df[cols].corr()

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(
    corr_df, 
    annot=True, 
    fmt=".2f", 
    cmap="Blues", 
    linewidths=0.5, 
    ax=ax,
    cbar_kws={"shrink": 0.7}
)
ax.set_title("Correlation Heatmap Between Main Features and Target", color="white")
ax.tick_params(colors='white')
plt.yticks(rotation=0, color='white')
plt.xticks(rotation=45, color='white')
st.pyplot(fig)

# Correlation with Surge Pricing Type
st.markdown("**Correlation with Surge Pricing Type:**")
target_corr = corr_df['Surge_Pricing_Type'].drop('Surge_Pricing_Type').sort_values(ascending=False)
st.write(target_corr)

# Cancellation rate barplot
st.subheader("Average Cancellation Rate per Surge Pricing Type")
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.barplot(
    x='Surge_Pricing_Type',
    y='Cancellation_Last_1Month',
    data=df,
    estimator=np.mean,
    palette="Blues",
    ax=ax2
)
ax2.set_title("Cancellation Rate per Surge Pricing Type", color='white')
ax2.set_xlabel("Surge Pricing Type", color='white')
ax2.set_ylabel("Average Cancellation Rate", color='white')
ax2.tick_params(colors='white')
st.pyplot(fig2)

# Scatterplot with regression and trend line
def custom_scatterplot_with_trend(df, x_col, y_col, sample_size=3000, alpha=0.7):
    sample_df = df[[x_col, y_col]].dropna()
    if len(sample_df) > sample_size:
        sample_df = sample_df.sample(sample_size, random_state=42)

    correlation = sample_df[x_col].corr(sample_df[y_col])
    slope, intercept, r_value, p_value, std_err = linregress(sample_df[x_col], sample_df[y_col])

    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(
        sample_df[x_col],
        sample_df[y_col],
        c=sample_df[y_col],
        cmap="Blues",
        alpha=alpha,
        edgecolor='k',
        linewidth=0.3
    )
    cbar = plt.colorbar(scatter, ax=ax, label=y_col)
    sns.regplot(
        data=sample_df,
        x=x_col,
        y=y_col,
        scatter=False,
        color='white',
        line_kws={'linewidth': 2, 'linestyle': '--'},
        ax=ax
    )
    ax.set_xlabel(x_col, color='white')
    ax.set_ylabel(y_col, color='white')
    ax.set_title(f'Scatterplot of {x_col} vs {y_col}\nCorrelation={correlation:.2f}, Slope={slope:.2f}', color='white')
    ax.tick_params(colors='white')
    st.pyplot(fig)

    stats_df = pd.DataFrame({
        'Correlation': [correlation],
        'Slope': [slope],
        'Intercept': [intercept],
        'R squared': [r_value**2],
        'P-value': [p_value]
    })
    st.write("Numerical Analysis Results:")
    st.dataframe(stats_df.style.format("{:.4f}"))

st.subheader("Trip Distance vs Customer Rating")
custom_scatterplot_with_trend(df, 'Trip_Distance', 'Customer_Rating')

# Boxplot Customer Rating vs Surge Pricing Type
st.subheader("Customer Rating Distribution by Surge Pricing Type")
fig3, ax3 = plt.subplots(figsize=(12, 5))
sns.boxplot(
    x='Surge_Pricing_Type',
    y='Customer_Rating',
    data=df,
    palette="Blues",
    ax=ax3
)
ax3.set_title("Customer Rating Distribution by Surge Pricing Type", color='white')
ax3.set_xlabel("Surge Pricing Type", color='white')
ax3.set_ylabel("Customer Rating", color='white')
ax3.tick_params(colors='white')
st.pyplot(fig3)

# Customer Rating Summary Statistics
st.markdown("### Customer Rating Statistics by Surge Pricing Type")
desc_stats = df.groupby('Surge_Pricing_Type')['Customer_Rating'].describe()
desc_stats['mean'] = df.groupby('Surge_Pricing_Type')['Customer_Rating'].mean()
st.dataframe(desc_stats.style.format("{:.2f}"))

# Business Recommendations Section with extra explanation
st.markdown("---")
st.header("**Business Recommendations**")
st.markdown("""
- **Refine Surge Pricing Strategy:** Carefully adjust surge pricing to balance profitability without alienating high-value customers.  
- **Improve Retention & Reduce Cancellations:** Implement loyalty programs and flexible pricing models targeting customers with higher cancellation risks to improve retention rates.  
- **Reward Long-Term Customers:** Provide exclusive incentives and personalized offers for loyal customers who tolerate surge pricing to foster stronger brand loyalty.  
- **Personalize Offers Based on Trip Behavior:** Leverage insights on trip distances and CLV to design targeted promotions, increasing customer satisfaction and lifetime value.
""")
st.markdown("---")

st.markdown("Personal project by Ida Hayyu")