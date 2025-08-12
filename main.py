import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# Page config
st.set_page_config(
    page_title="CUSTOMER INSIGHTS : LIFETIME VALUE PREDICTION FOR RIDE-HAILING",
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
    .css-18e3th9 { color: #00aaff !important; font-weight: 700 !important; }
    h2, h3, h4, h5 { color: #00aaff !important; font-weight: 600 !important; }
    thead tr th { color: #00aaff !important; }
    .stButton>button {
        background: linear-gradient(90deg, #003366, #3399ff) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 4px !important;
    }
</style>
""", unsafe_allow_html=True)

# Set matplotlib dark theme
plt.style.use("dark_background")

st.title("CUSTOMER INSIGHTS : LIFETIME VALUE PREDICTION FOR RIDE-HAILING")
st.markdown("---")

@st.cache_data
def load_data():
    return pd.read_csv("sigma_cabs.csv")

df = load_data()

# Imputation
df['Type_of_Cab'].fillna(df['Type_of_Cab'].mode()[0], inplace=True)
df['Confidence_Life_Style_Index'].fillna(df['Confidence_Life_Style_Index'].mode()[0], inplace=True)
for col in ['Customer_Since_Months', 'Life_Style_Index', 'Var1']:
    df[col].fillna(df[col].median(), inplace=True)

# Target CLV
penalty_factor = 2
df['CLV'] = df['Trip_Distance'] * 1.5 + df['Customer_Since_Months'] * 2 - df['Cancellation_Last_1Month'] * penalty_factor

# One-hot encode categorical
cat_cols = ['Type_of_Cab', 'Destination_Type', 'Gender', 'Confidence_Life_Style_Index']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
if 'Trip_ID' in df_encoded.columns:
    df_encoded.drop('Trip_ID', axis=1, inplace=True)

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

# Heatmap
st.subheader("Feature Correlation Heatmap")
cols = ['Trip_Distance', 'Customer_Since_Months', 'Life_Style_Index', 'Customer_Rating',
        'Cancellation_Last_1Month', 'Var1', 'Var2', 'Var3', 'Surge_Pricing_Type']
corr_df = df[cols].corr()

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5, ax=ax)
ax.set_title("Correlation Heatmap Between Main Features and Target", color="white")
ax.tick_params(colors='white')
plt.yticks(rotation=0, color='white')
plt.xticks(rotation=45, color='white')
st.pyplot(fig)

# Correlation with target
st.markdown("**Correlation with Surge Pricing Type:**")
target_corr = corr_df['Surge_Pricing_Type'].drop('Surge_Pricing_Type').sort_values(ascending=False)
st.write(target_corr)

# Barplot: Cancellation Rate
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

# Scatterplot with regression
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
    plt.colorbar(scatter, label=y_col)
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

# Boxplot: Customer Rating vs Surge Type
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

# Rating Summary
st.markdown("### Customer Rating Statistics by Surge Pricing Type")
desc_stats = df.groupby('Surge_Pricing_Type')['Customer_Rating'].describe()
desc_stats['mean'] = df.groupby('Surge_Pricing_Type')['Customer_Rating'].mean()
st.dataframe(desc_stats.style.format("{:.2f}"))

# Final Note
st.markdown("---")
st.header("Business Recommendations")
st.markdown("""
- **Refine Surge Pricing Strategy:** Adjust surge pricing to minimize negative impacts on satisfaction and cancellations of high-value customers.  
- **Improve Retention & Reduce Cancellations:** Develop loyalty programs and flexible pricing for customers prone to cancellations.  
- **Reward Long-Term Customers:** Offer special incentives to retain loyal customers who tolerate surge pricing.  
- **Personalize Offers Based on Trip Behavior:** Tailor promotions and packages for customers with long-distance trips and high CLV.
""")
st.markdown("---")
st.markdown("Personal project by Ida Hayyu")