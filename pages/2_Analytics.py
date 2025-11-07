import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Analytics", page_icon="📊")

st.title("📊 Advanced Analytics")

# Generate more complex sample data
@st.cache_data
def generate_analytics_data():
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    data = []
    for date in dates:
        # Simulate seasonal trends
        season_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
        
        for product in ['Product A', 'Product B', 'Product C']:
            for region in ['North', 'South', 'East', 'West']:
                sales = np.random.normal(1000 * season_factor, 200)
                customers = np.random.poisson(30)
                
                data.append({
                    'Date': date,
                    'Product': product,
                    'Region': region,
                    'Sales': max(0, sales),
                    'Customers': customers,
                    'Revenue': sales * np.random.uniform(10, 50)
                })
    
    return pd.DataFrame(data)

# Load data
with st.spinner('Loading analytics data...'):
    df = generate_analytics_data()

st.success(f"Loaded {len(df):,} records")

# Sidebar filters
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df['Date'].min(), df['Date'].max()),
    min_value=df['Date'].min(),
    max_value=df['Date'].max()
)

selected_products = st.sidebar.multiselect(
    "Select Products",
    options=df['Product'].unique(),
    default=df['Product'].unique()
)

selected_regions = st.sidebar.multiselect(
    "Select Regions",
    options=df['Region'].unique(),
    default=df['Region'].unique()
)

# Filter data
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[
        (df['Date'] >= pd.to_datetime(start_date)) &
        (df['Date'] <= pd.to_datetime(end_date)) &
        (df['Product'].isin(selected_products)) &
        (df['Region'].isin(selected_regions))
    ]
else:
    filtered_df = df[
        (df['Product'].isin(selected_products)) &
        (df['Region'].isin(selected_regions))
    ]

# KPI Cards
st.header("Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_revenue = filtered_df['Revenue'].sum()
    st.metric("Total Revenue", f"${total_revenue:,.0f}")

with col2:
    total_sales = filtered_df['Sales'].sum()
    st.metric("Total Sales", f"{total_sales:,.0f}")

with col3:
    total_customers = filtered_df['Customers'].sum()
    st.metric("Total Customers", f"{total_customers:,}")

with col4:
    avg_revenue_per_customer = total_revenue / total_customers if total_customers > 0 else 0
    st.metric("Avg Revenue/Customer", f"${avg_revenue_per_customer:.2f}")

# Charts
st.header("Analytics Dashboard")

# Revenue trend over time
st.subheader("Revenue Trend Over Time")
daily_revenue = filtered_df.groupby('Date')['Revenue'].sum().reset_index()
fig_trend = px.line(daily_revenue, x='Date', y='Revenue', 
                    title='Daily Revenue Trend')
st.plotly_chart(fig_trend, use_container_width=True)

# Revenue by product and region
col1, col2 = st.columns(2)

with col1:
    st.subheader("Revenue by Product")
    product_revenue = filtered_df.groupby('Product')['Revenue'].sum().reset_index()
    fig_product = px.bar(product_revenue, x='Product', y='Revenue',
                         title='Revenue by Product')
    st.plotly_chart(fig_product, use_container_width=True)

with col2:
    st.subheader("Revenue by Region")
    region_revenue = filtered_df.groupby('Region')['Revenue'].sum().reset_index()
    fig_region = px.pie(region_revenue, values='Revenue', names='Region',
                        title='Revenue Distribution by Region')
    st.plotly_chart(fig_region, use_container_width=True)

# Heatmap
st.subheader("Sales Heatmap: Product vs Region")
heatmap_data = filtered_df.pivot_table(
    values='Sales', 
    index='Product', 
    columns='Region', 
    aggfunc='sum'
)
fig_heatmap = px.imshow(heatmap_data, 
                        title='Sales by Product and Region',
                        color_continuous_scale='Blues')
st.plotly_chart(fig_heatmap, use_container_width=True)

# Data table
st.header("Detailed Data")
if st.checkbox("Show raw data"):
    st.dataframe(filtered_df.head(1000), use_container_width=True)
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=f'analytics_data_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv'
    )
