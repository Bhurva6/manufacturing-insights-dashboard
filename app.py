import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Manufacturing Insights Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Main title
    st.title("🏭 Manufacturing Insights Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Home", "Data Upload & Overview", "SKU Analysis", "Regional Analysis", "Manufacturing Recommendations"]
    )
    
    if page == "Home":
        show_home()
    elif page == "Data Upload & Overview":
        show_data_upload()
    elif page == "SKU Analysis":
        show_sku_analysis()
    elif page == "Regional Analysis":
        show_regional_analysis()
    elif page == "Manufacturing Recommendations":
        show_manufacturing_recommendations()

def show_home():
    st.header("� Manufacturing Insights Dashboard")
    
    # Welcome message
    st.markdown("""
    ### Welcome to your Manufacturing Analytics Platform!
    
    This dashboard helps manufacturing companies analyze their SKU performance and make data-driven decisions:
    
    🔹 **Data Upload & Overview**: Upload your Excel files and get instant insights  
    🔹 **SKU Analysis**: Analyze individual SKUs, performance metrics, and inventory levels  
    🔹 **Regional Analysis**: Understand geographic performance and regional trends  
    🔹 **Manufacturing Recommendations**: Get AI-powered recommendations for optimal production quantities  
    
    📊 **Key Features:**
    - Upload and analyze Excel files with manufacturing data
    - Track SKU performance across different regions
    - Calculate minimum manufacturing quantities
    - Visualize trends and patterns in your data
    - Generate actionable insights for production planning
    
    ### Getting Started:
    1. Navigate to **Data Upload & Overview** to upload your Excel file
    2. Explore your SKUs in the **SKU Analysis** section
    3. Check regional performance in **Regional Analysis**
    4. Get recommendations in **Manufacturing Recommendations**
    """)
    
    # Key metrics placeholder (will be populated when data is uploaded)
    if 'manufacturing_data' in st.session_state:
        df = st.session_state.manufacturing_data
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total SKUs",
                value=f"{len(df['SKU'].unique()) if 'SKU' in df.columns else 0:,}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Total Sales",
                value=f"${df['Sales'].sum():,.0f}" if 'Sales' in df.columns else "$0",
                delta=None
            )
        
        with col3:
            st.metric(
                label="Active Regions",
                value=f"{len(df['Region'].unique()) if 'Region' in df.columns else 0}",
                delta=None
            )
        
        with col4:
            st.metric(
                label="Avg Units/SKU",
                value=f"{df['Units_Sold'].mean():.0f}" if 'Units_Sold' in df.columns else "0",
                delta=None
            )
    else:
        st.info("📤 Upload your manufacturing data to see key metrics here!")

def show_data_upload():
    st.header("� Data Upload & Overview")
    
    st.markdown("""
    Upload your manufacturing Excel file to get started with the analysis. 
    
    **Expected columns in your Excel file:**
    - SKU (Stock Keeping Unit identifier)
    - Product_Name
    - Region
    - Sales (revenue)
    - Units_Sold
    - Date
    - Cost_Per_Unit
    - Current_Inventory
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an Excel file", 
        type=['xlsx', 'xls'],
        help="Upload your manufacturing data in Excel format"
    )
    
    if uploaded_file is not None:
        try:
            # Read the Excel file
            df = pd.read_excel(uploaded_file)
            st.success("✅ File uploaded successfully!")
            
            # Store in session state
            st.session_state.manufacturing_data = df
            
            # Show basic info
            st.subheader("📋 Dataset Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Records", len(df))
                st.metric("Total Columns", len(df.columns))
            
            with col2:
                st.metric("Null Values", df.isnull().sum().sum())
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Display column information
            st.subheader("📊 Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(col_info, width='stretch')
            
            # Preview the data
            st.subheader("🔍 Data Preview")
            st.dataframe(df.head(10), width='stretch')
            
            # Basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.subheader("📈 Statistical Summary")
                st.dataframe(df[numeric_cols].describe(), width='stretch')
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please make sure your file is a valid Excel file (.xlsx or .xls)")
    
    else:
        # Show sample data format
        st.subheader("📝 Sample Data Format")
        st.markdown("Here's an example of how your data should be structured:")
        
        sample_data = pd.DataFrame({
            'SKU': ['SKU001', 'SKU002', 'SKU003', 'SKU004', 'SKU005'],
            'Product_Name': ['Widget A', 'Widget B', 'Gadget X', 'Tool Y', 'Component Z'],
            'Region': ['North', 'South', 'East', 'West', 'North'],
            'Sales': [25000, 18000, 32000, 15000, 28000],
            'Units_Sold': [500, 360, 640, 300, 560],
            'Date': ['2024-01-15', '2024-01-15', '2024-01-15', '2024-01-15', '2024-01-15'],
            'Cost_Per_Unit': [30, 35, 28, 40, 32],
            'Current_Inventory': [150, 200, 80, 250, 120]
        })
        
        st.dataframe(sample_data, width='stretch')
        
        # Generate sample data button
        if st.button("🎲 Generate Sample Data for Testing"):
            df = generate_sample_manufacturing_data()
            st.session_state.manufacturing_data = df
            st.success("✅ Sample data generated! You can now explore other sections.")
            st.rerun()

def show_sku_analysis():
    st.header("🏷️ SKU Analysis")
    
    if 'manufacturing_data' not in st.session_state:
        st.warning("⚠️ Please upload your data first in the 'Data Upload & Overview' section.")
        return
    
    df = st.session_state.manufacturing_data
    
    # Check for required columns
    required_cols = ['SKU']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        return
    
    # SKU selection
    st.subheader("🔍 SKU Selection")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_skus = st.multiselect(
            "Select SKUs to analyze:",
            options=df['SKU'].unique(),
            default=df['SKU'].unique()[:5] if len(df['SKU'].unique()) > 5 else df['SKU'].unique()
        )
    
    with col2:
        analysis_period = st.selectbox(
            "Analysis Period:",
            ["All Time", "Last 30 Days", "Last 90 Days", "Last Year"]
        )
    
    if not selected_skus:
        st.info("Please select at least one SKU to analyze.")
        return
    
    # Filter data
    filtered_df = df[df['SKU'].isin(selected_skus)]
    
    # SKU Performance Overview
    st.subheader("📊 SKU Performance Overview")
    
    if 'Sales' in df.columns and 'Units_Sold' in df.columns:
        sku_summary = filtered_df.groupby('SKU').agg({
            'Sales': ['sum', 'mean'],
            'Units_Sold': ['sum', 'mean'],
            'Cost_Per_Unit': 'mean' if 'Cost_Per_Unit' in df.columns else lambda x: 0,
            'Current_Inventory': 'mean' if 'Current_Inventory' in df.columns else lambda x: 0
        }).round(2)
        
        sku_summary.columns = ['Total Sales', 'Avg Sales', 'Total Units', 'Avg Units', 'Avg Cost', 'Avg Inventory']
        sku_summary['Profit Margin'] = ((sku_summary['Total Sales'] - (sku_summary['Total Units'] * sku_summary['Avg Cost'])) / sku_summary['Total Sales'] * 100).round(2)
        
        st.dataframe(sku_summary, width='stretch')
        
        # Top performing SKUs
        st.subheader("🏆 Top Performing SKUs")
        col1, col2 = st.columns(2)
        
        with col1:
            top_sales = sku_summary.nlargest(5, 'Total Sales')[['Total Sales', 'Total Units']]
            fig = px.bar(
                x=top_sales.index, 
                y=top_sales['Total Sales'],
                title="Top 5 SKUs by Sales",
                labels={'x': 'SKU', 'y': 'Total Sales ($)'}
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            top_units = sku_summary.nlargest(5, 'Total Units')[['Total Units', 'Total Sales']]
            fig = px.bar(
                x=top_units.index, 
                y=top_units['Total Units'],
                title="Top 5 SKUs by Units Sold",
                labels={'x': 'SKU', 'y': 'Total Units'}
            )
            st.plotly_chart(fig, width='stretch')
        
        # SKU profitability analysis
        st.subheader("💰 Profitability Analysis")
        fig = px.scatter(
            sku_summary, 
            x='Total Units', 
            y='Profit Margin',
            size='Total Sales',
            hover_data=['Total Sales', 'Avg Cost'],
            title="SKU Profitability vs Volume"
        )
        fig.update_layout(
            xaxis_title="Total Units Sold",
            yaxis_title="Profit Margin (%)"
        )
        st.plotly_chart(fig, width='stretch')
        
    # Individual SKU deep dive
    if len(selected_skus) == 1:
        st.subheader(f"🔍 Deep Dive: {selected_skus[0]}")
        sku_data = filtered_df[filtered_df['SKU'] == selected_skus[0]]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            total_sales = sku_data['Sales'].sum() if 'Sales' in sku_data.columns else 0
            st.metric("Total Sales", f"${total_sales:,.2f}")
        
        with col2:
            total_units = sku_data['Units_Sold'].sum() if 'Units_Sold' in sku_data.columns else 0
            st.metric("Total Units Sold", f"{total_units:,}")
        
        with col3:
            avg_price = (total_sales / total_units) if total_units > 0 else 0
            st.metric("Average Price", f"${avg_price:.2f}")
        
        # Time series if date column exists
        if 'Date' in sku_data.columns:
            sku_data['Date'] = pd.to_datetime(sku_data['Date'])
            daily_sales = sku_data.groupby('Date')['Sales'].sum().reset_index()
            
            fig = px.line(daily_sales, x='Date', y='Sales', title=f"Sales Trend for {selected_skus[0]}")
            st.plotly_chart(fig, width='stretch')

def show_regional_analysis():
    st.header("🌍 Regional Analysis")
    
    if 'manufacturing_data' not in st.session_state:
        st.warning("⚠️ Please upload your data first in the 'Data Upload & Overview' section.")
        return
    
    df = st.session_state.manufacturing_data
    
    # Check for required columns
    if 'Region' not in df.columns:
        st.error("❌ 'Region' column not found in your data.")
        return
    
    st.subheader("🗺️ Regional Performance Overview")
    
    # Regional metrics
    if 'Sales' in df.columns and 'Units_Sold' in df.columns:
        regional_summary = df.groupby('Region').agg({
            'Sales': 'sum',
            'Units_Sold': 'sum',
            'SKU': 'nunique'
        }).round(2)
        regional_summary.columns = ['Total Sales', 'Total Units', 'Unique SKUs']
        regional_summary['Avg Price'] = (regional_summary['Total Sales'] / regional_summary['Total Units']).round(2)
        
        st.dataframe(regional_summary, width='stretch')
        
        # Regional visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales by region pie chart
            fig = px.pie(
                values=regional_summary['Total Sales'], 
                names=regional_summary.index,
                title="Sales Distribution by Region"
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            # Units by region bar chart
            fig = px.bar(
                x=regional_summary.index, 
                y=regional_summary['Total Units'],
                title="Units Sold by Region",
                labels={'x': 'Region', 'y': 'Total Units'}
            )
            st.plotly_chart(fig, width='stretch')
        
        # SKU performance by region
        st.subheader("🏷️ SKU Performance by Region")
        
        region_sku_pivot = df.pivot_table(
            values='Sales', 
            index='SKU', 
            columns='Region', 
            aggfunc='sum', 
            fill_value=0
        )
        
        # Heatmap of SKU performance by region
        fig = px.imshow(
            region_sku_pivot.values,
            x=region_sku_pivot.columns,
            y=region_sku_pivot.index,
            aspect="auto",
            title="SKU Sales Heatmap by Region",
            labels=dict(x="Region", y="SKU", color="Sales")
        )
        st.plotly_chart(fig, width='stretch')
        
        # Top performing SKUs by region
        st.subheader("🎯 Top SKUs by Region")
        selected_region = st.selectbox("Select Region:", df['Region'].unique())
        
        region_data = df[df['Region'] == selected_region]
        top_skus = region_data.groupby('SKU')['Sales'].sum().sort_values(ascending=False).head(10)
        
        fig = px.bar(
            x=top_skus.values,
            y=top_skus.index,
            orientation='h',
            title=f"Top 10 SKUs in {selected_region}",
            labels={'x': 'Total Sales ($)', 'y': 'SKU'}
        )
        st.plotly_chart(fig, width='stretch')
        
        # Regional trends over time
        if 'Date' in df.columns:
            st.subheader("📈 Regional Sales Trends")
            df['Date'] = pd.to_datetime(df['Date'])
            
            monthly_regional = df.groupby([df['Date'].dt.to_period('M'), 'Region'])['Sales'].sum().reset_index()
            monthly_regional['Date'] = monthly_regional['Date'].astype(str)
            
            fig = px.line(
                monthly_regional, 
                x='Date', 
                y='Sales', 
                color='Region',
                title="Monthly Sales Trends by Region"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, width='stretch')


def show_manufacturing_recommendations():
    st.header("🎯 Manufacturing Recommendations")
    
    if 'manufacturing_data' not in st.session_state:
        st.warning("⚠️ Please upload your data first in the 'Data Upload & Overview' section.")
        return
    
    df = st.session_state.manufacturing_data
    
    st.subheader("📊 Minimum Manufacturing Quantity Analysis")
    
    # Parameters for MOQ calculation
    col1, col2 = st.columns(2)
    
    with col1:
        safety_stock_days = st.slider("Safety Stock (days):", 7, 90, 30)
        lead_time_days = st.slider("Lead Time (days):", 1, 60, 14)
    
    with col2:
        order_frequency = st.selectbox("Order Frequency:", ["Weekly", "Monthly", "Quarterly"])
        service_level = st.slider("Service Level (%):", 85, 99, 95)
    
    # Calculate recommendations for each SKU
    if 'Units_Sold' in df.columns and 'SKU' in df.columns:
        sku_analysis = df.groupby('SKU').agg({
            'Units_Sold': ['sum', 'mean', 'std'],
            'Sales': 'sum' if 'Sales' in df.columns else lambda x: 0,
            'Current_Inventory': 'mean' if 'Current_Inventory' in df.columns else lambda x: 0
        }).round(2)
        
        sku_analysis.columns = ['Total_Units', 'Avg_Daily_Demand', 'Demand_Std', 'Total_Sales', 'Current_Stock']
        
        # Calculate MOQ recommendations
        sku_analysis['Safety_Stock'] = (sku_analysis['Avg_Daily_Demand'] * safety_stock_days).round(0)
        sku_analysis['Lead_Time_Demand'] = (sku_analysis['Avg_Daily_Demand'] * lead_time_days).round(0)
        
        # Order quantity based on frequency
        frequency_multiplier = {"Weekly": 7, "Monthly": 30, "Quarterly": 90}
        sku_analysis['Recommended_Order_Qty'] = (
            sku_analysis['Avg_Daily_Demand'] * frequency_multiplier[order_frequency]
        ).round(0)
        
        sku_analysis['Min_Manufacturing_Qty'] = (
            sku_analysis['Safety_Stock'] + 
            sku_analysis['Lead_Time_Demand'] + 
            sku_analysis['Recommended_Order_Qty']
        ).round(0)
        
        # Stock status
        sku_analysis['Stock_Status'] = sku_analysis.apply(
            lambda row: 'Low' if row['Current_Stock'] < row['Safety_Stock'] 
            else 'Adequate' if row['Current_Stock'] < row['Min_Manufacturing_Qty']
            else 'High', axis=1
        )
        
        # Display recommendations
        st.subheader("📋 SKU Manufacturing Recommendations")
        
        # Color coding for stock status
        def color_stock_status(val):
            if val == 'Low':
                return 'color: red'
            elif val == 'Adequate':
                return 'color: orange'
            else:
                return 'color: green'
        
        display_df = sku_analysis[['Avg_Daily_Demand', 'Current_Stock', 'Safety_Stock', 
                                   'Min_Manufacturing_Qty', 'Stock_Status']].copy()
        display_df.columns = ['Daily Demand', 'Current Stock', 'Safety Stock', 
                             'Min Manufacturing Qty', 'Stock Status']
        
        st.dataframe(
            display_df.style.applymap(color_stock_status, subset=['Stock Status']),
            width='stretch'
        )
        
        # Priority SKUs that need attention
        st.subheader("🚨 Priority SKUs Requiring Action")
        
        low_stock_skus = sku_analysis[sku_analysis['Stock_Status'] == 'Low']
        if not low_stock_skus.empty:
            st.error(f"⚠️ {len(low_stock_skus)} SKUs have critically low stock!")
            
            priority_df = low_stock_skus[['Current_Stock', 'Safety_Stock', 'Min_Manufacturing_Qty']].copy()
            priority_df['Immediate_Need'] = priority_df['Min_Manufacturing_Qty'] - priority_df['Current_Stock']
            
            st.dataframe(priority_df, width='stretch')
        else:
            st.success("✅ All SKUs have adequate stock levels!")
        
        # Manufacturing schedule visualization
        st.subheader("📅 Recommended Manufacturing Schedule")
        
        # Create a manufacturing priority score
        sku_analysis['Priority_Score'] = (
            (sku_analysis['Total_Sales'] / sku_analysis['Total_Sales'].max()) * 0.4 +  # Sales weight
            (sku_analysis['Avg_Daily_Demand'] / sku_analysis['Avg_Daily_Demand'].max()) * 0.3 +  # Demand weight
            ((sku_analysis['Min_Manufacturing_Qty'] - sku_analysis['Current_Stock']) / 
             sku_analysis['Min_Manufacturing_Qty'].max()) * 0.3  # Urgency weight
        )
        
        top_priority = sku_analysis.nlargest(10, 'Priority_Score')
        
        fig = px.bar(
            x=top_priority.index,
            y=top_priority['Min_Manufacturing_Qty'],
            title="Top 10 Priority SKUs for Manufacturing",
            labels={'x': 'SKU', 'y': 'Recommended Manufacturing Quantity'},
            color=top_priority['Priority_Score'],
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, width='stretch')
        
        # Export recommendations
        if st.button("📥 Download Manufacturing Recommendations"):
            output_df = sku_analysis.reset_index()
            csv = output_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"manufacturing_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


@st.cache_data
def generate_sample_manufacturing_data():
    """Generate sample manufacturing data for testing"""
    np.random.seed(42)
    
    # SKUs
    skus = [f"SKU{str(i).zfill(3)}" for i in range(1, 51)]  # 50 SKUs
    products = [f"Product_{i}" for i in range(1, 51)]
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    # Generate data for the last 90 days
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    
    data = []
    for date in dates:
        for sku, product in zip(skus, products):
            # Each SKU appears in 1-3 regions per day with some randomness
            num_regions = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            selected_regions = np.random.choice(regions, size=num_regions, replace=False)
            
            for region in selected_regions:
                if np.random.random() > 0.7:  # 30% chance of having sales on any given day
                    units_sold = np.random.poisson(20) + 1  # 1-40 units typically
                    cost_per_unit = np.random.normal(25, 5)  # $20-30 typically
                    price_per_unit = cost_per_unit * np.random.uniform(1.2, 2.0)  # 20-100% markup
                    sales = units_sold * price_per_unit
                    current_inventory = np.random.randint(50, 500)
                    
                    data.append({
                        'SKU': sku,
                        'Product_Name': product,
                        'Region': region,
                        'Date': date,
                        'Units_Sold': units_sold,
                        'Cost_Per_Unit': round(cost_per_unit, 2),
                        'Sales': round(sales, 2),
                        'Current_Inventory': current_inventory
                    })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    main()
