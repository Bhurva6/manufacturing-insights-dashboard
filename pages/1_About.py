import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="About", page_icon="ℹ️")

st.title("ℹ️ About This App")

st.markdown("""
## Overview

This Streamlit application serves as a comprehensive template for building data-driven web applications. 
It demonstrates various Streamlit features and best practices.

## Key Features

### 🏠 Home Dashboard
- Interactive metrics display
- Key performance indicators
- Quick overview of application status

### 📊 Data Analysis
- Sample data generation and exploration
- Data filtering and manipulation
- Statistical analysis display

### 📈 Visualizations
- Multiple charting libraries integration
- Interactive plots with Plotly
- Static plots with Matplotlib/Seaborn
- Built-in Streamlit charts

### 🎮 Interactive Components
- Form inputs and widgets
- File upload functionality
- Dynamic content generation
- State management examples

## Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly, Matplotlib, Seaborn
- **Styling**: Custom CSS (optional)

## Getting Started

1. Clone or download this project
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
4. Open `http://localhost:8501` in your browser

## Customization

This template is designed to be easily customizable:

- Modify the `app.py` file to change the main application logic
- Add new pages in the `pages/` directory
- Update styling with custom CSS
- Integrate your own data sources
- Add authentication if needed

## Best Practices Implemented

- **Caching**: Uses `@st.cache_data` for performance
- **Layout**: Responsive design with columns and containers
- **Navigation**: Clean sidebar navigation
- **Error Handling**: Graceful error management
- **Code Organization**: Modular function structure

## Need Help?

Feel free to explore the code and modify it according to your needs. The Streamlit documentation 
is excellent for learning more advanced features.

---

*Built with ❤️ using Streamlit*
""")

# Add some interactive elements
st.subheader("App Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Lines of Code", "~200", "📝")

with col2:
    st.metric("Dependencies", "7", "📦")

with col3:
    st.metric("Pages", "5", "📄")

# Version info
st.sidebar.markdown("---")
st.sidebar.markdown("**Version**: 1.0.0")
st.sidebar.markdown("**Last Updated**: November 2024")
