# 🏭 Manufacturing Insights Dashboard

A comprehensive Streamlit web application for manufacturing companies to analyze SKU performance, regional trends, and get data-driven manufacturing recommendations.

## 🚀 Features

- **📤 Data Upload & Overview**: Upload Excel files and get instant insights
- **🏷️ SKU Analysis**: Analyze individual SKUs, performance metrics, and inventory levels
- **🌍 Regional Analysis**: Understand geographic performance and regional trends
- **🎯 Manufacturing Recommendations**: Get AI-powered recommendations for optimal production quantities

## 📊 Key Capabilities

- Upload and analyze Excel files with manufacturing data
- Track SKU performance across different regions
- Calculate minimum manufacturing quantities based on configurable parameters
- Visualize trends and patterns in your data
- Generate actionable insights for production planning
- Export manufacturing recommendations

## 🛠️ Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd mnfctr
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## 📋 Data Format

Your Excel file should contain the following columns:
- `SKU`: Stock Keeping Unit identifier
- `Product_Name`: Name of the product
- `Region`: Geographic region
- `Sales`: Revenue amount
- `Units_Sold`: Number of units sold
- `Date`: Date of the transaction
- `Cost_Per_Unit`: Cost per unit
- `Current_Inventory`: Current inventory level

## 🔗 Live Demo

Access the live application at: [Manufacturing Insights Dashboard](https://manufacturing-insights-dashboard.streamlit.app/)

*Note: The app URL will be available after deployment. It typically follows the format: `https://[app-name]-[random-string].streamlit.app`*

## 📈 Usage

1. Navigate to **Data Upload & Overview** to upload your Excel file
2. Explore your SKUs in the **SKU Analysis** section
3. Check regional performance in **Regional Analysis**
4. Get recommendations in **Manufacturing Recommendations**

## Project Structure

```
mnfctr/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── pages/             # Additional pages (optional)
├── data/              # Data files (optional)
├── assets/            # Images, CSS, etc. (optional)
└── config/            # Configuration files (optional)
```

## Customization

- **Modify `app.py`** to add your own content and functionality
- **Add new pages** in the `pages/` directory (Streamlit will auto-discover them)
- **Update styling** by adding custom CSS
- **Add data sources** by placing files in the `data/` directory

## Development Tips

- Use `st.cache_data` for expensive computations
- Organize code into functions for better maintainability
- Use `st.session_state` for stateful applications
- Test your app with different screen sizes using the layout options

## Deployment

You can deploy this app to various platforms:

- **Streamlit Cloud**: Connect your GitHub repo
- **Heroku**: Add a `Procfile` and deploy
- **Docker**: Create a `Dockerfile` for containerized deployment

## Need Help?

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Community](https://discuss.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)
