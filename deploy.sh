#!/bin/bash

# Deployment script for Manufacturing Insights Dashboard
echo "🏭 Manufacturing Insights Dashboard Deployment"
echo "============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the application
echo "Starting Streamlit application..."
echo "Access your dashboard at: http://localhost:8501"
streamlit run app.py
