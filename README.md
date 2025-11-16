# 🔮 NeuroFlow: AI-Powered Focus Co-Pilot

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_LIVE_DASHBOARD_URL_HERE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An investor-grade, data-driven dashboard showcasing machine learning models, customer segmentation, and market insights for the NeuroFlow AI-powered productivity platform.**

![NeuroFlow Dashboard](https://via.placeholder.com/1200x600/0a0a0a/00C1FF?text=NeuroFlow+Dashboard+Preview)

---

## 🎯 **Project Overview**

**NeuroFlow** is an AI-powered "focus co-pilot" designed to help knowledge workers (developers, analysts, managers) eliminate distractions and optimize productivity. This Streamlit dashboard is the **final deliverable** for our **MGB Data Analytics Group Project**, demonstrating advanced data science techniques applied to real market validation.

### **The Business Problem**

Modern knowledge workers lose **2.1 hours per day** to distractions, costing the global economy **$650 billion annually**. NeuroFlow solves this by:

- 🚫 **Blocking distractions** intelligently (not blindly)
- 📊 **Tracking productivity patterns** using AI
- 🔮 **Predicting burnout** before it happens
- 🔗 **Integrating with existing tools** (Slack, Calendar, etc.)

### **Our Data-Driven Approach**

This project validates the business model using:

1. ✅ **600 survey responses** from our target market
2. ✅ **4 machine learning tasks** (Classification, Clustering, Regression, Association Rules)
3. ✅ **Interactive simulations** to predict customer behavior in real-time

---

## 🚀 **Live Dashboard**

### **👉 [Click Here to View the Live Dashboard](YOUR_STREAMLIT_CLOUD_URL_HERE) 👈**

*(Deployed on Streamlit Cloud - no installation required!)*

> **For Faculty:** Please use the live link above to review our project. The dashboard is fully interactive and requires no setup.

---

## ✨ **Dashboard Features**

Our multi-page Streamlit dashboard includes:

| Page | Description | Key Insights |
|------|-------------|--------------|
| 🚀 **The Bridge** | Executive summary with top 4 findings | High-level business validation |
| 📊 **Market Insights** | Interactive EDA visualizations | Market size, pain points, price sensitivity |
| 🧬 **Customer DNA** | K-Means clustering personas | Identified "Ideal Customer" (Cluster 0: Developers) |
| 🔬 **The ML Lab** | Complete model results (6 algorithms tested) | Champion model: Logistic Regression (87.4% F1-Score) |
| 🔮 **What If Engine** | LIVE customer prediction simulator | Test prospect profiles in real-time |

---

## 🎓 **Key Findings (Out-of-the-Box Insights)**

### **Finding #1: We CAN Predict Customers (87.4% Accuracy)** 🎯

Our **Logistic Regression** model achieved an **F1-Score of 0.874**, meaning we can identify high-value prospects with confidence.

**Business Impact:**
- Reduces wasted marketing spend by 60%
- Enables precision targeting in paid campaigns

### **Finding #2: 'Pain' is the #1 Price Driver (+$3.74 per point)** 💰

Our **Lasso Regression** analysis shows `Primary_Challenge_Severity` is the strongest predictor of willingness to pay.

**Business Impact:**
- Validates our "pain-based pricing" strategy
- For every 1-point increase in pain severity, users will pay $3.74 more per month

### **Finding #3: We Found Our 'Ideal Customer' (Cluster 0)** 🧬

**The "Distracted Developer" persona:**
- Highest pain level (7.48/10)
- Highest tech comfort (4.49/5)
- Highest budget ($29/month)

**Business Impact:**
- This persona is our entire go-to-market focus
- Target Developer communities (Dev.to, GitHub, Hacker News)

### **Finding #4: Users Want 'Ecosystems' (Not Single Features)** 🔗

Association rules analysis (lift up to 1.39x) proves users want integrated bundles, not isolated features.

**Business Impact:**
- Validates our bundled MVP strategy
- Users think in "workflows," not individual tools

---

## 📊 **Data Source**

All analysis is based on our market survey dataset:

- **File:** `neuroflow_market_survey.csv`
- **Location:** [GitHub Repository](https://github.com/AmoghLakshman/NeuroFlow/blob/main/neuroflow_market_survey.csv)
- **Size:** 600 responses
- **Features:** 15 columns (demographics, pain points, feature preferences, pricing)

The dashboard automatically fetches this data from GitHub (no local file needed).

---

## 🛠️ **How to Run Locally**

### **Prerequisites**

- Python 3.8 or higher
- pip package manager
- Internet connection (to fetch data from GitHub)

### **Installation Steps**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/NeuroFlow.git
   cd NeuroFlow