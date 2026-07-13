# 📊 End-to-End Sales Forecasting & Demand Intelligence System

An end-to-end data science project that analyzes historical retail sales, forecasts future demand using multiple time-series and machine learning models, detects unusual sales patterns, segments products based on demand behaviour, and presents the results through an interactive Streamlit dashboard.

This project was developed as the **Final Project of the Xylofy Data Science Internship Program**.

---

## 📌 Project Overview

Retail businesses need accurate demand forecasts to make better decisions about inventory, replenishment, and resource planning. However, sales demand is often affected by trends, seasonality, unexpected spikes, and differences in product behaviour.

This project builds a complete sales intelligence system that:

- Analyzes historical sales patterns and business trends.
- Studies trend, seasonality, and residual noise in monthly sales.
- Tests the statistical properties of the sales time series.
- Forecasts future sales using three different modelling approaches.
- Compares forecasting models using common evaluation metrics.
- Detects unusual high- and low-sales weeks.
- Segments product sub-categories based on demand behaviour.
- Provides business-oriented stocking recommendations.
- Presents key insights through an interactive multi-page Streamlit dashboard.

---

## 🎯 Business Objectives

The project aims to answer the following business questions:

- How have sales changed over time?
- Are there recurring seasonal patterns in customer demand?
- Which regions and product categories contribute the most to sales?
- Can future sales be forecast accurately?
- Which forecasting approach performs best?
- Which weeks show unusual sales activity?
- Which products have stable, volatile, high-volume, or rapidly growing demand?
- How should inventory strategies differ across product segments?

---

## 📂 Project Structure

```text
SalesForecasting_NeevKhandelwal/
│
├── analysis.ipynb
├── train.csv
├── summary.pdf
├── requirements.txt
│
├── app.py
