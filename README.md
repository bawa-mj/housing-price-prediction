# 🏠 Housing Price Predictor
An interactive, machine learning–powered web application designed to estimate residential property values in real time. This tool processes structural and locational property attributes to surface an instant market price estimate, helping users understand valuation drivers before buying, selling, or listing a home.

## 🚀 Key Features Tracked
*   **Price Estimation:** Real-time prediction of market value based on a trained Scikit-learn regression pipeline.
*   **Property Specification Analysis:** Insights driven by area, bedroom/bathroom count, stories, and parking capacity.
*   **Amenity Impact Modeling:** Accounts for main road access, preferred area, guest room, basement, hot water heating, and air conditioning.
*   **Furnishing Status Weighting:** Differentiates value impact across furnished, semi-furnished, and unfurnished properties.

## 📁 Repository Structure
The repository contains the core architectural components of the application:
*   **`main.py`** — The FastAPI backend containing the `/predict` route, feature engineering logic, and Indian currency formatting.
*   **`models/house_price_model.pkl`** — The trained Scikit-learn regression pipeline used for inference.
*   **`templates/index.html`** — The responsive, single-page frontend UI handling user input and displaying predictions.
*   **`README.md`** — Project documentation and setup guide.

## 🛠️ Setup & Local Deployment
### Prerequisites
*   Python 3.8+ installed.
*   pip package manager.

### Installation
```bash
git clone https://github.com/bawa-mj/housing-price-prediction.git
cd housing-price-prediction
pip install fastapi uvicorn pandas numpy scikit-learn joblib
uvicorn main:app --reload
```
Open `http://127.0.0.1:8000` in your browser.

## 📸 Application Preview
### 🎛️ Prediction Form UI
![Housing Price Predictor UI](https://github.com/bawa-mj/housing-price-prediction/blob/main/Housing%20Price%20Predictor.png)

## ⚠️ Disclaimer
This tool provides an estimated value based on a machine learning model trained on historical data. It is intended for reference and educational purposes only, and should not be treated as a certified property appraisal.

## 👤 Author
**Jayant Majumdar** — AI/ML Engineering Student
