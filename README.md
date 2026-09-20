cat << 'EOF' > README.md
# 🏡 RealEstateIQ — AI House Price Prediction & Valuation Platform

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://housepriceprediction-sofkbghj7kkbmkyqtuq63p.streamlit.app/)
![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![ML](https://img.shields.io/badge/Model-XGBoost%20%7C%20Scikit--Learn-orange.svg)
![Visuals](https://img.shields.io/badge/Charts-Plotly-purple.svg)
![PDF Export](https://img.shields.io/badge/Reports-FPDF2-green.svg)

🌐 **Live Application**: [https://housepriceprediction-sofkbghj7kkbmkyqtuq63p.streamlit.app/](https://housepriceprediction-sofkbghj7kkbmkyqtuq63p.streamlit.app/)

RealEstateIQ is an intelligent, end-to-end real estate valuation platform built with **Streamlit**, **XGBoost**, **Plotly**, and **FPDF2**. It delivers automated property appraisals, interactive micro-market analytics, loan EMI calculators, and downloadable PDF appraisal certificates.

---

## 🌟 Key Features

* **⚡ Instant AI Property Valuation**: Precision price estimation based on built-up area, BHK configuration, location tier, property age, and furnishings.
* **🎯 Valuation Range & Metrics**: Displays a $\pm10\%$ fair market range, price per sq.ft rate, and dynamic valuation gauge.
* **🏦 Integrated Mortgage & EMI Calculator**: Calculates 20% down payment and 20-year monthly EMI estimates (80% LTV @ 8.5% p.a.).
* **📊 Interactive Market Intelligence**:
  * Scatter plot with trendlines highlighting your subject property.
  * Micro-market benchmark comparisons across neighborhood zones.
  * Furnishing price distributions and age depreciation curves.
* **📄 Downloadable PDF Appraisal Reports**: 1-click export of official appraisal certificates with property specs, financial schedules, and model metadata.
* **🤖 Architecture & Interpretability**: Breakdown of model specifications ($R^2 = 0.894$, MAE, RMSE) and feature importance rankings.

---

## 🛠️ Tech Stack

* **Frontend / UI**: [Streamlit](https://streamlit.io/) with Custom CSS Glassmorphism
* **Machine Learning**: [XGBoost](https://xgboost.readthedocs.io/), [Scikit-Learn](https://scikit-learn.org/)
* **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Joblib](https://joblib.readthedocs.io/)
* **Visualization**: [Plotly Express & Graph Objects](https://plotly.com/python/)
* **Document Export**: [FPDF2](https://py-pdf.github.io/fpdf2/)

---

## 🚀 Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/prajwalferrao/HousePricePrediction.git
cd HousePricePrediction

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run application
streamlit run app.py
