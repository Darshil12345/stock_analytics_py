# 📊 Global Stock Analytics Platform

A **FastAPI + Dash powered analytics platform** for exploring global stock market data, building datasets, performing exploratory analysis, running machine learning predictions, and analyzing financial news sentiment.

The system integrates **data pipelines, machine learning models, explainability tools (SHAP & LIME), and interactive dashboards**.

---

# 🚀 Features

## 📈 Data Pipeline

* Fetches historical market data from **Yahoo Finance**
* Supports multiple global indices
* Custom index addition
* Feature engineering:

  * OHLC
  * Returns
  * Ratios
  * Time features
  * Custom derived formulas

---

## 📊 Exploratory Data Analysis (EDA)

### Univariate Analysis

* Time series charts
* Histograms + KDE
* Box plots
* Rolling volatility
* Cumulative returns
* Drawdown analysis
* Statistical summaries

### Bivariate Analysis

* Scatter plots
* Scatter with trendline
* Correlation analysis
* Lagged correlations

### Multivariate Analysis

* Correlation matrix
* Scatter matrix
* PCA analysis
* Multi-index comparison

---

## 🤖 Machine Learning

Supported models:

* Logistic Regression
* Random Forest
* Gradient Boosting
* SVC
* KNN

Capabilities:

* Train/test split by date
* Probability threshold tuning
* Model comparison
* Confusion matrix
* ROC curves
* Forecasting future dates

---

## 🔍 Model Explainability

Integrated explainability tools:

* **SHAP**
* **LIME**

Understand:

* Feature importance
* Local predictions
* Global model behavior

---

## 📰 Sentiment Analysis

Sources:

* Upload CSV news dataset
* Fetch articles from **Google News**

Outputs:

* Wordcloud
* Sentiment distribution
* Average sentiment score
* Article preview table

---

## 🌐 External Resources

Quick access to:

* Market tools
* Financial dashboards
* Research resources

---

# 🏗️ Project Architecture

```
stock-analytics
│
├── api/
├── callbacks/
├── models/
├── pipelines/
├── services/
├── utils/
│
├── dashboard.py
├── main.py
├── config.py
├── state.py
│
├── requirements.txt
├── render.yaml
└── README.md
```

---

# ⚙️ Installation (Local)

Clone the repository

```bash
git clone https://github.com/yourusername/stock-analytics.git
cd stock-analytics
```

Create environment

```bash
python -m venv venv
```

Activate environment

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Server

Start the FastAPI server:

```bash
uvicorn main:api --reload
```

---

# 🌐 Access the Platform

API root:

```
http://127.0.0.1:8000
```

Swagger docs:

```
http://127.0.0.1:8000/docs
```

Dashboard:

```
http://127.0.0.1:8000/dashboard
```

---

# ☁️ Deployment (Render)

Push repository to GitHub.

Render automatically reads `render.yaml`.

Deployment command:

```
uvicorn main:api --host 0.0.0.0 --port 10000
```

Dashboard will be available at:

```
https://your-app.onrender.com/dashboard
```

---

# 🧠 Technology Stack

Backend

* FastAPI
* Python

Dashboard

* Dash
* Plotly
* Bootstrap

Machine Learning

* Scikit-learn
* TensorFlow
* SHAP
* LIME

Data

* Pandas
* NumPy
* Yahoo Finance

NLP

* Wordcloud
* Feedparser

---

# 📌 Future Improvements

* Real-time market streaming
* Portfolio optimization
* LSTM price prediction
* Redis caching
* Automated data pipelines
* Docker container deployment

---

# 👨‍💻 Author

Darshil Shetty

Integrated MSc–PhD (Mathematics)
Indian Institute of Science (IISc)

---

# ⭐ If you find this project useful

Please consider giving the repository a **star**.
