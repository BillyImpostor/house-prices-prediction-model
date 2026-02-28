# 🏠 House Price Prediction — Linear Regression Model

> A supervised machine learning project that predicts house prices per unit area using a Linear Regression algorithm, built with scikit-learn and Python.

---

## 📋 Project Overview

| Attribute     | Details                              |
|---------------|--------------------------------------|
| **Author**    | Cheisa Billy Putra Antoni            |
| **Date**      | January 31, 2026                     |
| **Type**      | Supervised Learning — Regression     |
| **Goal**      | Predict house price of unit area     |
| **Algorithm** | Linear Regression                    |
| **Language**  | Python 3.14                          |

---

## 📁 Project Structure

```
house-prices-prediction-model/
│
├── data/
│   └── real_estate.csv                   # Raw dataset
│
├── notebooks/
│   └── linear_regression.ipynb           # Main analysis notebook
│
├── models/
│   └── house_price_prediction_model.pkl  # Exported trained model (Pipeline)
│
└── README.md
```

---

## 📊 Dataset

- **Source:** Real Estate dataset ([Kaggle](https://www.kaggle.com/datasets/quantbruce/real-estate-price-prediction))
- **Total Records:** 414 rows × 8 columns
- **Missing Values:** None
- **Duplicate Rows:** None

### Features

| Column                                   | Renamed To  | Type    | Description                                    |
|------------------------------------------|-------------|---------|------------------------------------------------|
| X1 transaction date                      | *(dropped)* | float64 | Date of the property transaction               |
| X2 house age                             | `Age`       | float64 | Age of the house (years), range: 0–43.8        |
| X3 distance to the nearest MRT station   | `Station`   | float64 | Distance to nearest MRT (m), range: 23–6488    |
| X4 number of convenience stores          | `Stores`    | int64   | Number of nearby convenience stores (0–10)     |
| X5 latitude                              | `Latitude`  | float64 | Geographic latitude coordinate                 |
| X6 longitude                             | `Longitude` | float64 | Geographic longitude coordinate                |
| **Y house price of unit area**           | `Price`     | float64 | **Target variable** — unit price (7.6–117.5)   |

> **Note:** `No` (row index) and `X1 transaction date` were dropped during preprocessing as they are not predictive features.

### Descriptive Statistics

| Statistic | Age       | Station     | Stores   | Latitude  | Longitude  | Price     |
|-----------|-----------|-------------|----------|-----------|------------|-----------|
| Count     | 414       | 414         | 414      | 414       | 414        | 414       |
| Mean      | 17.71     | 1083.89     | 4.09     | 24.97     | 121.53     | 37.98     |
| Std       | 11.39     | 1262.11     | 2.95     | 0.012     | 0.015      | 13.61     |
| Min       | 0.00      | 23.38       | 0.00     | 24.93     | 121.47     | 7.60      |
| Max       | 43.80     | 6488.02     | 10.00    | 25.01     | 121.57     | 117.50    |

---

## ⚙️ Workflow

### 1. Setup & Configuration
- Imported libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`
- Configured plot styling with Seaborn and Matplotlib

### 2. Data Description
- Loaded and inspected the raw dataset (`df.head()`, `df.shape`, `df.info()`)
- Reviewed value ranges per column

### 3. Data Preprocessing
- Dropped unused columns: `No`, `X1 transaction date`
- Renamed remaining feature columns for cleaner readability
- Checked for **missing values** → none found
- Checked for **duplicate rows** → none found
- Reviewed **descriptive statistics** per feature
- Detected **outliers** using boxplots across all features
- Visualized feature distributions using **pairplot**

### 4. Exploratory Data Analysis (EDA)
- **Lineplot of Average House Prices by Age** — trend of average price across age groups
- **Barplot of Number of Houses by Age** — distribution of house count per age group
- **Pearson Correlation Heatmap** — correlation analysis between all features and the target variable

### 5. Modelling
- **Feature set (X):** `Age`, `Station`, `Stores`, `Latitude`, `Longitude`
- **Target (Y):** `Price`
- **Train/Test Split:** 80% training — 20% testing (`random_state=42`)
- **Scaling:** `StandardScaler` applied on training data (`fit_transform`) and test data (`transform`)
- **Model:** `LinearRegression()` fitted directly on scaled training data
- Inspected model **intercept** and **coefficients** after training

### 6. Model Evaluation
Model performance was evaluated on the test set using the following metrics:

| Metric       | Value      |
|--------------|------------|
| **MSE**      | 54.5809    |
| **RMSE**     | 7.3879     |
| **MAE**      | 5.3501     |
| **R² Score** | **0.6746** |

> The model explains approximately **67.46%** of the variance in house prices. An R² of ~0.67 indicates a moderately good fit for a baseline linear model.

- Visualized **Actual vs. Predicted** values using a line plot on the test set

### 7. Exporting
- Final model saved as a `scikit-learn Pipeline` (StandardScaler + LinearRegression) using `joblib`:
  ```python
  joblib.dump(pipeline, 'house_price_prediction_model.pkl')
  ```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Notebook

```bash
jupyter notebook notebooks/linear_regression.ipynb
```

### Loading the Saved Model

```python
import joblib
import numpy as np

# Load the pipeline
pipeline = joblib.load('models/house_price_prediction_model.pkl')

# Example prediction: [Age, Station, Stores, Latitude, Longitude]
sample = np.array([[10.0, 500.0, 5, 24.98, 121.54]])
predicted_price = pipeline.predict(sample)
print(f"Predicted Price: {predicted_price[0]:.2f}")
```

---

## 🛠️ Technologies Used

| Library        | Purpose                                          |
|----------------|--------------------------------------------------|
| `pandas`       | Data loading and manipulation                    |
| `numpy`        | Numerical computation                            |
| `matplotlib`   | Data visualization                               |
| `seaborn`      | Statistical data visualization                   |
| `scikit-learn` | ML modelling, preprocessing, and evaluation      |
| `joblib`       | Model serialization                              |

---

## 📌 Notes

- The `X1 transaction date` feature was excluded from modelling as it represents a temporal identifier rather than a structural property attribute.
- Future improvements could include feature engineering, outlier removal, or experimenting with regularization techniques (Ridge, Lasso) to improve the R² score.

---

## 👤 Author

**Cheisa Billy Putra Antoni**  
📅 January 31, 2026