# 🏠 Gurgaon House Pricing Prediction 2.0

A machine learning project that predicts **house prices in Gurgaon** using **Scikit-Learn**.
It analyzes features such as area, number of bedrooms, bathrooms, and location to estimate realistic market prices for residential properties.

---

## 📋 Table of Contents

* [Overview](#-overview)
* [Tech Stack](#-tech-stack)
* [Dataset](#-dataset)
* [Project Workflow](#-project-workflow)
* [Repository Structure](#-repository-structure)
* [Results](#-results)
* [How to Run](#-how-to-run)
* [Future Improvements](#-future-improvements)
* [License](#-license)
* [Author](#-author)

---

## 🧠 Overview

This project demonstrates how **machine learning** can be used to predict **real estate prices** based on key housing features.
The model uses **supervised learning (regression)** to estimate property prices within Gurgaon city, one of India’s fastest-growing real estate markets.

The focus is on:

* Cleaning and preprocessing messy real-estate data
* Building a predictive model with **Scikit-Learn**
* Saving the trained model pipeline for easy reuse
* Generating predictions for new input data

---

## ⚙️ Tech Stack

* **Language:** Python 3.x
* **Libraries:**

  * Scikit-Learn
  * Pandas
  * NumPy
  * Matplotlib / Seaborn
* **Tools:** Jupyter Notebook / VS Code

---

## 🧩 Dataset

The dataset (`housing.csv`) includes listings from Gurgaon with features like:

| Feature    | Description                       |
| ---------- | --------------------------------- |
| `location` | Area or sector of Gurgaon         |
| `sqft`     | Total built-up area (square feet) |
| `bhk`      | Number of bedrooms                |
| `bath`     | Number of bathrooms               |
| `price`    | Property price (target variable)  |

You can modify or replace it with updated Gurgaon housing data as needed.

---

## 🔁 Project Workflow

1. **Data Cleaning**

   * Handled missing values and duplicate entries
   * Removed outliers in `price` and `sqft`

2. **Feature Engineering**

   * Encoded categorical features like `location`
   * Created new features such as `price_per_sqft`

3. **Model Training**

   * Algorithms tested:

     * Linear Regression
     * Decision Tree Regressor
     * Random Forest Regressor
   * Chose the best based on R² score and RMSE

4. **Model Saving**

   * The trained pipeline is stored as `pipeline.pkl` for inference

5. **Prediction**

   * Reads new data from `input.csv`
   * Generates predictions into `output.csv`

---

## 📂 Repository Structure

```
Project-Gurgaon-House-Pricing-2.0/
├── housing.csv
├── input.csv
├── main.py
├── main_01.py
├── main_02(version.ipynb).py
├── main_03(train&inference).py
├── pipeline.pkl
├── output.csv
├── LICENSE
└── README.md
```

---

## 📈 Results

Example prediction:

```python
Input → Sector 56 | 3 BHK | 1800 sqft  
Predicted Price → ₹1.24 Crore
```

The model achieves strong accuracy after cleaning and proper feature scaling.

---

## 🚀 How to Run

1. **Clone this repository**

   ```bash
   git clone https://github.com/samiran123-pappu/Project-Gurgaon-House-Pricing-2.0.git
   cd Project-Gurgaon-House-Pricing-2.0
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the project**

   ```bash
   python main.py
   ```

   or open `main_03(train&inference).py` for full training + prediction.

4. **View results**
   Check `output.csv` for predicted prices.

---

## 🌱 Future Improvements

* Integrate **XGBoost** or **Gradient Boosting** for higher accuracy
* Add **Streamlit UI** for interactive web app
* Visualize Gurgaon map with property clusters
* Automate retraining with updated datasets

---

## 📄 License

This project is licensed under the **MIT License**.
See [LICENSE](LICENSE) for full details.

---

## ✍️ Author

**Samiran Sarkar**
📧 Email: *(optional — add if you want)*
🔗 GitHub: [samiran123-pappu](https://github.com/samiran123-pappu)

---

⭐ *If you found this helpful, consider giving the repo a star!*
