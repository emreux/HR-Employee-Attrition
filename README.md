## HR Employee Attrition Prediction

This project focuses on predicting employee attrition using advanced machine learning techniques. By analyzing workplace factors such as job satisfaction, overtime, and tenure, the goal is to provide HR departments with data-driven, actionable insights.


### Dataset Overview
The dataset used in this analysis presents a significant class imbalance challenge: <br>
**Stayed (No):** ~1200 employees  
**Left (Yes):** ~200 employees  

Initially, models trained on this raw data were biased toward the majority class, showing poor performance in detecting the minority Yes class. This was clearly evidenced by the high number of false negatives in the initial confusion matrix results.

---

## Methodology & Solutions

###   Exploratory Data Analysis
Detailed visualizations were created using **Seaborn** and **Matplotlib** to identify key correlations and drivers behind attrition, such as work-life balance and age distribution.

### Data Preprocessing
To prepare the data for modeling:
* **Label Encoding** and **One-Hot Encoding** were applied to transform categorical variables.
* Irrelevant features were dropped to improve model focus.

###  Handling Imbalance
To address the lack of "Yes" samples, I applied **SMOTE (Synthetic Minority Over-sampling Technique)**. This balanced the training set artificially, allowing the models to learn the specific characteristics of employees who are likely to resign.

### Machine Learning Models & Optimization
The following algorithms were implemented and compared:
* **Logistic Regression**
* **Random Forest Classifier**
* **XGBoost** <br>

**Hyperparameter Tuning:** I utilized *GridSearchCV* to fine-tune the models. By testing various combinations of hyperparameters, I ensured that the models were not just using default values but were optimized specifically for this dataset's characteristics.


---

## Results
The implementation of **SMOTE** proved to be a turning point. After balancing the data:
* There was a substantial increase in **Recall** and **F1-Score** for the "Attrition: Yes" class.
* The **Confusion Matrix** shifted from ignoring the minority class to identifying potential resignations with much higher accuracy.

---

##  Web App with Streamlit
The final trained model is deployed as an interactive web application built with **Streamlit**. 
* **Input:** Users can manually input specific employee metrics.
* **Output:** The app provides a real-time prediction and probability score for attrition risk.

