<div align="center">

# 📊 Machine Learning Assignment 2

</div>

# 🎯 Problem Statement

Marketing campaigns often involve contacting thousands of customers, making it important to identify those who are most likely to subscribe to a term deposit. The **BankMarketingProject** addresses this challenge by leveraging customer information and campaign-related attributes to predict the likelihood of a successful subscription.

Multiple machine learning classification models are developed, evaluated, and compared to determine the best-performing approach. The final solution is presented through an interactive Streamlit web application that enables users to explore model predictions and performance...

---


# 📂 Dataset Description

| **Attribute** | **Details** |
|:--------------|:------------|
| **Dataset Name** | Bank Marketing Dataset |
| **Source** | Kaggle |
| **Dataset Type** | Binary Classification |
| **Target Variable** | `y` (Term Deposit Subscription) |
| **Total Records** | 45,211 |
| **Input Features** | 16 |
| **Target Classes** | Yes / No |


## 📑 Feature Categories

The dataset contains **16 input features** describing customer demographics, financial status, communication details, and marketing campaign history. These features are used to predict whether a customer will subscribe to a term deposit.

| **Category** | **Feature** | **Description** | **Possible Values** |
|:-------------|:------------|:----------------|:--------------------|
| 👤 **Customer Demographics** | `age` | Age of the customer | Integer (18–95 years) |
| | `job` | Customer's occupation | `admin.`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown` |
| | `marital` | Marital status | `married`, `single`, `divorced` |
| | `education` | Highest education level | `primary`, `secondary`, `tertiary`, `unknown` |
| 💰 **Financial Information** | `default` | Has credit in default | `yes`, `no` |
| | `balance` | Average yearly account balance | Numeric value (account balance) |
| | `housing` | Has a housing loan | `yes`, `no` |
| | `loan` | Has a personal loan | `yes`, `no` |
| 📞 **Communication Details** | `contact` | Communication type used to contact the customer | `cellular`, `telephone`, `unknown` |
| | `day` | Last contact day of the month | Integer (1–31) |
| | `month` | Last contact month | `jan`, `feb`, `mar`, `apr`, `may`, `jun`, `jul`, `aug`, `sep`, `oct`, `nov`, `dec` |
| | `duration` | Duration of the last contact | Numeric value (seconds) |
| 📢 **Marketing Campaign Information** | `campaign` | Number of contacts during the current campaign | Positive integer |
| | `pdays` | Number of days since the customer was previously contacted (`-1` indicates not previously contacted) | Integer (`-1` or positive value) |
| | `previous` | Number of contacts performed before the current campaign | Non-negative integer |
| | `poutcome` | Outcome of the previous marketing campaign | `success`, `failure`, `other`, `unknown` |

## 🎯 Target Variable

| **Column Name** | **Target Variable** | **Description** | **Possible Values** |
|:---------------|:--------------------|:----------------|:-------------------:|
| `deposit` | **Term Deposit Subscription** | Indicates whether the customer subscribed to a term deposit after the marketing campaign. | `yes`, `no` |
---

# 🔗 GitHub Repository

> **Repository Link**

**👉 [View the GitHub Repository](https://github.com/DivyaPurple/BankMarketingProject)**

---

# 🤖 Models Used

| ✔ | Machine Learning Model |
|:--:|-----------------------|
| ✅ | Logistic Regression |
| ✅ | Decision Tree Classifier |
| ✅ | K-Nearest Neighbors (KNN) |
| ✅ | Gaussian Naive Bayes |
| ✅ | Random Forest Classifier |

---

# 📈 Model Comparison

## 📊 Model Performance Comparison

| **Rank** | **ML Model Name** | **Accuracy** | **AUC** | **Precision** | **Recall** | **F1 Score** | **MCC** |
|:--:|:--------------------------|:----------:|:-------:|:-------------:|:----------:|:------------:|:-------:|
| 1 | **Random Forest (Ensemble)** | **0.8580** | **0.9215** | **0.8236** | **0.8913** | **0.8561** | **0.7186** |
| 2 | **K-Nearest Neighbors (KNN)** | **0.8285** | **0.8923** | **0.8280** | **0.8053** | **0.8165** | **0.6558** |
| 3 | **Logistic Regression** | **0.8253** | **0.9073** | **0.8275** | **0.7977** | **0.8123** | **0.6495** |
| 4 | **Decision Tree** | **0.8249** | **0.8944** | **0.7843** | **0.8696** | **0.8247** | **0.6542** |
| 5 | **Gaussian Naive Bayes** | **0.7568** | **0.8464** | **0.7877** | **0.6664** | **0.7220** | **0.5142** |

---

# 📝 Model Performance Observations

| **ML Model Name** | **Observation about Model Performance** |
|:------------------|:----------------------------------------|
| **Logistic Regression** | Logistic Regression provided a strong and consistent baseline model. It achieved high AUC and precision, demonstrating good class discrimination and reliable generalization. However, its recall was slightly lower than the tree-based models, meaning it missed some positive subscription cases. |
| **Decision Tree** | The Decision Tree effectively captured non-linear relationships in the data and achieved high recall, identifying a large proportion of customers who subscribed. However, its lower precision indicates that it generated more false positive predictions than some of the other models. |
| **K-Nearest Neighbors (KNN)** | KNN delivered balanced performance across all evaluation metrics after feature scaling. It achieved the highest precision, indicating reliable positive predictions, while maintaining competitive accuracy and F1 score. |
| **Gaussian Naive Bayes** | Gaussian Naive Bayes produced the weakest performance among the evaluated models. The algorithm assumes feature independence and normally distributed continuous features, assumptions that are not fully satisfied by the Bank Marketing dataset, resulting in reduced predictive performance. |
| **Random Forest (Ensemble)** | Random Forest consistently outperformed the other models by combining multiple decision trees to reduce overfitting and improve generalization. It achieved the best overall balance across Accuracy, AUC, Recall, F1 Score, and MCC, making it the most reliable classifier for this dataset. |
| **🏆 Overall Winner** | **Random Forest Classifier** was selected as the best-performing model because it demonstrated the strongest overall predictive capability across nearly all evaluation metrics. Its ensemble learning approach improved robustness, reduced overfitting, and provided the most reliable predictions for identifying customers likely to subscribe to a term deposit. |
---

# 🏆 Overall Winner

| **Overall Winner for the Dataset** | 
|------------------------------------|
| **Random Forest (Ensemble)**  | 

---

# 🌐 Streamlit Application

> **Live Application Link**

**👉 https://bankmarketingprojectml1.streamlit.app/**