# Machine Learning — Assignment 2

**Name:** Santhosh Saai S  
**BITS ID:** 2026AJ05047  
**Subject:** Machine Learning  
**Subject Code:** AIMLZG565  
**Assignment:** 2

---

## Assignment 2 — Medical Device-Related ML Classification

### Project: MedGuard — CTG Classification

This project uses Machine Learning classification algorithms to predict fetal health status using Cardiotocography (CTG) data.

The dataset contains measurements related to fetal heart rate and uterine contractions. The objective is to classify the fetal state into one of the following three categories:

- Normal
- Suspect
- Pathological

### Dataset

- **Dataset:** Cardiotocography (CTG)
- **Source:** UCI Machine Learning Repository
- **Instances:** 2,126
- **Input Features:** 21
- **Target Variable:** NSP
- **Type:** Multiclass Classification

### Machine Learning Algorithms

The following five classification algorithms were implemented and compared:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Evaluation Metrics

The models were evaluated using the following six performance metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1-Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1-Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 85.68% | 0.9602 | 0.7406 | 0.8497 | 0.7798 | 0.6868 |
| Decision Tree | 88.50% | 0.9021 | 0.7961 | 0.8703 | 0.8263 | 0.7240 |
| KNN | 89.91% | 0.9640 | 0.8452 | 0.7470 | 0.7887 | 0.7061 |
| Gaussian Naive Bayes | 79.81% | 0.8694 | 0.6474 | 0.7348 | 0.6706 | 0.5710 |
| SVM | 88.50% | 0.9718 | 0.7912 | 0.8710 | 0.8208 | 0.7346 |

Based on the results, KNN achieved the highest accuracy of 89.91%, while SVM achieved the highest AUC of 0.9718.

### Streamlit Application

A Streamlit web application was developed to interact with the trained classification models and evaluate their performance.

The application provides the following functionalities:

- CSV file upload for model evaluation
- Selection of a classification model
- Display of six evaluation metrics
- Confusion matrix
- Classification report
- Comparison of all five models

The application uses the previously trained models. Uploaded test data is used for evaluation and does not trigger model retraining.

**GitHub Repository:**

[MedGuard — CTG Classification](https://github.com/SanthoshSaai/Medical-Device-Related-ML-Classification-ML-Assignment2)

**Live Streamlit Application:**

[Open MedGuard CTG Application](https://medical-device-related-ml-classification-ml-assignment2-tnsese.streamlit.app/)

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib

### How to Run the Application

Clone the repository:

```bash
git clone https://github.com/SanthoshSaai/Medical-Device-Related-ML-Classification-ML-Assignment2.git
```

Navigate to the project directory:

```bash
cd Medical-Device-Related-ML-Classification-ML-Assignment2
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will be available in the browser at the local Streamlit address displayed in the terminal.

### Conclusion

In this assignment, five Machine Learning classification algorithms were implemented using the CTG dataset. Their performance was evaluated and compared using six evaluation metrics.

KNN achieved the highest accuracy, while SVM achieved the highest AUC. A Streamlit application was also developed to allow users to select classification models, upload test data, and view their evaluation results interactively.
