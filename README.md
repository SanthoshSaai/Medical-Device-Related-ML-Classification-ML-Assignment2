# Machine Learning — Assignment 2

**Name:** Santhosh Saai S  
**BITS ID:** 2026AJ05047  
**Subject:** Machine Learning  
**Subject Code:** AIMLZG565  
**Assignment:** 2

---

## Assignment 2 — Medical Device-Related ML Classification

### Project: MedGuard — CTG Classification

This project uses Machine Learning classification algorithms to predict the fetal health status using Cardiotocography (CTG) data.

The dataset contains measurements related to fetal heart rate and uterine contractions. The objective is to classify the fetal state into one of the following categories:

- Normal
- Suspect
- Pathological

### Dataset

- **Dataset:** Cardiotocography (CTG)
- **Source:** UCI Machine Learning Repository
- **Instances:** 2,126
- **Input Features:** 21
- **Target:** NSP
- **Type:** Multiclass Classification

### Machine Learning Algorithms

The following five classification algorithms were implemented and compared:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Evaluation Metrics

The models were evaluated using the following metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1-Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison

| Model | Accuracy | AUC |
|---|---:|---:|
| Logistic Regression | 85.68% | 0.9602 |
| Decision Tree | 88.50% | 0.9021 |
| KNN | 89.91% | 0.9640 |
| Gaussian Naive Bayes | 79.81% | 0.8694 |
| SVM | 88.50% | 0.9718 |

KNN achieved the highest accuracy, while SVM achieved the highest AUC among the evaluated models.

The application also displays the confusion matrix and classification report for the selected model.

### Streamlit Application

A Streamlit application was developed to interact with the trained classification models.

The application provides:

- CSV file upload for model evaluation
- Selection of a classification model
- Display of evaluation metrics
- Confusion matrix
- Classification report
- Model comparison

The uploaded test data is used for evaluation and does not retrain the models.

**GitHub Repository:**  
https://github.com/SanthoshSaai/Medical-Device-Related-ML-Classification-ML-Assignment2

**Streamlit Application:**  
[Add the deployed Streamlit URL here]

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

The application will open in the browser.

### Conclusion

In this assignment, five classification algorithms were implemented using the CTG dataset. Their performance was compared using multiple evaluation metrics. KNN achieved the highest accuracy, and SVM achieved the highest AUC. A Streamlit application was developed to evaluate the models and display their results interactively.
