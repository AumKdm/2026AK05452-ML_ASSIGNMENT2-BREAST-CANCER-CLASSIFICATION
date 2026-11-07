# Breast Cancer Classification – ML Model Comparison & Streamlit App

## a. Problem Statement

The goal is to predict whether a breast tumour is **Malignant (1)** or **Benign (0)** using features measured from a digitised image of a fine needle aspirate (FNA) of the breast mass.

Five classification models are trained on the same dataset and compared using six evaluation metrics. The trained models are then served through an interactive Streamlit web app. In the app, a user uploads test data, picks a model, and sees the metrics and confusion matrix. The models are not retrained in the app.

## b. Dataset Description

| Item | Details |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI Machine Learning Repository (also bundled in scikit-learn as `load_breast_cancer`) |
| Link | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |
| Instances | 569 (meets the minimum of 500) |
| Columns | 31 (30 features + 1 target) |
| Features | 30, all numeric (continuous); meets the minimum of 12 |
| Target | `diagnosis`: 1 = Malignant, 0 = Benign |
| Class balance | 212 Malignant (37.3%), 357 Benign (62.7%) |
| Missing values | None |

**Features:** Ten properties are computed for each cell nucleus:
- radius
- texture
- perimeter
- area
- smoothness
- compactness
- concavity
- concave points
- symmetry
- fractal dimension

Each property is reported three ways: as a **mean**, a **standard error**, and a **worst** value (mean of the three largest). That gives 10 × 3 = 30 features, for example `mean_radius`, `radius_error`, and `worst_radius`.

**Preprocessing:**
- 80/20 stratified train/test split (`random_state=42`):

| File | Rows | Columns | Benign (0) | Malignant (1) | % Malignant |
|---|---|---|---|---|---|
| `train_data.csv` | 455 | 31 | 285 | 170 | 37.4 |
| `test_data.csv` | 114 | 31 | 72 | 42 | 36.8 |
| **Total** | **569** | 31 | 357 | 212 | 37.3 |

- Both splits are saved with raw (unscaled) features plus the `diagnosis` column. The models are trained on `train_data.csv`, and every metric below is computed on `test_data.csv`, which is also the file uploaded in the Streamlit app.
- Every model is a scikit-learn `Pipeline` of `StandardScaler` followed by the classifier. The scaler is saved inside each `.pkl` file, so the app can work directly on the raw CSV.

## c. GitHub Repository Link

**Repository:** https://github.com/AumKdm/2026AK05452-ML_ASSIGNMENT2-BREAST-CANCER-CLASSIFICATION

**Live Streamlit App:** https://2026ak05452-breast-cancer-classifier.streamlit.app

**App features:** test CSV upload (with a sample download), a model selection dropdown, all 6 evaluation metrics, a confusion matrix, a classification report and ROC curve, and a side-by-side comparison of all 5 models on the uploaded data. The saved models are only loaded; nothing is retrained.

**Repository structure**

```
project-folder/
├── app.py                              # Streamlit app
├── requirements.txt
├── README.md
├── train_data.csv                      # 455-row training set
├── test_data.csv                       # 114-row held-out test set
├── 2026AK05452_MLAssignment2_BreastCancer.ipynb   # training + evaluation notebook
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── support_vector_machine.pkl
    ├── metadata.json                   # dataset info, feature names, labels, sklearn version
    └── metrics.csv                     # comparison table below
```

## d. Models Used

| # | Model | Key settings |
|---|---|---|
| 1 | Logistic Regression | `max_iter=1000` |
| 2 | Decision Tree Classifier | `max_depth=5` |
| 3 | K-Nearest Neighbor Classifier | `n_neighbors=5` |
| 4 | Naive Bayes Classifier | Gaussian NB, which suits continuous features |
| 5 | Support Vector Machine | RBF kernel, `probability=True` (needed to compute AUC) |

### Comparison Table (on test_data.csv, 114 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | **0.9737** | 0.9947 | **1.0000** | **0.9286** | **0.9630** | **0.9442** |

*Positive class = Malignant (1). The test set has 72 Benign and 42 Malignant cases. Bold marks the best value in each column, except AUC, where Logistic Regression is best.*

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong all-round model with the **highest AUC (0.9960)**, so it ranks malignant cases above benign ones almost perfectly. It made only 4 errors out of 114: 3 missed malignant cases and 1 false alarm. The near-linear separation of the scaled features suits a linear model, making it a close second overall. |
| Decision Tree | The **weakest model** (accuracy 0.9211, lowest AUC of 0.9448 and lowest MCC of 0.8299). It missed 7 of the 42 malignant cases (recall 0.8333), the most of any model. A single tree splits on one feature threshold at a time, so it approximates the smooth class boundary poorly and is sensitive to the training sample, even with `max_depth=5`. |
| KNN | Good performance (accuracy 0.9561, MCC 0.9058) with only 1 false positive, but it missed 4 malignant cases. Scaling inside the pipeline is essential here because KNN is distance-based. With 30 features it is slightly less sharp than LR or SVM, as distances become less informative in higher dimensions. |
| Naive Bayes | Accuracy is tied lowest (0.9211), but its **AUC is high (0.9891)**. It ranks cases well, but its 0.5 threshold decisions are less accurate, giving 3 false positives and 6 missed malignant cases. The independence assumption is violated here: features such as radius, perimeter and area are highly correlated, and this pushes its probabilities toward over-confident values. |
| Support Vector Machine | The **best model**, with the highest accuracy (0.9737), F1 (0.9630) and MCC (0.9442) and **perfect precision (1.0000)**: no benign tumour was wrongly flagged as malignant. It made only 3 errors, all missed malignant cases. The RBF kernel captures a slightly non-linear boundary that a linear model cannot. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (RBF kernel).** It leads on accuracy, F1 and MCC. MCC is the most reliable single metric here because the classes are imbalanced (37% malignant). Logistic Regression is a very close second, with a marginally higher AUC and the same recall. In a medical setting recall on Malignant matters most; SVM and LR tie at 0.9286, so lowering the decision threshold would be the next step to cut missed cancers further. |
