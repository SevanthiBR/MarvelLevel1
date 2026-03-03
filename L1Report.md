# Task 1 : MATLAB ML Onramp Course
## Aim

Enroll in and complete the MATLAB Machine Learning Onramp course.

## Learnings

MATLAB is a computational software platform used for performing advanced mathematical analysis, simulations, and programming to solve complex problems.

This course focused on handwritten digit recognition using the MNIST dataset.

The coursework emphasized the core steps involved in a typical machine learning workflow, including:

* Importing and understanding the dataset
* Preprocessing and cleaning the data
* Normalizing or scaling features before training
* Building and training the model
* Evaluating performance using relevant metrics
* Optimizing the model through hyperparameter tuning

It provided hands-on experience with the complete pipeline required to develop and improve a machine learning model.

![Alt text](https://github.com/SevanthiBR/MarvelLevel1/blob/main/Task1Cert.png?raw=true)

# Task 2 :  Kaggle Crafter - Build & Publish Your Own Dataset

## Aim

Create a dataset and upload it to Kaggle with proper metadata and formatting. The dataset should meet the following usability criteria (total score ≥ 8.5)


## Learnings 

Faker is a Python library used to generate fake but realistic-looking data such as names, addresses, cities, company names, dates, and more.It works by using predefined data providers and randomly selecting values from them to generate realistic outputs. It uses Python’s randomization functions along with locale-based data patterns to ensure the generated information looks natural and contextually accurate.

For this task, I created a synthetic dataset of 100 café records using Python, utilizing the pandas library for data handling and the Faker library to generate realistic random data. I began by initializing the Faker object and setting a seed value (42) to ensure reproducibility of results. Then, I created an empty list and used a loop to generate 100 structured café records. For each entry, I generated attributes such as café name, city, rating (between 3.0 and 5.0), price for two (₹200–₹1200), specialty drink, and opening hours, storing them as dictionaries. Finally, I converted the list into a pandas DataFrame and exported it which achieved a 10/10 usability score on Kaggle due to its clean structure and consistent formatting.

[Python code](https://github.com/SevanthiBR/MarvelLevel1/blob/bd2a9fb3cfde404200f1e13b08c5dddb6db1ceb6/Task%202.ipynb)

[Kaggle dataset](https://www.kaggle.com/datasets/sevanthibr/cafe-ratings-and-prices-dataset)

# Task 3 : Data Detox - Data Cleaning using Pandas

## Aim 

Preprocess and clean raw, messy datasets using Pandas for better machine learning outcomes.

## Learnings 

Pandas is a powerful Python library used for data manipulation and analysis. It provides flexible data structures like DataFrames that make it easy to clean, transform, and analyze structured data. Pandas is widely used in data science and machine learning for handling real-world datasets efficiently.

In this task, I worked on cleaning a raw dataset using Pandas to prepare it for analysis and machine learning. I identified issues such as missing values, outliers, text inconsistencies, invalid numeric entries, improper date formats, and potential email errors. I removed rows with null values in critical fields like Email and imputed non-critical categorical nulls with appropriate replacements. For numeric columns like Age and TotalPurchase, I handled outliers and missing values by replacing them with mean values and ensuring valid ranges and correct data types. I corrected typos in categorical fields, standardized text formatting, converted date columns to proper datetime format while maintaining chronological order, validated emails using regex, and removed duplicate rows. After these steps, I ensured the dataset was clean, consistent, and ready for reliable analysis or model building.

[Link to my notebook](https://github.com/SevanthiBR/MarvelLevel1/blob/bd2a9fb3cfde404200f1e13b08c5dddb6db1ceb6/Task3.ipynb)


# Task 4 :  Anomaly Detection

## Aim
To detect unusual patterns in user activity logs using anomaly detection techniques. 


## Learnings 

In this task, I performed anomaly detection using both a statistical method  and a machine learning method . I first selected relevant behavioral features such as login duration, data accessed, and files downloaded. These features were scaled using StandardScaler to ensure consistent magnitudes for machine learning processing.

The IQR method was applied to each feature individually. For every column, I calculated the 25th percentile (Q1) and 75th percentile (Q3), computed the IQR (Q3 − Q1), and flagged values outside 1.5 × IQR as anomalies. If a user was an outlier in any feature, they were marked as a statistical anomaly.

Next, I implemented Isolation Forest with a contamination rate of 5%, assuming approximately 5% suspicious activity. The model built random isolation trees and identified users that were isolated quickly as anomalies. I also computed a risk score using Z-scores to measure how far each user’s behavior deviated from the mean.

The results provided anomaly flags from both statistical and machine learning approaches, along with ranked high-risk users. Combining both methods improved detection reliability by capturing both extreme statistical outliers and unusual multivariate behavior patterns.

[Link to my notebook](https://github.com/SevanthiBR/MarvelLevel1/blob/bd2a9fb3cfde404200f1e13b08c5dddb6db1ceb6/Task4.ipynb)

![Alt text](https://github.com/SevanthiBR/MarvelLevel1/blob/main/Task4img.png?raw=true)



# Task 5 :  Logistic Regression from Scratch

## Aim

To build a logistic regression model from scratch and compare it with a standard library implementation. The chosen use-case: predicting heart disease.

## Learnings
Logistic Regression is a supervised machine learning algorithm used for binary classification problems, where the output variable has two possible outcomes (such as 0 or 1). Unlike linear regression, it does not predict continuous values; instead, it estimates the probability of a data point belonging to a particular class. It uses the sigmoid (logistic) function to convert a linear combination of input features into a probability value between 0 and 1. A threshold (commonly 0.5) is then applied to classify the output into one of the two classes. The model parameters are optimized using techniques like gradient descent to minimize the loss function.

Logistic Regression is a supervised learning algorithm used for binary classification that models the probability of a class using a linear function combined with the sigmoid function. First, a linear combination of inputs is computed as

 ![z-image](https://github.com/SevanthiBR/MarvelLevel1/blob/main/z-eqn.png?raw=true)

where ( w ) represents weights, ( x ) represents features, and ( b ) is the bias term. 
This value is passed through the sigmoid function to obtain the probability ( P(y=1) ).

![sigmoid-function](https://github.com/SevanthiBR/MarvelLevel1/blob/main/sigmoid.png?raw=true)

The model is trained by minimizing the cross-entropy (log-loss) cost function 

![loss function](https://github.com/SevanthiBR/MarvelLevel1/blob/main/loss-func.png?raw=true)

The parameters are updated using gradient descent: 

![gradient-descent](https://github.com/SevanthiBR/MarvelLevel1/blob/main/gradient-descent.png?raw=true)

where (alpha ) is the learning rate. 
Logistic Regression effectively learns a linear decision boundary and models the log-odds as 

![](https://github.com/SevanthiBR/MarvelLevel1/blob/main/logisticregg.png?raw=true)


In this task, I implemented Logistic Regression from scratch using NumPy to understand how the algorithm works internally. I used the Framingham Heart Disease dataset, where the target variable TenYearCHD indicates whether a person has a 10-year risk of heart disease. Since the outcome is binary (0 or 1), Logistic Regression was suitable.

I first handled missing values using mean imputation, then split the data into training (80%) and testing (20%) sets. I applied feature scaling using StandardScaler to improve model performance.

I manually implemented the sigmoid function, calculated gradients, and used gradient descent to update the model parameters. Predictions were made using a 0.5 threshold. The model achieved about 85% training accuracy and 86% test accuracy.Upon comparing my implementation with scikit-learn’s LogisticRegression, and both produced identical results.

[Link to my notebook](https://github.com/SevanthiBR/MarvelLevel1/blob/e6f444245c8091d0896e1020ce61f86ef93dc3f7/Task5.ipynb)


# Task 6 : Battle-Test Your Model - Support Vector Machines

## Aim

Implement SVM on a wine quality dataset and stress-test the model by injecting noise into the data to observe how its performance deteriorates.

## Learning:

Support Vector Machine (SVM) is a supervised machine learning algorithm used for both classification and regression problems, but it is most commonly used for classification. It works by finding the best boundary that separates different classes while keeping the maximum possible distance between them. SVM focuses on the data points closest to the boundary, called support vectors, as they play a key role in defining the decision line. It allows some errors to make the model more flexible, controlled by a parameter called C. When the data cannot be separated using a straight line, SVM uses kernels to create curved boundaries. Overall, SVM is powerful and effective, especially for complex datasets.

**Steps taken:**
* Converted the dataset into a binary classification problem (quality ≥ 6 as good-1, others as bad-0).
* Split the data into training and testing sets and applied feature scaling using StandardScaler.
* Trained an SVM model with RBF kernel to handle non-linear patterns.
* Added Gaussian noise (0–0.5 levels) to test how robust the model is.
* Evaluated accuracy at each level and observed slight improvement at low noise (0.2) but performance drop at higher noise (above 0.3).

![Plot](https://github.com/SevanthiBR/MarvelLevel1/blob/main/Task6img.png?raw=true)

[Link to my notebook](https://github.com/SevanthiBR/MarvelLevel1/blob/e6f444245c8091d0896e1020ce61f86ef93dc3f7/Task6.ipynb)


# Task 7: Fairness Meets Functionality - Decision tree from scratch

## Aim:

To understand and build a Decision Tree from scratch and perform the fairness analysis

## Learnings:

Entropy is a measure of uncertainty or impurity in a dataset. It helps determine how mixed the classes are within a node of the decision tree.
* If all samples in a node belong to the same class, the entropy is low (pure node).
* If the samples are evenly distributed among different classes, the entropy is high (impure node).

The objective of a decision tree is to reduce entropy at each split, resulting in more homogeneous groups.

This task uses the ID3 (Iterative Dichotomiser 3) algorithm, which relies on entropy for splitting decisions.

Entropy is calculated using the formula:

![](https://github.com/SevanthiBR/MarvelLevel1/blob/main/entropy.png?raw=true)

Information Gain measures how much entropy decreases after splitting the dataset on a particular feature. It helps determine which feature is the best choice for splitting.

![](https://github.com/SevanthiBR/MarvelLevel1/blob/main/InformationGain.png?raw=true)

The ID3 algorithm is built using entropy and information gain.

* First, the total entropy of the dataset is calculated.
* Then, entropy after splitting on each feature is computed.
* Information Gain is calculated for each feature.
* The feature with the highest Information Gain is selected as the root node.
* The process is repeated recursively for each branch until all nodes are pure or no features remain.

This recursive splitting continues until stopping conditions are met.

**Steps taken:**
* Loaded and cleaned the dataset.
* Converted age into categorical ranges and performed train-test split.
* Implemented the ID3 algorithm from scratch.
* Evaluated model performance using classification metrics.
* Conducted fairness analysis across demographic groups.

[Link to my notebook](https://github.com/SevanthiBR/MarvelLevel1/blob/e55a20400be10d318f3404378428d9b415ec0e85/Task7.ipynb)

# Task 8: KNN with Ablation Study

## Aim

Build a K-Nearest Neighbors (KNN) classifier using the Breast Cancer Wisconsin dataset and conduct a feature ablation study to determine which features are most important for accurate classification
## Learnings

K-Nearest Neighbors (KNN) is a simple, distance-based machine learning algorithm used for classification and regression. It works by finding the *k* closest data points (neighbors) to a new input sample based on a distance metric like Euclidean distance. For classification, it assigns the class that is most common among those neighbors. For regression, it predicts the average value of the nearest neighbors. Since KNN relies on distance calculations, feature scaling is important to ensure all features contribute equally to the prediction.

I implemented a K-Nearest Neighbors (KNN) classifier on the Breast Cancer dataset to predict whether a tumor is malignant or benign. After preprocessing the data by removing unnecessary columns, encoding the target variable, and scaling the features, I trained a baseline KNN model and evaluated it using accuracy, precision, recall, and F1-score. To understand feature importance, I performed an ablation study by removing one feature at a time, retraining the model, and comparing the performance with the baseline. Features whose removal caused the largest drop in accuracy were considered most important for classification.

![](https://github.com/SevanthiBR/MarvelLevel1/blob/main/Task8img.png?raw=true)


[Link to my notebook](https://github.com/SevanthiBR/MarvelLevel1/blob/e6f444245c8091d0896e1020ce61f86ef93dc3f7/Task8.ipynb)


# Task 9: Evaluation Metrics - Pick the Best Performer!

## Aim

To evaluate five pretrained ML models using the Iris dataset, compare their performance and identify the best performing model

## Learnings

Joblib is a Python library used for efficient serialization and deserialization of Python objects. In machine learning, it is primarily used to save trained models as `.pkl` (pickle) files so they can be reused later without retraining. It allows developers to quickly load pre-trained models into memory and deploy them for predictions. Compared to the standard pickle module, Joblib is more efficient when handling large NumPy arrays, making it especially suitable for storing machine learning models that contain large datasets or parameters.


Accuracy, Precision, Recall, and F1-score are all calculated using values from the Confusion Matrix.
A Confusion Matrix is built using four key components:
* True Positive (TP)
* False Positive (FP)
* False Negative (FN)
* True Negative (TN)

 
 ![](https://github.com/SevanthiBR/MarvelLevel1/blob/main/Accuracy.png?raw=true)

 ![](https://github.com/SevanthiBR/MarvelLevel1/blob/main/F1Score.png?raw=true)

 ![](https://github.com/SevanthiBR/MarvelLevel1/blob/main/Precision.png?raw=true)

 ![](https://github.com/SevanthiBR/MarvelLevel1/blob/main/Recall.png?raw=true)

 These four values help measure how well a classification model performs.

 ![](https://github.com/SevanthiBR/MarvelLevel1/blob/main/Task9.png?raw=true)

[Link to my notebook](https://github.com/SevanthiBR/MarvelLevel1/blob/e55a20400be10d318f3404378428d9b415ec0e85/Task9.ipynb)





---



