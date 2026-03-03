# Task 1

## Matlab OnRamp Course

The objective of this task was to gain hands-on experience with Machine Learning fundamentals using MATLAB. The course provided a guided and interactive introduction to practical ML workflows, helping understand how MATLAB handles the different stages of a machine learning project.

![Alt text](https://github.com/SevanthiBR/MarvelLevel1/blob/main/Task1Cert.png?raw=true)

# Task 2 

## Kaggle Crafter - Build your own dataset
The goal of this task was to create and publish a dataset on Kaggle. The dataset I created is titled “Fake Café Ratings Dataset”. It contains randomly generated information about fictional cafés, including their names, locations, average ratings, number of reviews, and other related details. The dataset was generated using Python’s Faker library, which helps create realistic-looking synthetic data. I used Faker to generate café names, locations, and customer reviews.


# Task 3

## Data Detox - Data Cleaning using pandas

This task was about cleaning a raw dataset using Pandas so it can be used for analysis or machine learning. I handled missing values, fixed spelling mistakes, corrected wrong numbers, validated emails, and converted date columns into proper format. Important missing data like Email was removed, while other missing values were filled in. Outliers were replaced with average values, and duplicate rows were deleted. In the end, the dataset became clean, consistent, and ready for further use. 


# Task 4

## Anomaly Detection

This task focuses on detecting anomalous user behavior by analyzing activity patterns using statistical techniques and machine learning methods. The aim is to identify users whose behavior significantly deviates from normal usage patterns.

StandardScaler was used to normalize the data. This ensures that no single feature dominates the anomaly detection process.The Interquartile Range (IQR) method was used to identify extreme values in each feature.To detect more complex and hidden patterns, an Isolation Forest model was used. To quantify how suspicious a user is, a risk score was calculated using Z-scores. Users were sorted based on their risk scores, and the top five most suspicious users were identified.



# Task 5

## Logistic Regression from scratch

I built Logistic Regression from scratch using NumPy. I created the sigmoid function, calculated the gradients manually, and updated the weights using gradient descent. After cleaning the data and scaling the features, I trained the model and evaluated it using accuracy, precision, recall, and F1-score. I then compared my results with scikit-learn’s Logistic Regression, and both gave almost the same results.

# Task 6

## Support Vector Machines
In this task, I implemented a Support Vector Machine (SVM) using scikit-learn on the Red Wine Quality dataset. I converted the quality scores into a binary classification problem and scaled the features since SVM is sensitive to feature magnitudes. After training the model, I evaluated its accuracy. To test robustness, I gradually added Gaussian noise to the feature data and retrained the model at each noise level. I observed that the model performed well with small noise but its accuracy decreased as noise increased. This experiment helped me understand how SVM works and how sensitive it is to noisy data.
# Task 7

## Fairness meets functionality
This project used a decision tree (ID3) model to predict hiring decisions based on candidate information such as age, gender, and experience. The dataset was cleaned, and age groups were created to support fairness analysis. The model achieved reasonable accuracy, but fairness evaluation showed differences in hiring rates between gender and age groups. Demographic parity and equal opportunity were calculated to measure bias. Results suggest that some groups were predicted to be hired more often than others. This may reflect historical bias in the dataset. To reduce bias, sensitive features could be removed or fairness-aware adjustments applied.

# Task 8

## KNN with ablation study


In this task, I implemented a K-Nearest Neighbors (KNN) classifier on the Breast Cancer dataset to predict whether a tumor is malignant or benign. After preprocessing the data by removing unnecessary columns, encoding the target variable, and scaling the features, I trained a baseline KNN model and evaluated it using accuracy, precision, recall, and F1-score. To understand feature importance, I performed an ablation study by removing one feature at a time, retraining the model, and comparing the performance with the baseline. Features whose removal caused the largest drop in accuracy were considered most important for classification.


# Task 9

## Evaluation metrics
The goal of this task was to compare five pretrained machine learning models to find the best one. The models tested were Decision Tree, Logistic Regression, K-Nearest Neighbors, Support Vector Machine, and Random Forest. Using the Iris dataset, I loaded the test data and evaluated each model. I measured Accuracy, Precision, Recall, and F1-score to compare their performance. After analyzing these results, I identified which model performed best. This task helped me understand how different algorithms work on the same data and why using multiple evaluation metrics gives a better understanding of model performance.



