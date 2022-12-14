# Breast Cancer Diagnosis

In this project, we develop a model for predicting if breast lumps are benign or malignant.

This project is done as a part of [Project of the Week at DataTalks.Club](https://github.com/DataTalksClub/project-of-the-week/blob/main/2022-08-14-frontend.md)

Dataset: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

## About Dataset

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

This database is also available through the UW CS ftp server:
ftp ftp.cs.wisc.edu
cd math-prog/cpo-dataset/machine-learn/WDBC/

Also can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

Attribute Information:

1) ID number
2) Diagnosis (M = malignant, B = benign)
3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features. For instance, field 3 is Mean Radius, field
13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

Missing attribute values: none

Class distribution: 357 benign, 212 malignant

## Result

Following results were obtained on test data by fitting three models: logistic regression, kNN and Random Forest

| values   | Model         | Precision | Recall | f1-score|
|----------|---------------|-----------|--------|---------|
|  0       | Logistic      |   1.0     |  .91   |  .95    |
|  1       | Logistic      |    .83    | 1.0    |  .91    |
|  0       | kNN           |    .96    |  .96   |  .96    |
|  1       | kNN           |    .91    |  .91   |  .91    |
|  0       | Random Forest |    .99    |  .95   |  .97    |
|  1       | Random Forest |    .89    |  .97   |  .93    |


Performance of **Random Forest (f1 = 0.93)** is better than other models: logistic (f1 = .91) and kNN (f1 = .91) on test data.

So we use random forest model with best hyperparameters for deployment.

### Deployment
The model is deployed in [**heroku**](https://github.com/aryalsohan0/BreastCancerDiagnosis/deployments/activity_log?environment=breast-cancer-diagnosis1).
We can enter values of variables as comma separated and predict the type of breast lump as benign or malignant.