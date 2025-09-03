
Description:

This code performs a comprehensive analysis of white wine quality (Wine Quality Dataset) using statistical analysis, dimensionality reduction (PCA), Linear Discriminant Analysis (LDA), Neural Networks (MLPClassifier), and Gradient Boosted Trees. The following steps outline the workflow:


Workflow steps:

1. Libraries and Data Loading

Imports core Python libraries: pandas, numpy, matplotlib, seaborn, and machine learning tools from sklearn.
Reads the dataset from the Excel file winequality-white_without9and3.xlsx.
Defines column names for features and the target variable (quality_class).

2. Data Preparation

Categorizes wines into two classes:
Good wine (quality >= 6)
Bad wine (quality < 6)
Balances the number of samples for each class to avoid class imbalance.
Splits the data into training (75%) and testing (25%) sets for both classes.

3. Data Visualization

Creates histograms for each feature (vars) to compare the distribution between good and bad wines.
Computes correlation matrices for both good and bad wines and visualizes them using heatmaps.

4. Standardization and PCA

Standardizes all features to have mean 0 and standard deviation 1.
Performs PCA to reduce dimensionality:
Selects 7 principal components.
Computes explained variance and scree plots.
Applies inverse transformation for use in supervised models.

5. Supervised Machine Learning

Random Forest Classifier
Trains the model on the training set.
Evaluates accuracy on the test set.
Linear Discriminant Analysis (LDA)
Trains on PCA-transformed data.
Predicts classes and calculates accuracy.
Computes class probabilities (P_good, P_bad) and feature weights.
Neural Network (MLPClassifier)
Trains an MLP network using selected features.
Evaluates performance using ROC curves.
Compares different activation functions (relu, logistic, tanh) and plots loss curves.
Gradient Boosted Trees
Trains a GradientBoostingClassifier.
Computes 0-1 loss and prediction probabilities.

6. Model Evaluation

Computes confusion matrices and classification reports for LDA, Neural Network, and Boosted Trees.
Measures training time for each model.
Plots ROC curves for all models and compares results.
Plots probability distributions (Kolmogorov-Smirnov style) for all models.

7. Feature Importance

Extracts feature weights for LDA.
Estimates feature importance for Neural Network and Boosted Trees.
Visualizes feature importance using bar plots.


Conclusion:


The code provides a full pipeline for analyzing and predicting wine quality:

Statistical analysis and visualization of features.
Dimensionality reduction using PCA.
Training and evaluation of multiple supervised models.
Model comparison using accuracy, ROC-AUC, confusion matrices, and feature importance.
Enables understanding of which features most influence wine quality and provides insight into model performance across different algorithms.
