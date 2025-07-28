Credit Card Fraud Detection - Machine Learning Project


This project is a comprehensive hands-on machine learning pipeline to detect fraudulent credit card transactions. It includes data exploration, feature engineering, model building, evaluation, cross-validation, and class imbalance handling techniques.
🔢 Project Goal
Detect fraudulent credit card transactions using supervised machine learning, helping financial institutions proactively prevent fraud.
________________________________________
🧬 Skills and Tools Used
•	Language: Python
•	Environment: Google Colab (Jupyter Notebook)
•	Libraries:
o	Data Analysis: NumPy, Pandas
o	Visualization: Matplotlib, Seaborn
o	Modeling: Scikit-learn, XGBoost
o	Evaluation: SciPy, Statsmodels
________________________________________
📅 Dataset Overview
•	Source: European cardholder transaction dataset from September 2013
•	Size: 284,807 transactions
•	Fraudulent: 492 transactions (~0.172%)
•	Features:
o	Time (seconds elapsed since first transaction)
o	Amount
o	V1 to V28: Principal components from PCA (feature anonymization)
o	Class: Target variable (1 = Fraud, 0 = Legitimate)
________________________________________
💡 Project Steps
1. Data Understanding & EDA
•	Load and inspect the data: .head(), .shape(), .info(), .describe()
•	Analyze class imbalance using bar/pie plots
•	Compute and visualize feature correlations via heatmaps
2. Feature Engineering
•	Transform Time into time_day, time_hour, time_minute
•	Drop irrelevant columns, retain time_hour
3. Train-Test Split
•	80% Training / 20% Testing using train_test_split
4. Visualization of Feature Distributions
•	Plots (histograms) show skewness and class-based variation
________________________________________
📊 Evaluation Metrics
•	Confusion Matrix: TP, FP, TN, FN breakdown
•	Classification Report: Precision, Recall, F1-score
•	AUC-ROC: Area under curve for model discrimination performance
________________________________________
📈 Model Building & Evaluation
Reusable functions were built for training and evaluating each model type:
Models Used
•	Logistic Regression (L1 & L2 Regularization)
•	K-Nearest Neighbors (KNN)
•	Decision Trees (gini, entropy)
•	Random Forest
•	Support Vector Machine (SVM) (sigmoid kernel)
•	XGBoost
Each model is evaluated using: - Accuracy - ROC AUC - Optimal threshold (from ROC curve) - Confusion matrix and ROC plot
All results are stored in a unified df_results DataFrame.
________________________________________
📊 Methodologies Applied
1. Repeated K-Fold Cross-Validation
•	Ensures robustness through multiple splits
•	Logistic Regression with L2 performs best
2. Stratified K-Fold Cross-Validation
•	Maintains class balance across folds
•	Logistic Regression with L2 again outperforms others
3. Hyperparameter Tuning
•	Best Base Model: Logistic Regression with L2 (Stratified K-Fold)
•	Tuned parameters: C, solver, tolerance
•	Feature importances (coefficients) are plotted
________________________________________
🎯 Handling Class Imbalance
Applied oversampling techniques with Stratified K-Fold:
Techniques:
•	Random Over Sampler
•	SMOTE (Synthetic Minority Over-sampling)
•	ADASYN (Adaptive Synthetic Sampling)
Results:
•	XGBoost performed best for all oversampling methods
________________________________________
⚖️ Final Hyperparameter Tuning
•	Model: XGBoost + Random Over Sampling + Stratified K-Fold
•	Tuned using RandomizedSearchCV
•	Parameters tuned: max_depth, min_child_weight, n_estimators, learning_rate
•	Feature importance plotted from final model
________________________________________
🌟 Conclusion
•	Best Base Model: Logistic Regression with L2 (Stratified K-Fold)
•	Best Oversampled Model: XGBoost with Random Over Sampler (Stratified K-Fold)
This project showcases a complete ML workflow from EDA to advanced modeling and tuning, offering a powerful template for fraud detection problems.
________________________________________
📂 Files Included
•	credit_card_fraud_detection.ipynb: Full notebook with code and explanations
•	creditcard.csv: Dataset (should be uploaded separately)
•	requirements.txt: List of required packages
________________________________________
📍 How to Run
1.	Clone the repo or upload the notebook to Google Colab
2.	Upload the dataset
3.	Run cells step-by-step to execute the pipeline
________________________________________
✈️ Future Improvements
•	Try neural networks or autoencoders
•	Apply cost-sensitive learning
•	Deploy as a REST API using FastAPI or Flask
________________________________________
🌐 Acknowledgments
Dataset provided by Kaggle - Credit Card Fraud Detection
________________________________________
📗 References
•	Scikit-learn Documentation
•	XGBoost Documentation
•	Imbalanced-learn (SMOTE, ADASYN)
•	Seaborn & Matplotlib Docs
________________________________________
“Even with highly imbalanced data, smart preprocessing and model tuning can yield strong results.”
________________________________________
Author: Maniska Biswas
linkedIn: www.linkedin.com/in/maniska-biswas
