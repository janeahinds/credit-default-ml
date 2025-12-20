## Credit Card Default Prediction

# Project Overview

This project builds an end-to-end machine learning pipeline to predict whether a customer will default on their credit card payment next month.

The goal is to explore the data, train baseline and improved models, and evaluate performance using appropriate classification metrics.

The dataset is based on the UCI Credit Card Default dataset and contains customer demographic information, credit limits, payment history, and bill amounts.

Primary objective:  
Minimize missed defaulters (false negatives), which carry higher business cost than false positives.  


# Data Loading & Initial Exploration  
Notebook: 01_eda.ipynb  
Goals:  
* Understand dataset structure and features
* Identify the target variable
* Check for obvious data quality issues
* 
Key Work:  
* Loaded and inspected the dataset
* Identified default.payment.next.month as the prediction target
* Reviewed feature types and distributions
* Removed non-informative ID column
* Performed initial sanity checks on missing and unusual value
* 
Outcome:  
A good understanding of the dataset and confirmation that the problem is a binary classification task.  

# Target Analysis & Business Framing
Notebook: 01_eda.ipynb  
Goals:  
* Understand class imbalance
* Frame the modeling task in business terms
  
Key Work:  
* Analyzed class distribution (~22% defaults)
* Identified class imbalance and implications for evaluation
* Determined that false negatives (missed defaults) are more costly than false positives
* Selected recall, precision, F1, and ROC-AUC as primary evaluation metrics
  
Outcome:  
Clear business framing that guides model evaluation and decision-making.  

# Baseline Logistic Regression Model
Notebook: 02_baseline_logistic_regression.ipynb  

Goals:  
* Establish a baseline
* Create a fair benchmark for comparison  
  
Key Work:  
* Performed stratified train/test split
* Built a pipeline with feature scaling and logistic regression
* Trained the baseline model
  
Evaluated using:  
* Confusion matrix
* Precision, recall, F1
* ROC-AUC
* Interpreted results with focus on minority (default) class
  
Outcome:  
A reliable baseline model that highlights the challenge of catching defaulters in an imbalanced dataset.

# Random Forest Model & Comparison
Notebook: 03_random_forest_baseline.ipynb  
Goals:  
* Test whether a non-linear model improves performance
* Compare models using consistent metrics
  
Key Work:  
* Trained a Random Forest model using the same train/test split
* Evaluated performance using the same metrics as logistic regression
* Compared confusion matrices and class-level metrics
* Analyzed tradeoffs between recall and precision

Outcome:  
The Random Forest model showed stronger performance in identifying defaulters while maintaining competitive ranking quality, making it a strong candidate for further decision optimization.

# Threshold Tuning & Final Model Decision
Notebook: 04_threshold_tuning.ipynb  
Goals:  
* Align model decisions with business cost
* Move beyond the default 0.5 probability threshold
  
Key Work:  
* Evaluated multiple probability thresholds (0.3, 0.4, 0.5)
* Measured precision and recall tradeoffs at each threshold
* Selected a threshold that reduces false negatives while keeping false positives acceptable
* Generated a final confusion matrix at the chosen threshold
* Documented the final model decision and rationale  

Outcome:  
A final decision rule was selected that improves defaulter detection while remaining aligned with business cost considerations.

# Final Takeaways:  
* Logistic regression provided a stable, interpretable baseline for credit default prediction.
* The Random Forest model improved recall for defaulters while maintaining strong overall ranking performance as measured by ROC-AUC.
* ROC-AUC was used to evaluate global ranking ability, while threshold tuning was used to control operational behavior.
* Threshold tuning proved essential for aligning model decisions with business priorities, particularly minimizing missed defaults.
* Final model selection was driven by business impact and error tradeoffs, not by reliance on a single metric.

Next Steps:
* Feature importance visualization
* Model monitoring and retraining strategy
