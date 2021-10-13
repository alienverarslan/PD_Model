# PD Model Machine Learning Case Study Document

## Ali Enver Arslan

In the case study, a Probability of Default prediction task is given. In this document, we aim to describe the steps taken and explain the model building details. 
The project consists of two sections:

1-	Exploratory Data Analysis (EDA)

2-	Model Development

### 1.	Exploratory Data Analysis (EDA)

In the EDA section, 

• The data to be predicted is excluded from the EDA section to isolate the EDA from the unseen data 

• To have an idea about the data and the features basic summary statistics are calculated

•	To check if there are outliers in the data, an outlier check is done via box plots. It is seen that many features might have outliers. This led us to further treatment of the outliers

•	In the next step, a missing value check is implemented. Some of the features have common missing patterns as can be seen from the graph in the python file, as well. Due to the similar missingness patterns and existence of features with high missing rates we analyzed to detect if the missing values are completely random, or there is a reason for the missingness. 

•	By grouping each feature over missing and non-missing values, all the remaining features are divided into two groups and checked whether there is a significant relationship between missing values and other variables using the t-test.

•	The results of the t-tests indicate that missing values significantly differ from the non-missing values. It makes removing missing values column or row-base unhealthy. They should be kept within the data and we should treat them differently.

•	For the missing imputation task, features are split into 3 categories as numeric (continuous), ordinal and categorical. The details of the splitting can be found in the python file. To explain the ordinal feature selection, we inferred those features about the account status are following an ordinal structure. As a result, we categorized them as ordinal features to treat them differently from the rest of the features.

•	Since we deducted that missingness maybe because of the nature of the features, it is not a good idea to use a mean/median imputation or imputation via regression for continuous features. Instead, we will impute them with an extraordinary value and create a flag for missingness to see the effect of being missing. A similar procedure is applied for categorical features. For the ordinal features, it is inferred that the missing values can be imputed with zero since there is no 0 when the feature has missing values but has 0 when there are no missing values.

•	In the correlation analysis, correlation within the feature groups and with the target are conducted separately. Correlation between feature groups is left out of the scope of the project. For each correlation, an appropriate method is applied and bivariate correlation analysis is conducted for all features.

•	As a result of the correlation analysis, one of the features that have >= 50% correlation is eliminated. The feature which has a stronger correlation with the target is kept. The purpose of the feature elimination via correlation analysis is both dimension reduction and meeting the assumptions of the base model (Logistic Regression).



### 2.	 Model Development

•	Model development data is split into train and validation sets with the proportions 75% and 25% respectively. Since the dataset is highly imbalanced, this step is applied with Stratified K-Fold to keep the target distribution the same in each split.

•	Since the imputation method we use does not contain any risks regarding data leakage, we implemented the imputation right after the data split.

•	In the model development section, Logistic Regression is selected as the baseline model due to its nature. 

•	For the model performance metric, there are a couple of alternatives depending on the priorities. Since the data is highly imbalanced, precision, recall, and F1 scores are the most commonly used success metrics. On the other hand, AUC is also used for the overall model performance regardless of the prediction accuracy. By acquiring a high AUC and optimized class probability threshold, one can develop a high-performance model. Here in this study, we control all those metrics we mentioned. However, AUC allows us to evaluate our model across a range of risk tolerances and provide a general model that we can tune for the business strategy.

•	For the baseline score, we used Logistic Regression with its default parameters in Scikit-Learn. Data preprocessing is implemented in a pipeline and continuous, categorical, and ordinal features are treated separately according to their needs. One preprocessing is tried for each feature group in the baseline score calculation except the ordinal features. Ordinal features are included in the baseline model without a preprocessing step. Since logistic regression is affected by outliers, we scaled the continuous variables. However, ordinal feature scaling can also be tested. Categorical features are encoded with the One Hot Encoding method as a classical approach.

•	For the baseline model without any feature elimination, the AUC score of the model is calculated as 0.891. However, the model mostly can predict the True Negatives so far. 

•	As an alternative to our baseline predictor, we then built a model only using the feature set that we selected in the correlation elimination step previously in model2. AUC score falls to 0.877. 

•	As another alternative to the baseline, in model3, we, then, exclude the categorical features from the model since they have high cardinality, and excluding them may improve model performance. This time AUC increases to 0.883. it is better than feature elimination but still worse than our baseline.

•	In the model4, we used the same parameters as the model3. The only difference was to see how far can we go on predicting the minority class true. To do this, we changed the threshold for probabilities by using the ROC Curve.

•	So far, we have decided that the best alternative is the baseline with all features. In the model5, we, now, select the best preprocessing steps that improve model performance. We used several alternative preprocessing methods for each feature group and used the same Logistic Regression with its default parameters. With the preprocessing method optimization, we reach 0.893.

•	In the model6, we compare multiple algorithms with the preprocessing methods that we chose in the model5. XG Boost gives the best AUC although it is quite unsuccessful in the other metrics. On the other hand, our baseline model Logistic Regression has one of the highest AUC scores as well. Although AdaBoost also has one of the top AUC scores, we do not want to continue with two boosting algorithms for further improvements. As a result, XG Boost and Logistic Regression are the two potential algorithms.

•	In the model7, we optimized the preprocessing steps for XG Boost. XG Boost performance is 0.907. However, it is not quite successful at predicting the minority class. On the other hand, we already optimized preprocessing steps for Logistic regression and we will go to the parameter tuning directly for logistic regression.

•	In the model8, we tune the hyperparameters of the XG Boost Classifier with the preprocessing steps that we chose in the model7. The AUC is calculated as 0.911. however, there is a significant increase in the recall and TP increases a lot.

•	In the model9, we tune the hyperparameters of Logistic Regression with the preprocessing steps that we chose in the model5. The AUC is calculated as 0.893.

•	In model10, we tried an oversampling method SMOTE but it did not perform well. This is because SMOTE is better when we have low-dimensional data. However, our final dataset contains 70 features which is not relatively a small number. 

•	Overall, XG Boost gives the best AUC score after parameter tuning. However, it still performs poorly on predicting the positive class.

•	Focusing on another metric, trying out neural networks, and extensive parameter tuning for all the models could be further improvements. 

•	We also did not consider feature generation like finding the intersection terms, trying polynomial terms, or other feature engineering methods in this study. Calibration of the probabilities could be applied for an improvement in the model’s discriminatory power, as well.

•	As a result, our final model has a high AUC which makes it a successful model, but low accuracy on predicting the true defaults. However, it is very strong in predicting non-defaults. With the trust on high AUC, we can change and find the optimum decision threshold for the probabilities and improve performance as well.
