# Multiple Regression Model  
1.Data Loading & Preprocessing 
The initial cells involve loading and inspecting the dataset, including showing the first few 
rows with d3.head() and setting display options such as showing all columns with 
pd.set_option('display.max_columns', None). 
2. Data Preprocessing 
* Setting decimal precision: 
o The code pd.set_option('display.float_format', lambda x: '%.2f' 
% x) ensures that floats are shown with two decimal places. 
* Feature Selection and Labeling: The notebook selects a series of columns relevant 
for the regression analysis, including demographic and academic features such as: 
o studytime_y, failures_y, schoolsup_y, famsup_y, health_y, absences_y, 
and the dependent variable likely linked to final grades (G3_y). 
3. Model Preparation 
* Library Import: The required libraries from the sklearn package are imported, such 
as: 
o LinearRegression from linear_model for applying linear regression, 
o mean_squared_error and mean_absolute_error for evaluation metrics, 
o train_test_split for splitting the data, and 
o cross_val_score for cross-validation. 
4. Model Building and Training 
* Train-Test Split: The data is split into training and testing sets. This helps in 
evaluating the model's performance on unseen data. 
* Regression Model Creation: 
reg_model = LinearRegression() 
reg_model.fit(X_train, y_train) 
The regression model is trained on the training data X_train and y_train. 
5. Model Evaluation 
* RMSE Calculation (Root Mean Squared Error): 
np.sqrt(mean_squared_error(y_test, y_pred)) 
After predicting the test set (y_pred = reg_model.predict(X_test)), RMSE is 
calculated to measure how well the model fits the data. Lower RMSE indicates a 
better fit. 
* Model Accuracy: 
reg_model.score(X_test, y_test) 
This score is an R² value, showing how much of the variance in the dependent variable 
(the target) is explained by the independent variables (features). In this case, the R² 
value is approximately 76%. 
* Cross-Validation: 
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, 
scoring="neg_mean_squared_error"))) 
This applies cross-validation (10-fold) to ensure the model's performance is not overly 
dependent on any single train-test split. It calculates the average RMSE across 10 
different splits, giving a more robust evaluation.
