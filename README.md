# Individual project (big data course)
I chose to analyze **card fraud** data, focusing on **predicting the amount of fraudulent money based on personal characteristics, place of residence, and merchant category using PySpark**.

To accomplish this, I utilized PySpark along with machine learning models such as linear regression, decision tree regressor, and GBT regressor. 

Two datasets were provided: one for training and one for testing. The variables used in this analysis included sex, merchant category, city, city population, age (a new variable), occupation, and amount (amt). In this analysis, the dependent variable **y** is the amount (amt) of money, while the other variables serve as independent variables **x**. 

Before starting the analysis and implementing the machine learning models, I performed some data preparation, which included:
- Adding a new column for age, calculated as the difference between the transaction date (trans_date_trans_time) and the date of birth (dob).
- Filtering the dataset to include only the rows where fraud occurred (is_fraud = 1).
```python
df_with_age=df.withColumn("age", round(months_between(col('trans_date_trans_time'), col("dob")) / lit(12), 0))\
    .filter(col('is_fraud')==1)
```
- Encoding the string column for use in machine learning models.
```python
def encoding_string(df, string_columns, output_columns):
    indexers = [StringIndexer(inputCol=col, outputCol=out) 
                for col, out in zip(string_columns, output_columns)]
    pipeline = Pipeline(stages=indexers)
    model = pipeline.fit(df)
    indexed = model.transform(df)
    
    return indexed

indexed = encoding_string(
    df_train_fraud,
    ['category', 'gender', 'city', 'job'],
    ['category_idx', 'gender_idx', 'city_idx', 'job_idx']
)
```
- Combine the feature columns into a single column named 'features' while retaining the dependent variable.
```python
def making_features(df, input_columns, y):
    # Assemble features into a single vector
    assembler = VectorAssembler(inputCols=input_columns, outputCol='features')
    feature_data = assembler.transform(df)
    
    # Select the features and target variable
    final_data = feature_data.select('features', y)
    return final_data
```

For linear regression, all variables were used; however, for the decision tree and GBT, only the variables of sex, merchant category, city population, and age (a new variable) were used due to the requirement that categorical variables should not exceed 32 distinct values.
## For each machine learning model, I created a parameter grid with cross-validation (using 3 folds) to identify the best combination of parameters for the data (this was performed on the training dataset).

For linear regression, the hyperparameters included:
- Regularization parameter
- Option to fit an intercept term
- The ElasticNet mixing parameter, which ranges from 0 to 1. When alpha = 0, the penalty is an L2 penalty; when alpha = 1, it is an L1 penalty.
- Shape parameter to control the amount of robustness
- Option to standardize the training features before fitting the model.
  
```python
  lr = LinearRegression(labelCol='amt', featuresCol='features')
  paramGrid_lr = ParamGridBuilder() \
  .addGrid(lr.regParam, [0.1, 0.01]) \
  .addGrid(lr.fitIntercept, [True, False]) \
  .addGrid(lr.elasticNetParam, [0, 0.1,  0.3, 0.5,  0.7, 0.9,1])\
  .addGrid(lr.epsilon , [1.5, 2, 5, 10])\
  .addGrid(lr.standardization, [True, False])\
  .build()
 ```

For decision tree regression hyperparameters were:
  - Maximum depth of the tree
  - Minimum number of instances each child must have after split.
```python
dtr = DecisionTreeRegressor(labelCol='amt', featuresCol='features')

paramGrid_dtr = ParamGridBuilder() \
.addGrid(dtr.maxDepth, [0, 1, 2, 3, 4, 5, 6,7, 8, 9, 10, 15, 20, 25, 30]) \
.addGrid(dtr.minInstancesPerNode, [1, 5, 10, 20, 50])\
.build()
``` 
  For GBT regression hyperparameters were:
  - Maximum depth of the tree
  - Minimum number of instances each child must have after split
  - Step size (a.k.a. learning rate) in interval (0, 1] for shrinking the contribution of each estimator.
```python
gbt = GBTRegressor(labelCol='amt', featuresCol='features')

paramGrid_gbt = ParamGridBuilder() \
.addGrid(gbt.maxDepth, [1,  5, 8,  15,  25]) \
.addGrid(gbt.minInstancesPerNode, [1,  10, 20])\
.addGrid(gbt.stepSize, [ 0.1,   0.5,   0.8])\
.build()
```
The evaluator used RMSE for cross-validation, selecting the hyperparameter combination that minimized RMSE.
```python
evaluator = RegressionEvaluator(
    labelCol="amt", predictionCol="prediction", metricName="rmse")

def perform_cv(estimator, paramgrid, evaluator, df):
    cross_val = CrossValidator(estimator=estimator,
    estimatorParamMaps=paramgrid,
    evaluator=evaluator,
    numFolds=3)
    cv_model = cross_val.fit(df)
    return cross_val, cv_model
```
## The best hypermater combination for each ML
For Linear regression with RMSE 342.522195:
- regularization parameter 0.1
- whether to fit an intercept term - True
- the ElasticNet mixing parameter - 0.0 (L2 penalty)
- The shape parameter - 1.5
- Standartization - True

For decision tree regression hyperparameters were with RMSE 390.553729: 
  - Maximum depth of the tree - 0
  - Minimum number of instances each child must have after split - 1
    
For GBT regression hyperparameters were with RMSE 95.958362:
  - Maximum depth of the tree - 0 
  - Minimum number of instances each child must have after split - 1
  - Step size - 0.1

## Prediction results for each machine learning model using the optimal hyperparameters on the training dataset

<img width="500" alt="image" src="https://github.com/Ugneona/Individual_project_big_data/blob/main/lr_train.png?raw=true" />
<img width="500" alt="image" src="https://github.com/Ugneona/Individual_project_big_data/blob/main/dtr_train.png?raw=true" />
<img width="500" alt="image" src="https://github.com/Ugneona/Individual_project_big_data/blob/main/gbt_train.png?raw=true" />

For the training dataset, the best models are the decision tree and GBT, as they show lower RMSE values.

## Prediction results for each machine learning model using the optimal hyperparameters on the testing dataset

<img width="500" alt="image" src="https://github.com/Ugneona/Individual_project_big_data/blob/main/lr_test.png?raw=true" />
<img width="500" alt="image" src="https://github.com/Ugneona/Individual_project_big_data/blob/main/dtr_test.png?raw=true" />
<img width="500" alt="image" src="https://github.com/Ugneona/Individual_project_big_data/blob/main/gbt_test.png?raw=true" />

For the test dataset, the best model is linear regression, with an RMSE of 311.501 and an R² of 0.371.

## Conclusions
The applied models did not achieve the best results in predicting the amount of fraudulent money. The best model was a linear regressor with standardization, a fitted intercept, L2 penalty, a shape parameter of 1.5, and a regularization parameter of 0.1, resulting in an RMSE of 311.501 and an R² value of 0.371. 

To improve the models, I suggest the following:  
- Standardize the features before applying the model.  
- Group job variables into broader categories for use in the tree models, and replace the city variable with state variables.  
- Examine the cross-validation results presented in the graphs to determine where the models may be overfitting; this occurs when they adapt too closely to the training data and perform poorly on new data. Consider creating line graphs to visualize RMSE and R² values, and identify the optimal combination before the overfitting stage.

