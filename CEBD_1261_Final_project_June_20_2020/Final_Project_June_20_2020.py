# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook will show you one of the many ways to import a data set into Databricks,  interact with that data and run a ML model on it. When  I say many ways, the beauty of Databricks is you can manipulate the data in a language of your choice and switch on the fly. This notebook assumes that you have a file already inside of DataBricks FileSystem that you would like to read from.
# MAGIC 
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.
# MAGIC 
# MAGIC A little about the dataset: The data represents 303 participants, each row representing one participant and 13 observations and 1 label which is binary in  nature. Since this is a labeled data with an outcome of a category, we deem it supervised classification problem. The data was clean, didnt need any preprocessing nor any encoding. All values were numerical. 

# COMMAND ----------

# MAGIC %md
# MAGIC > #### Column description 
# MAGIC 
# MAGIC >age: The person's age
# MAGIC 
# MAGIC >sex : 1 = male, 0 = female.
# MAGIC 
# MAGIC >cp : the chest pain experience 
# MAGIC 
# MAGIC >trestbps: The person's resting blood pressure
# MAGIC 
# MAGIC >chol : the person's cholesterol measurement in mg/dl
# MAGIC 
# MAGIC >fbs: the person's fasting blood sugar (>120mg/dl, 1 = true, 0 =false)
# MAGIC 
# MAGIC >restecg : resting resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probale or define left ventricular hypertrophy by Estes' criteria)
# MAGIC 
# MAGIC >thalach : the person's maximum heart rate achieved
# MAGIC 
# MAGIC >exag: Exercise induced angina ( 1= yes, 0 = no)
# MAGIC 
# MAGIC >oldpeak: ST depression induced by exercise relative to rest ('ST' related to positions on the ECG plot)
# MAGIC 
# MAGIC >slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3:downsloping)
# MAGIC 
# MAGIC >ca: The number of major vessels (0-3)
# MAGIC 
# MAGIC >thal: A blood disorder called thalassemia (3= normal, 6= fixed, 7 =reversable defeat)
# MAGIC 
# MAGIC >##### target: Heart disease (0=no, 1= yes), the target actually stipulates inflamation of the coronary arteries which lead to heart disease.

# COMMAND ----------

# MAGIC %md ### What are the factors that contribute to heart disease. In particular inflamation of the Coronary artery. 

# COMMAND ----------

# MAGIC %md #### Import mlflow libraries, can be done in cluster aswell for all notebooks that feed off that cluster which need those libraries.

# COMMAND ----------

dbutils.library.installPyPI("mlflow")
dbutils.library.restartPython()
import mlflow


# COMMAND ----------

# MAGIC %md ##### Importing dataset and transforming it into a spark data frame

# COMMAND ----------

# File location and type
file_location = "dbfs:/FileStore/tables/heart_disease.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# MAGIC %md ##### Setup up a temporary or permenent view for queries with SQL.

# COMMAND ----------

# MAGIC %md Temporary view (only accessable to this notebook)

# COMMAND ----------

# Create a view or table

temp_table_name = "HeartDisease_csv"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC 
# MAGIC select * from `HeartDisease_csv`

# COMMAND ----------

# MAGIC %md Permenant view accessable outside the scope of this notebook via a write to a parquet file

# COMMAND ----------

# permanent_table_name = "heartdisease_csv"
# df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

# MAGIC %md Dataset length of records

# COMMAND ----------

print("Our dataset has %d rows." % df.count())

# COMMAND ----------

# MAGIC %md ##### Transform spark data frame into a pandas dataframe to run some EDA 

# COMMAND ----------

import pandas as pd
dt = df.toPandas()

# COMMAND ----------

# MAGIC %md #### Extract DataFrame correlations

# COMMAND ----------

dt.corr()

# COMMAND ----------

# MAGIC %md #### Visualize correlation matrix using seaborn

# COMMAND ----------

# MAGIC %md High correlation between thal (a blood disorder called thalassemia), oldpeak (ElectroCardioGraphy ECG/EKG), thalach (participants maximum heart rate archieved) and the target (disease or not diseased)

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,10))
corr = dt.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(50, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

# COMMAND ----------

# MAGIC %md ##### Some stats on the features average, minimum and maximum of factors

# COMMAND ----------

display(df.describe())

# COMMAND ----------

# MAGIC %md ##### Stats on the label (target), especially if this is an imbalanced dataset.

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select target from `HeartDisease_csv` count

# COMMAND ----------

# MAGIC %md ##### Split the dataset randomly into 70% for training and 30% for testing. 

# COMMAND ----------

train, test = df.randomSplit([0.7, 0.3], seed = 0)
(train.count(), test.count())
print("We have %d training examples and %d test examples." % (train.count(), test.count()))



# COMMAND ----------

# MAGIC %md Cache the data for speed (data set is really small so this is really not needed)

# COMMAND ----------

train.cache()
test.cache()


# COMMAND ----------

display(train)

# COMMAND ----------

display(test)

# COMMAND ----------

# MAGIC %md ### Data visualization

# COMMAND ----------

# MAGIC %md ##### Older people are more likely than younger people to suffer from Heart disease.

# COMMAND ----------

display(train.select("target", "age"))

# COMMAND ----------

# MAGIC %md ##### No real correllation by sex

# COMMAND ----------

display(train.select("sex", "target"))
#sex: The person's sex (1 = male, 0 = female)

# COMMAND ----------

# MAGIC %md ##### High blood pressure is a high risk factor for heart condition

# COMMAND ----------

display(train.select("target", "trestbps"))

# COMMAND ----------

# MAGIC %md ##### Going higher than your maximum heart rate for long periods of time could be a risk factor for heart condition

# COMMAND ----------

display(train.select("target", "thalach"))

# COMMAND ----------

# MAGIC %md ##### When there is high cholesterol in your blood, it clogs the coronary arteries in the form of cholesterol plaque beoming a high risk factor

# COMMAND ----------

display(train.select("target", "chol"))

# COMMAND ----------

# MAGIC %md ##### Deeper and more widespread ST depression (ECG/EKG) generally indicates more severe or extensive disease.

# COMMAND ----------

display(train.select("target","oldpeak"))

# COMMAND ----------

# MAGIC %md ##### Thalassemia (thal) is a blood disorder that causes heart disease

# COMMAND ----------

display(train.select("target","thal"))

# COMMAND ----------

# MAGIC %md #### Separate features from label 

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, VectorIndexer
featuresCols = df.columns
featuresCols.remove('target')
# This concatenates all feature columns into a single feature vector in a new column "rawFeatures".
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures")
# This identifies categorical features and indexes them.
vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=4)

# COMMAND ----------

# MAGIC %md Gradient boosting Tree

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(labelCol="target", featuresCol="features")

# COMMAND ----------

# MAGIC %md Decision Tree

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol="target", featuresCol="features", maxDepth=3)

# COMMAND ----------

# MAGIC %md #### Cross validation, Grid Search and Evaluators(AUC ROC)

# COMMAND ----------

# MAGIC %md Decision Tree

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC", labelCol=dt.getLabelCol())
cv = CrossValidator(estimator=dt, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=5)
paramGrid = ParamGridBuilder()\
  .addGrid(dt.maxDepth, [2])\
  .build()

# COMMAND ----------

# MAGIC %md Gradient Boosting Tree

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

paramGrid = ParamGridBuilder()\
  .addGrid(gbt.maxDepth, [2])\
  .addGrid(gbt.maxIter, [10])\
  .build()
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC", labelCol=dt.getLabelCol())
cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=5)


# COMMAND ----------

# MAGIC %md #### Stages and Pipleline

# COMMAND ----------

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, cv])


# COMMAND ----------

# MAGIC %md #### Train model

# COMMAND ----------

pipelineModel = pipeline.fit(train)

# COMMAND ----------

# MAGIC %md #### Predictions

# COMMAND ----------

predictions = pipelineModel.transform(test)

# COMMAND ----------

display(predictions.select("target", "prediction", *featuresCols))

# COMMAND ----------

predictions.columns

# COMMAND ----------

selected = predictions.select('target', "prediction", "probability", "age", "chol", "thalach", "thal", "oldpeak", 'trestbps')
display(selected)



# COMMAND ----------

# MAGIC %md #### Evaluation metrics : AUC ROC

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

# Evaluate model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol= 'target')
BinaryClassificationEvaluator()

# COMMAND ----------

evaluator.evaluate(predictions)

# COMMAND ----------

# MAGIC %md ###Training and Making Predictions from sklearn

# COMMAND ----------

# MAGIC %md ### The x contains all the columsn from the dataset except the 'target' Columns. The y varaible contains the value from the 'target' Columns

# COMMAND ----------

# Separating features(X) and target(y)
X = dt.drop('target', axis=1)

# COMMAND ----------

print(X)

# COMMAND ----------

y = dt['target']
print(y)

# COMMAND ----------

# MAGIC %md ### We use split up 35% of the data in to the test set and 65% for training

# COMMAND ----------

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

# COMMAND ----------

print(f"X.shape: {X.shape}, y.shape: {y.shape}")
#original dataset

# COMMAND ----------

print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")
#splited datasets

# COMMAND ----------

# MAGIC %md ### Use the DecisionTreeClassifier to train the algorithm 

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# COMMAND ----------

y_pred = classifier.predict(X_test)
print(y_pred)

# COMMAND ----------

# MAGIC %md ### Evaluating the Algorithm from sklearn

# COMMAND ----------

# MAGIC %md ### From the confusion matrix, our alogrithm misclassified only 24 out of 107. This is 77% accuracy

# COMMAND ----------

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# COMMAND ----------

# MAGIC %md ### Conclusion: It shows that people with heart disease tend to be older, and have higher blood pressure, higher cholesterol levels, deeper and more widespread ST etc., than people without the disease.
