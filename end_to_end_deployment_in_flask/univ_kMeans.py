# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 11:28:43 2024

@author: Swapnil Mishra
"""

import pandas as pd
import numpy as np
import sweetviz
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans
from sklearn import metrics
import joblib
import pickle


uni = pd.read_excel(r"C:\Users\Swapnil Mishra\Desktop\DS\Hierarchical Clustering 360 code\University_Clustering\University_Clustering.xlsx")

# Credenntials to connect to Database
from sqlalchemy import create_engine

# Define your connection parameters
user = 'postgres'
pw = 'swapnil1989'
host = 'localhost'
port = '5432'
db = 'univ_db'

# Create the connection string
engine = create_engine(f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}")

# to_sql() - function to push the dataframe onto a SQL table.
uni.to_sql('univ_tbl',con = engine , if_exists = 'replace' , chunksize = 1000,index = False )



# sql query to get the data from sql database
sql = 'select * from univ_tbl;'
df = pd.read_sql_query(sql,engine)

# Data Tyoes
df.info()

# EDA / Discriptive Statistics
# Data Preprocessing
# Cleaning unwanted columns
# We can safely ignore the ID columns by dropping the column.

df1 = df.drop(['UnivID','Univ'] , axis = 1 )
df1.head()
# Discriptive Statistics and Data Distribution Function

df1.describe()

# Check Unique Values for categorical data
df1.State.unique()
df1.State.unique().size
df1.State.value_counts()
## Automated Libraries
# Auto EDA

my_report = sweetviz.analyze([df1,'df1'])
my_report.show_html('Report.html')


# Missing Data \ Checking Null Values
df1.isnull().sum()
df1.info

# By using Mean imputation Null values can be imputed
numeric_features = df1.select_dtypes(exclude = ['object']).columns
numeric_features

# Non-numeric columns
categorical_features = df1.select_dtypes(include = ['object']).columns
categorical_features

# Define pipelineto deal with missing data and scaling numerical data
num_pipeline = Pipeline([('impute',SimpleImputer(strategy = 'mean' )),('scale',MinMaxScaler())])
num_pipeline

# Fit the numeric data to the piprline.Ignoring State Column
processed = num_pipeline.fit(df1[numeric_features])

# Save the pipeline
joblib.dump(processed,'processed1')

# Transform the data with pipeline on numeric columns to get clean data
univ_clean = pd.DataFrame(processed.transform(df1[numeric_features]), columns = numeric_features)
univ_clean

# Encoding Non Numeric fields
# Converting categorical data 'State' to numerical data using OneHotEncoding
#categ_pipeline = Pipeline([('OnehotEncode',OneHotEncoder(drop = 'first'))])
#categ_pipeline

# Using Column Transformer to transform the columns of an arrat or pandas dataframe
# This estimator allows different columns or column subsets of the input 
# To be transformed seperately and the features generated by each transformer 
# will be concatenated to form a single feature space 
#preprocess_pipeline = ColumnTransformer([('categorical',categ_pipeline,categorical_features),('numerical',num_pipeline,numeric_features)],
#                                        remainder = 'passthrough') # Skips the transformations for remaining
#preprocess_pipeline
# Pass the raw data throught the pipeline
#processed2 = preprocess_pipeline.fit(df1)

# Save the Imputation  and Encoding pipeline
# import joblib
#joblib.dump(processed2,'processed2')
# File gets saved under the current working directory
#import os
# os.getcwd()

# Clean the processed data for clustering
# univ = pd.DataFrame(processed2.transform(df1),columns = list(processed2.get_features))
# univ

# Clean data
univ_clean.describe()

# Clustering model Building
## K Means clustering
# Scree plot ot Elbow curve
TWSS = []
k = list(range(2,9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(univ_clean)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Creating a scree plot to find out No of Clusters
plt.plot(k,TWSS,'ro-')
plt.xlabel('No of Clusters')
plt.ylabel('total within sum of squares')


# Using KneeLocator
List = []

for k in range(2,9):
    kmeans = KMeans(n_clusters = k , init = 'random',max_iter = 30,random_state = 1,n_init = 10)
    kmeans.fit(univ_clean)
    List.append(kmeans.inertia_)
#pip install kneed   
from kneed import KneeLocator
kl = KneeLocator(range(2, 9), List, curve = 'convex') 
# The line is pretty linear hence Kneelocator is not able to detect the knee/elbow appropriately
kl.elbow
plt.style.use("seaborn")
plt.plot(range(2, 9), List)
plt.xticks(range(2, 9))
plt.ylabel("Interia")
plt.axvline(x = kl.elbow, color = 'r', label = 'axvline - full height', ls = '--')
plt.show()



# Detecting the best k = 3 using TWSS Value from scree plot
model = KMeans(n_clusters= 3)
yy = model.fit(univ_clean)
# We can see labels of clusters
model.labels_

# Cluster Evaluation
# Silhouette Coefficient
# 1 denotes best and 0 denotes overlapping clusters
from sklearn import metrics
metrics.silhouette_score(univ_clean,model.labels_)

# Calinski Harabasz:
# Higher value of CH index means cluster are well separated.
# There is no acceptable ot cut-off value defined.

metrics.calinski_harabasz_score(univ_clean, model.labels_)

# Davies-Bouldin Index:
# Unlike the previous two metrics, this score measures the similarity of 
# clusters. 
# The lower the score the better separation there is between your clusters.

metrics.davies_bouldin_score(univ_clean, model.labels_)

# Evaluation of Number of Clusters using Silhouette Coefficient Technique

from sklearn.metrics import silhouette_score

silhouette_coefficients = []

for k in range (2, 9):
    kmeans = KMeans(n_clusters = k ) # , init = "random", random_state = 1)
    kmeans.fit(univ_clean)
    score = silhouette_score(univ_clean, kmeans.labels_)
    k = k
    Sil_coff = score
    silhouette_coefficients.append([k, Sil_coff])
    
silhouette_coefficients

sorted(silhouette_coefficients , reverse = True,key = lambda x: x[1])

# Silhouette coefficients shows the no of clusters 'k' = 2
# Building KMeans clustering
bestmodel = KMeans(n_clusters = 2)
result = bestmodel.fit(univ_clean)

# save the KMeans clustering Model
# import pickle
pickle.dump(result,open('Clust_Univ.pkl', 'wb'))
import os
os.getcwd()

# Cluster labels
bestmodel.labels_

mb = pd.Series(bestmodel.labels_)
# Concat the results with the data
df_clust = pd.concat([mb,df.Univ,df1] , axis = 1)
df_clust = df_clust.rename(columns = {0:'cluster_id'})
df_clust.head()

# Aggegrate using the mean of each cluster
cluster_agg = df_clust.iloc[:,3:].groupby(df_clust.cluster_id).mean()
cluster_agg

# Save the results to a csv file
df_clust.to_csv('University.csv',encoding = 'utf-8',index = False)

import os
os.getcwd()