
# coding: utf-8

# In[1]:


# Import Libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


# In[3]:


# Get Sample Data
# This Sample data is already undersampled, as the number of fraud cases was very less (1%)
# The proportion of fraud versus non-fraud is 50%.
df = pd.read_csv("https://raw.githubusercontent.com/veeranalytics/Fraudulent-Transactions-Model/master/fraud_full_sample.csv")


# In[4]:


# Take look at the data
df.head()


# In[5]:


# Check for missing values
df.isnull().sum() ## Good News -- there are no missing values


# In[6]:


# Creating datasets for Dependent (y) and Independent Variables (X)
X = df.iloc[:, 1:8].values
y = df.iloc[:, 0].values


# In[7]:


# Set seed for reproducibility
np.random.seed(123)


# In[8]:


# Encoding categorical data
# Encoding the Independent Variable
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# In[9]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[10]:


# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[11]:


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 8))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# In[ ]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# In[ ]:


# Model Evaluation Metrics
# Making the Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred)

# Get metrics from confusion metrics
#[row, column]
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# Calculate model evaluation metrics
prediction_accuracy = (TP + TN) / float(TP + TN + FP + FN) # metrics.accuracy_score(y_test, y_pred)
miss_classfication_rate = 1 - prediction_accuracy # 1- accuracy_score
sensitivity = TP / float(FN + TP) # Also, kmown as Recall Score-- metrics.recall_score(y_test, y_pred)
specificity = TN / (TN + FP)
false_positive_rate = FP / float(TN + FP) # 1 - specificity
precision = TP / float(TP + FP) # metrics.precision_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
# print("prediction_accuracy_in_percentage: {0:.2f}%".format(prediction_accuracy*100))
# print("miss_classfication_rate_in_percentage: {0:.2f}%".format(miss_classfication_rate*100))
print("prediction_accuracy: %f" % prediction_accuracy)
print("miss_classfication_rate: %f" % miss_classfication_rate)
print("sensitivity: %f" % sensitivity)
print("specificity: %f" % specificity)
print("false_positive_rate: %f" % false_positive_rate)
print("precision: %f" % precision)
print("recall: %f" % sensitivity)

