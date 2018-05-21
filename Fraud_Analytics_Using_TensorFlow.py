
# coding: utf-8

# In[2]:


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


# In[20]:


# Take look at the data
df.head()


# In[24]:


# Check for missing values
df.isnull().sum() ## Good News -- there are no missing values


# In[27]:


# Creating datasets for Dependent (y) and Independent Variables (X)
X = df.iloc[:, 1:8].values
y = df.iloc[:, 0].values


# In[ ]:


# Set seed for reproducibility
np.random.seed(123)


# In[33]:


# Encoding categorical data
# Encoding the Independent Variable
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# In[36]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[38]:


# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[39]:


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


# In[50]:


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


# In[83]:


# Model architecture parameters
n_var = 8
n_neurons_1 = 6
n_neurons_2 = 6
n_target = 1
epochs = 100
batch_size = 10
learning_rate = 0.001
sigma = 1


# In[84]:


# Use Tensorflow to build the same model
# Placeholders
X = tf.placeholder(dtype=tf.float32, shape=[None, n_var])
Y = tf.placeholder(dtype=tf.float32, shape=[None])


# In[86]:


# Initializers
sigma = sigma
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()


# In[87]:


# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_var, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))


# In[88]:


# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_2, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))


# In[89]:


# Activation Function:  Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))


# In[90]:


# Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_2, W_out), bias_out))


# In[91]:


# Cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= out, labels= Y))


# In[92]:


# Optimizer
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# In[93]:


# Make Session
# net = tf.Session()
# Run initializer
# net.run(tf.global_variables_initializer())


# In[102]:


# Lauch the graph

with tf.Session() as net:
    # Run initializer
    net.run(tf.global_variables_initializer())
    
    for e in range(epochs):

        # Initialize the value of average cost
        avg_cost = 0.
        total_batch = int(len(y_train) // batch_size)

        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]

        # Minibatch training
        for i in range(0, total_batch):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]
            # Run optimizer with batch
            net.run(opt, feed_dict={X: batch_x, Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch

        # Show progress
        # Display logs per epoch step
        if epochs % display_step == 0:
                print("Epoch:", '%04d' % (e+1), "cost=", "{:.9f}".format(avg_cost))
        print("Optimization Finished!")
        
    # Test model
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))
    global result 
    result = tf.argmax(out, 1).eval({x: X_test, y: Y_test})


# In[100]:


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

