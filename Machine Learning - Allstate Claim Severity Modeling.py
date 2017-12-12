# -*- coding: utf-8 -*-
"""
Allstate Claim Severity Modeling:
The dataset allstate_train.csv  should be used for the purposes of model building and validation. 
The purpose of this project is to predict the continuous variable ‘loss’ which quantifies 
the severity of a claim to the insurance company. The information in this dataset has been 
completely stripped of its business context and simply contains both categorical and 
continuous attributes without any variable descriptions or hints. There are 116 categorical 
variables (named cat1-cat116) and 14 continuous attributes (named cont1-cont14).

Writeup:
Our team started the prediction process by examining the training data set provided. 
We noticed that the target variable (Loss variable) is not normally distributed, 
so we did a Log Function transformation (log(1+x) to all elements of the loss variable)  
and as seen in Figure 1, normality was fairly attained by this transformation.

Afterwards, We looked at the correlation patterns between the continuous variables, 
and we found some highly correlated pairs, review Table 1. 
However, we didn’t remove any from the model so as not to introduce any bias in the analysis. 
As for the categorical we observed that from  variable cat1 to cat 72 they have only two labels A and B. 
Where in most of the cases, B has very few entries. While from the variable cat73 to cat 108 have 
more than two labels. In order for us to predict the loss variable values using Machine Learning 
algorithms we are going to require numerical data. So we used the One-hot encoding converts 
the attribute variables to a binary vector. Our last step for exploratory data analysis, 
we split our data into training and validation set with 80/20 respectively, (our seed was 12345)

Our process was to implement different modelling methods and compare the mean absolute 
errors (MAE) from our predicted loss values and the actual loss values in the validation 
dataset we created. Our final model with the lowest MAE was a deep learning model. 
We used the keras library in python to build the model. We started with a simple model of 
one hidden layer with 30 nodes. In the second iteration we increased the number of nodes 
and kept doing so until there was no improvement in the mae. We then added another hidden 
layer and started tuning the number of nodes for that layer. If adding complexity, meaning 
number of nodes and hidden layers, improved the MAE on the validation set we kept doing so 
until we saw a drop. It was an iterative method. Our activation function was Rectified Linear Unit (ReLU). 
It returns the actual weighted value from the prior nodes if the value is positive otherwise 
it returns zero. We used both the adam method and the stochastic gradient descent method at 
different learning rates for the optimization of the model and compared the mae across them. 
We used 30 epochs for training the model.

Our final model with the least MAE had the input layer then 50 nodes in the first hidden layer, 
then 100 nodes in the second hidden layer, then again 100 nodes in the third hidden layer, 
then 50 nodes in the fourth hidden layer and finally the output layer. 
The activation function was ReLU throughout and stochastic gradient descent was the optimization method. 
We were able to achieve a MAE of 1153.8 on the validation set we had created.

"""


# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings('ignore')

# Read raw data from the file

import pandas 

#Read the train dataset
dataset = pandas.read_csv("C:/Users/Google Drive/MSA/Machine Learning/Kaggle/allstate_train.csv") 

#Read test dataset
dataset_test = pandas.read_csv("C:/Users/Google Drive/MSA/Machine Learning/Kaggle/allstate_test.csv")
#Save the id's for submission file
ID = dataset_test['id']
#Drop unnecessary columns
dataset_test.drop('id',axis=1,inplace=True)

#Print all rows and columns. Dont hide any
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)

#Display the first five rows to get a feel of the data
print(dataset.head(5))

# Size of the dataframe size & variables

print(dataset.shape)

# We can see that there are 188318 instances having 132 attributes

#Drop the first column 'id' since it just has serial numbers. Not useful in the prediction process.
dataset = dataset.iloc[:,1:]

# Statistical description

print(dataset.describe())

# Learning :
# No attribute in continuous columns is missing as count is 188318 for all, all rows can be used
# No negative values are present. Tests such as chi2 can be used
# Statistics not displayed for categorical data
# Skewness of the distribution

print(dataset.skew())

# We will visualize all the continuous attributes using Violin Plot - a combination of box and density plots

import numpy

#import plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

#range of features considered
split = 116 

#number of features considered
size = 15

#create a dataframe with only continuous features
data=dataset.iloc[:,split:] 

#get the names of all the columns
cols=data.columns 

#Plot violin for all attributes in a 7x2 grid
n_cols = 2
n_rows = 7

for i in range(n_rows):
    fg,ax = plt.subplots(nrows=1,ncols=n_cols,figsize=(12, 8))
    for j in range(n_cols):
        sns.violinplot(y=cols[i*n_cols+j], data=dataset, ax=ax[j])
        
        

#log1p function applies log(1+x) to all elements of the column

dataset["loss"] = numpy.log1p(dataset["loss"])
#visualize the transformed column
sns.violinplot(data=dataset,y="loss")  
plt.show()

# Calculates pearson co-efficient for all combinations
data_corr = data.corr()

# Set the threshold to select only highly correlated attributes
threshold = 0.5

# List of pairs along with correlation above threshold
corr_list = []

#Search for the highly correlated pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index

#Sort to show higher ones first            
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))

# Strong correlation is observed between the following pairs
# This represents an opportunity to reduce the feature set through transformations such as PCA

# Scatter plot of only the highly correlated pairs
for v,i,j in s_corr_list:
    sns.pairplot(dataset, size=6, x_vars=cols[i],y_vars=cols[j] )
    plt.show()

#cont11 and cont12 give an almost linear pattern...one must be removed
#cont1 and cont9 are highly correlated ...either of them could be safely removed 
#cont6 and cont10 show very good correlation too

# Count of each label in each category

#names of all the columns
cols = dataset.columns

#Plot count plot for all attributes in a 29x4 grid
n_cols = 4
n_rows = 29
for i in range(n_rows):
    fg,ax = plt.subplots(nrows=1,ncols=n_cols,sharey=True,figsize=(12, 8))
    for j in range(n_cols):
        sns.countplot(x=cols[i*n_cols+j], data=dataset, ax=ax[j])

#cat1 to cat72 have only two labels A and B. In most of the cases, B has very few entries
#cat73 to cat 108 have more than two labels
#cat109 to cat116 have many labels


import pandas

#cat1 to cat116 have strings. The ML algorithms we are going to study require numberical data
#One-hot encoding converts an attribute to a binary vector

#Variable to hold the list of variables for an attribute in the train and test data
labels = []

for i in range(0,split):
    train = dataset[cols[i]].unique()
    test = dataset_test[cols[i]].unique()
    labels.append(list(set(train) | set(test)))    

del dataset_test

#Import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#One hot encode all categorical attributes
cats = []





for i in range(0, split):
    #Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(dataset.iloc[:,i])
    feature = feature.reshape(dataset.shape[0], 1)
    #One hot encode
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)

# Make a 2D array from a list of 1D arrays
encoded_cats = numpy.column_stack(cats)

# Print the shape of the encoded data
print(encoded_cats.shape)

#Concatenate encoded attributes with continuous attributes
dataset_encoded = numpy.concatenate((encoded_cats,dataset.iloc[:,split:].values),axis=1)
del cats
del feature
del dataset
del encoded_cats
print(dataset_encoded.shape)


#get the number of rows and columns
r, c = dataset_encoded.shape

#create an array which has indexes of columns
i_cols = []
for i in range(0,c-1):
    i_cols.append(i)

#Y is the target column, X has the rest
X = dataset_encoded[:,0:(c-1)]
Y = dataset_encoded[:,(c-1)]
del dataset_encoded

#Validation chunk size
val_size = 0.2

#Use a common seed in all experiments so that same chunk is used for validation
seed = 12345

#Split the data into chunks
from sklearn import cross_validation
X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(X, Y, test_size=val_size, random_state=seed)
del X
del Y

#All features
X_all = []

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae = []

#Scoring parameter

#Add this version of X to the list 
n = "All"
#X_all.append([n, X_train,X_val,i_cols])
X_all.append([n, i_cols])

X_train.shape

from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.optimizers import SGD

mae = [None]*20
mape = [None]*20
mae_sgd = [None]*20
mape_sgd = [None]*20

#Model1
i = 0
model = Sequential()
n_cols = X_train.shape[1]

model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')


early_stopping_monitor = EarlyStopping(patience=2)
model.fit(X_train, Y_train, epochs = 30, callbacks = [early_stopping_monitor])


model.save('model3.h5')
model = load_model('model3.h5')

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1

error = abs(loss_val - loss_pred)

mae[i] = numpy.mean(abs(error))
mape[i] = numpy.mean(abs(error)*100/loss_val)

mae[i]
mape[i]             


#Model2
i = 1
model = Sequential()
n_cols = X_train.shape[1]

model.add(Dense(250, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')


early_stopping_monitor = EarlyStopping(patience=2)
model.fit(X_train, Y_train, epochs = 30, callbacks = [early_stopping_monitor])


predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1

error = abs(loss_val - loss_pred)

mae[i] = numpy.mean(abs(error))
mape[i] = numpy.mean(abs(error)*100/loss_val)

mae[i]
mape[i]             
     

#Model3
i = 2
model = Sequential()
n_cols = X_train.shape[1]

model.add(Dense(100, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')


early_stopping_monitor = EarlyStopping(patience=2)
model.fit(X_train, Y_train, epochs = 30, callbacks = [early_stopping_monitor])


predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1

error = abs(loss_val - loss_pred)

mae[i] = numpy.mean(abs(error))
mape[i] = numpy.mean(abs(error)*100/loss_val)

mae[i]
mape[i]             
     
#Model4
i = 3
model = Sequential()
n_cols = X_train.shape[1]

model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')


early_stopping_monitor = EarlyStopping(patience=2)
model.fit(X_train, Y_train, epochs = 30, callbacks = [early_stopping_monitor])


predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1

error = abs(loss_val - loss_pred)

mae[i] = numpy.mean(abs(error))
mape[i] = numpy.mean(abs(error)*100/loss_val)

mape[i]             
mae[i]

     
#Model5
i = 4
model = Sequential()
n_cols = X_train.shape[1]

model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)


lr_to_test = [.05, 0.01, 0.1]
# loop over learning rates
j = 0
for lr in lr_to_test:
    my_optimizer = SGD(lr=0.01)
    model.compile(optimizer = my_optimizer, loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])
        
    predictions = model.predict(X_val)
    loss_val = numpy.exp(Y_val) - 1
    loss_pred = numpy.exp(predictions[:,0]) - 1
    
    error = abs(loss_val - loss_pred)
    
    mae_sgd[j] = numpy.mean(abs(error))
    mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
    j = j + 1


model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1], mae_sgd[2]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1], mape_sgd[2]]

mape[i]             
mae[i]


     
#Model6
i = 5
model = Sequential()
n_cols = X_train.shape[1]

model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)


lr_to_test = [.05, 0.01, 0.1]
# loop over learning rates
j = 0
for lr in lr_to_test:
    my_optimizer = SGD(lr=0.01)
    model.compile(optimizer = my_optimizer, loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])
        
    predictions = model.predict(X_val)
    loss_val = numpy.exp(Y_val) - 1
    loss_pred = numpy.exp(predictions[:,0]) - 1
    
    error = abs(loss_val - loss_pred)
    
    mae_sgd[j] = numpy.mean(abs(error))
    mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
    mape_sgd[j]
    mae_sgd[j]
    j = j + 1


model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1], mae_sgd[2]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1], mape_sgd[2]]

mape[i]             
mae[i]



     
#Model7
i = 6
model = Sequential()
n_cols = X_train.shape[1]

model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)


lr_to_test = [0.01, 0.005, 0.001]
# loop over learning rates
j = 0
for lr in lr_to_test:
    my_optimizer = SGD(lr=lr)
    model.compile(optimizer = my_optimizer, loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])
        
    predictions = model.predict(X_val)
    loss_val = numpy.exp(Y_val) - 1
    loss_pred = numpy.exp(predictions[:,0]) - 1
    
    error = abs(loss_val - loss_pred)
    
    mae_sgd[j] = numpy.mean(abs(error))
    mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
    mape_sgd[j]
    mae_sgd[j]
    j = j + 1


model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1], mae_sgd[2]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1], mape_sgd[2]]

mape[i]             
mae[i]

     
#Model8
i = 7
model = Sequential()
n_cols = X_train.shape[1]

model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(350, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)


lr_to_test = [ 0.005, 0.001, 0.0005,]
# loop over learning rates
j = 0
for lr in lr_to_test:
    my_optimizer = SGD(lr=lr)
    model.compile(optimizer = my_optimizer, loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])
        
    predictions = model.predict(X_val)
    loss_val = numpy.exp(Y_val) - 1
    loss_pred = numpy.exp(predictions[:,0]) - 1
    
    error = abs(loss_val - loss_pred)
    
    mae_sgd[j] = numpy.mean(abs(error))
    mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
    mape_sgd[j]
    mae_sgd[j]
    j = j + 1


model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1], mae_sgd[2]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1], mape_sgd[2]]

mape[i]             
mae[i]          


     
#Model9
i = 8
model = Sequential()
n_cols = X_train.shape[1]


model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)


lr_to_test = [0.001, 0.0005, 0.0008]
# loop over learning rates
j = 0
for lr in lr_to_test:
    my_optimizer = SGD(lr=lr)
    model.compile(optimizer = my_optimizer, loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])
        
    predictions = model.predict(X_val)
    loss_val = numpy.exp(Y_val) - 200
    loss_pred = numpy.exp(predictions[:,0]) - 200
    
    error = abs(loss_val - loss_pred)
    
    mae_sgd[j] = numpy.mean(abs(error))
    mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
    mape_sgd[j]
    mae_sgd[j]
    j = j + 1


model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 200
loss_pred = numpy.exp(predictions[:,0]) - 200
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1], mae_sgd[2]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1], mape_sgd[2]]

mape[i]             
mae[i]          




#Model10
i = 9
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)


lr_to_test = [0.001, 0.0005, 0.0008]
# loop over learning rates
j = 0
for lr in lr_to_test:
    my_optimizer = SGD(lr=lr)
    model.compile(optimizer = my_optimizer, loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])
        
    predictions = model.predict(X_val)
    loss_val = numpy.exp(Y_val) - 200
    loss_pred = numpy.exp(predictions[:,0]) - 200
    
    error = abs(loss_val - loss_pred)
    
    mae_sgd[j] = numpy.mean(abs(error))
    mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
    mape_sgd[j]
    mae_sgd[j]
    j = j + 1


model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 200
loss_pred = numpy.exp(predictions[:,0]) - 200
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1], mae_sgd[2]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1], mape_sgd[2]]

mape[i]             
mae[i]          

#Model11
i = 10
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)


lr_to_test = [0.001,0.0001]
# loop over learning rates
j = 0
for lr in lr_to_test:
    my_optimizer = SGD(lr=lr)
    model.compile(optimizer = my_optimizer, loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs = 30, callbacks = [early_stopping_monitor])
        
    predictions = model.predict(X_val)
    loss_val = numpy.exp(Y_val) - 1
    loss_pred = numpy.exp(predictions[:,0]) - 1
    
    error = abs(loss_val - loss_pred)
    
    mae_sgd[j] = numpy.mean(abs(error))
    mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
    mape_sgd[j]
    mae_sgd[j]
    j = j + 1


model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1], mae_sgd[2]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1], mape_sgd[2]]

mape[i]             
mae[i]          

#Model12
i = 11
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)


lr_to_test = [0.001]
# loop over learning rates
j = 0
for lr in lr_to_test:
    my_optimizer = SGD(lr=lr)
    model.compile(optimizer = my_optimizer, loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs = 30, callbacks = [early_stopping_monitor])
        
    predictions = model.predict(X_val)
    loss_val = numpy.exp(Y_val) - 1
    loss_pred = numpy.exp(predictions[:,0]) - 1
    
    error = abs(loss_val - loss_pred)
    
    mae_sgd[j] = numpy.mean(abs(error))
    mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
    mape_sgd[j]
    mae_sgd[j]
    j = j + 1


model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 30, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1], mae_sgd[2]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1], mape_sgd[2]]

mape[i]             
mae[i]          



#Model12
i = 11
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)


lr_to_test = [0.001]
# loop over learning rates
j = 0
for lr in lr_to_test:
    my_optimizer = SGD(lr=lr)
    model.compile(optimizer = my_optimizer, loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs = 10, callbacks = [early_stopping_monitor])
        
    predictions = model.predict(X_val)
    loss_val = numpy.exp(Y_val) - 1
    loss_pred = numpy.exp(predictions[:,0]) - 1
    
    error = abs(loss_val - loss_pred)
    
    mae_sgd[j] = numpy.mean(abs(error))
    mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
    mape_sgd[j]
    mae_sgd[j]
    j = j + 1


model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 10, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1], mae_sgd[2]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1], mape_sgd[2]]

mape[i]             
mae[i]          


#Model13
i = 12
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(150, activation='relu', input_shape = (n_cols,)))
model.add(Dense(300, activation='relu', init='normal'))
model.add(Dense(150, activation='relu', init='normal'))
model.add(Dense(50, activation='relu', init='normal'))
model.add(Dense(200, activation='relu', init='normal'))
model.add(Dense(50, activation='relu', init='normal'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)


lr_to_test = [0.0005]
# loop over learning rates
j = 0
for lr in lr_to_test:
    my_optimizer = SGD(lr=lr)
    model.compile(optimizer = my_optimizer, loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])
        
    predictions = model.predict(X_val)
    loss_val = numpy.exp(Y_val) - 1
    loss_pred = numpy.exp(predictions[:,0]) - 1
    
    error = abs(loss_val - loss_pred)
    
    mae_sgd[j] = numpy.mean(abs(error))
    mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
    mape_sgd[j]
    mae_sgd[j]
    j = j + 1


model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1], mae_sgd[2]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1], mape_sgd[2]]

mape[i]             
mae[i]          


from keras.layers import Dropout

#Model14
i = 13
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(150, activation='relu', input_shape = (n_cols,)))
model.add(Dropout(0.2))
model.add(Dense(300, activation='relu', init='normal'))
model.add(Dropout(0.2))
model.add(Dense(150, activation='relu', init='normal'))
model.add(Dropout(0.2))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)


lr_to_test = [1,0.01,0.001]
# loop over learning rates
j = 0
for lr in lr_to_test:
    my_optimizer = SGD(lr=lr, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(optimizer = my_optimizer, loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])
        
    predictions = model.predict(X_val)
    loss_val = numpy.exp(Y_val) - 1
    loss_pred = numpy.exp(predictions[:,0]) - 1
    
    error = abs(loss_val - loss_pred)
    
    mae_sgd[j] = numpy.mean(abs(error))
    mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
    mape_sgd[j]
    mae_sgd[j]
    j = j + 1


model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1]]

mape[i]             
mae[i]          



#Model15
i = 14
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dropout(0.2))
model.add(Dense(250, activation='relu', init='normal'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu', init='normal'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu', init='normal'))
model.add(Dropout(0.2))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)


lr_to_test = [0.01]
# loop over learning rates
j = 0
for lr in lr_to_test:
    my_optimizer = SGD(lr=lr)
    model.compile(optimizer = my_optimizer, loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])
        
    predictions = model.predict(X_val)
    loss_val = numpy.exp(Y_val) - 1
    loss_pred = numpy.exp(predictions[:,0]) - 1
    
    error = abs(loss_val - loss_pred)
    
    mae_sgd[j] = numpy.mean(abs(error))
    mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
    mape_sgd[j]
    mae_sgd[j]
    j = j + 1


model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1], mae_sgd[2]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1], mae_sgd[2]]

mape[i]             
mae[i]          


#Model16
i = 15
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(200, activation='relu', input_shape = (n_cols,)))
model.add(Dropout(0.2))
model.add(Dense(350, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)


model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1], mae_sgd[2]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1], mae_sgd[2]]

mape[i]             
mae[i]          



#Model17
i = 16
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(10, activation='relu', input_shape = (n_cols,)))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)


model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1], mae_sgd[2]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1], mae_sgd[2]]

mape[i]             
mae[i]          




#Model17
i = 16
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dropout(0.2))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)

lr_to_test = [0.01,0.001,0.0001]
# loop over learning rates
j = 0
for lr in lr_to_test:
    my_optimizer = SGD(lr=lr)
    model.compile(optimizer = my_optimizer, loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])
        
    predictions = model.predict(X_val)
    loss_val = numpy.exp(Y_val) - 1
    loss_pred = numpy.exp(predictions[:,0]) - 1
    
    error = abs(loss_val - loss_pred)
    
    mae_sgd[j] = numpy.mean(abs(error))
    mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
    mape_sgd[j]
    mae_sgd[j]
    j = j + 1

model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1], mae_sgd[2]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1], mae_sgd[2]]

mape[i]             
mae[i]          




#Model18
i = 17
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)
#==============================================================================
# 
# lr_to_test = [0.01,0.001,0.0001]
# # loop over learning rates
# j = 0
# for lr in lr_to_test:
#     my_optimizer = SGD(lr=lr)
#     model.compile(optimizer = my_optimizer, loss='mean_squared_error')
#     model.fit(X_train, Y_train, epochs = 10, callbacks = [early_stopping_monitor])
#         
#     predictions = model.predict(X_val)
#     loss_val = numpy.exp(Y_val) - 1
#     loss_pred = numpy.exp(predictions[:,0]) - 1
#     
#     error = abs(loss_val - loss_pred)
#     
#     mae_sgd[j] = numpy.mean(abs(error))
#     mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
#     mape_sgd[j]
#     mae_sgd[j]
#     j = j + 1
# 
#==============================================================================
model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1], mae_sgd[2]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1], mae_sgd[2]]

mape[i]             
mae[i]          


#Model19
i = 18
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(350, activation='relu', input_shape = (n_cols,)))
model.add(Dense(250, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)

lr_to_test = [0.001,0.0001]
# loop over learning rates
j = 0
for lr in lr_to_test:
    my_optimizer = SGD(lr=lr)
    model.compile(optimizer = my_optimizer, loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs = 10, callbacks = [early_stopping_monitor])
        
    predictions = model.predict(X_val)
    loss_val = numpy.exp(Y_val) - 1
    loss_pred = numpy.exp(predictions[:,0]) - 1
    
    error = abs(loss_val - loss_pred)
    
    mae_sgd[j] = numpy.mean(abs(error))
    mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
    mape_sgd[j]
    mae_sgd[j]
    j = j + 1

model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1], mae_sgd[2]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1], mae_sgd[2]]

mape[i]             
mae[i]          


#Model20
i = 19
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(150, activation='relu', input_shape = (n_cols,)))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)

lr_to_test = [0.001,0.0001]
# loop over learning rates
j = 0
for lr in lr_to_test:
    my_optimizer = SGD(lr=lr)
    model.compile(optimizer = my_optimizer, loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs = 10, callbacks = [early_stopping_monitor])
        
    predictions = model.predict(X_val)
    loss_val = numpy.exp(Y_val) - 1
    loss_pred = numpy.exp(predictions[:,0]) - 1
    
    error = abs(loss_val - loss_pred)
    
    mae_sgd[j] = numpy.mean(abs(error))
    mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
    mape_sgd[j]
    mae_sgd[j]
    j = j + 1

model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1]]

mape[i]             
mae[i]          



#Model20
i = 19
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(500, activation='relu', input_shape = (n_cols,)))
model.add(Dense(350, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)
#==============================================================================
# 
# lr_to_test = [0.001,0.0001]
# # loop over learning rates
# j = 0
# for lr in lr_to_test:
#     my_optimizer = SGD(lr=lr)
#     model.compile(optimizer = my_optimizer, loss='mean_squared_error')
#     model.fit(X_train, Y_train, epochs = 10, callbacks = [early_stopping_monitor])
#         
#     predictions = model.predict(X_val)
#     loss_val = numpy.exp(Y_val) - 1
#     loss_pred = numpy.exp(predictions[:,0]) - 1
#     
#     error = abs(loss_val - loss_pred)
#     
#     mae_sgd[j] = numpy.mean(abs(error))
#     mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
#     mape_sgd[j]
#     mae_sgd[j]
#     j = j + 1
#==============================================================================

model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1]]

mape[i]             
mae[i]          




#Model20
i = 19
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(150, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)
#==============================================================================
# 
# lr_to_test = [0.001,0.0001]
# # loop over learning rates
# j = 0
# for lr in lr_to_test:
#     my_optimizer = SGD(lr=lr)
#     model.compile(optimizer = my_optimizer, loss='mean_squared_error')
#     model.fit(X_train, Y_train, epochs = 10, callbacks = [early_stopping_monitor])
#         
#     predictions = model.predict(X_val)
#     loss_val = numpy.exp(Y_val) - 1
#     loss_pred = numpy.exp(predictions[:,0]) - 1
#     
#     error = abs(loss_val - loss_pred)
#     
#     mae_sgd[j] = numpy.mean(abs(error))
#     mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
#     mape_sgd[j]
#     mae_sgd[j]
#     j = j + 1
#==============================================================================

model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1]]

mape[i]             
mae[i]          




#Model20
i = 19
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)
#==============================================================================
# 
# lr_to_test = [0.001,0.0001]
# # loop over learning rates
# j = 0
# for lr in lr_to_test:
#     my_optimizer = SGD(lr=lr)
#     model.compile(optimizer = my_optimizer, loss='mean_squared_error')
#     model.fit(X_train, Y_train, epochs = 10, callbacks = [early_stopping_monitor])
#         
#     predictions = model.predict(X_val)
#     loss_val = numpy.exp(Y_val) - 1
#     loss_pred = numpy.exp(predictions[:,0]) - 1
#     
#     error = abs(loss_val - loss_pred)
#     
#     mae_sgd[j] = numpy.mean(abs(error))
#     mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
#     mape_sgd[j]
#     mae_sgd[j]
#     j = j + 1
#==============================================================================

model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1]]

mape[i]             
mae[i]   






#Model20
i = 19
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(100, activation='relu', input_shape = (n_cols,)))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)
#==============================================================================
# 
# lr_to_test = [0.001,0.0001]
# # loop over learning rates
# j = 0
# for lr in lr_to_test:
#     my_optimizer = SGD(lr=lr)
#     model.compile(optimizer = my_optimizer, loss='mean_squared_error')
#     model.fit(X_train, Y_train, epochs = 10, callbacks = [early_stopping_monitor])
#         
#     predictions = model.predict(X_val)
#     loss_val = numpy.exp(Y_val) - 1
#     loss_pred = numpy.exp(predictions[:,0]) - 1
#     
#     error = abs(loss_val - loss_pred)
#     
#     mae_sgd[j] = numpy.mean(abs(error))
#     mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
#     mape_sgd[j]
#     mae_sgd[j]
#     j = j + 1
#==============================================================================

model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1]]

mape[i]             
mae[i]   



#Model20
i = 19
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)

lr_to_test = [0.001,0.0001]
# loop over learning rates
j = 0
for lr in lr_to_test:
    my_optimizer = SGD(lr=lr)
    model.compile(optimizer = my_optimizer, loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs = 10, callbacks = [early_stopping_monitor])
        
    predictions = model.predict(X_val)
    loss_val = numpy.exp(Y_val) - 1
    loss_pred = numpy.exp(predictions[:,0]) - 1
    
    error = abs(loss_val - loss_pred)
     
    mae_sgd[j] = numpy.mean(abs(error))
    mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
    mape_sgd[j]
    mae_sgd[j]
    j = j + 1

model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1]]

mape[i]             
mae[i]   


#Model21
i = 10
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)

lr_to_test = [0.001,0.0001]
# loop over learning rates
j = 0
for lr in lr_to_test:
    my_optimizer = SGD(lr=lr)
    model.compile(optimizer = my_optimizer, loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs = 10, callbacks = [early_stopping_monitor])
        
    predictions = model.predict(X_val)
    loss_val = numpy.exp(Y_val) - 1
    loss_pred = numpy.exp(predictions[:,0]) - 1
    
    error = abs(loss_val - loss_pred)
     
    mae_sgd[j] = numpy.mean(abs(error))
    mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
    mape_sgd[j]
    mae_sgd[j]
    j = j + 1

model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1]]

mape[i]             
mae[i]   


#Model21
i = 11
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)

lr_to_test = [0.001,0.0001]
# loop over learning rates
j = 0
for lr in lr_to_test:
    my_optimizer = SGD(lr=lr)
    model.compile(optimizer = my_optimizer, loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs = 10, callbacks = [early_stopping_monitor])
        
    predictions = model.predict(X_val)
    loss_val = numpy.exp(Y_val) - 1
    loss_pred = numpy.exp(predictions[:,0]) - 1
    
    error = abs(loss_val - loss_pred)
     
    mae_sgd[j] = numpy.mean(abs(error))
    mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
    mape_sgd[j]
    mae_sgd[j]
    j = j + 1

model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1]]

mape[i]             
mae[i]   


#Model21
i = 11
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)

#==============================================================================
# lr_to_test = [0.001,0.0001]
# # loop over learning rates
# j = 0
# for lr in lr_to_test:
#     my_optimizer = SGD(lr=lr)
#     model.compile(optimizer = my_optimizer, loss='mean_squared_error')
#     model.fit(X_train, Y_train, epochs = 10, callbacks = [early_stopping_monitor])
#         
#     predictions = model.predict(X_val)
#     loss_val = numpy.exp(Y_val) - 1
#     loss_pred = numpy.exp(predictions[:,0]) - 1
#     
#     error = abs(loss_val - loss_pred)
#      
#     mae_sgd[j] = numpy.mean(abs(error))
#     mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
#     mape_sgd[j]
#     mae_sgd[j]
#     j = j + 1
#==============================================================================

model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 20, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1]]

mape[i]             
mae[i]   


#Model22
i = 12
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)

#==============================================================================
# lr_to_test = [0.001,0.0001]
# # loop over learning rates
# j = 0
# for lr in lr_to_test:
#     my_optimizer = SGD(lr=lr)
#     model.compile(optimizer = my_optimizer, loss='mean_squared_error')
#     model.fit(X_train, Y_train, epochs = 10, callbacks = [early_stopping_monitor])
#         
#     predictions = model.predict(X_val)
#     loss_val = numpy.exp(Y_val) - 1
#     loss_pred = numpy.exp(predictions[:,0]) - 1
#     
#     error = abs(loss_val - loss_pred)
#      
#     mae_sgd[j] = numpy.mean(abs(error))
#     mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
#     mape_sgd[j]
#     mae_sgd[j]
#     j = j + 1
#==============================================================================

model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 40,callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1]]

mape[i]             
mae[i]   



#Model22
i = 12
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)

#==============================================================================
# lr_to_test = [0.001,0.0001]
# # loop over learning rates
# j = 0
# for lr in lr_to_test:
#     my_optimizer = SGD(lr=lr)
#     model.compile(optimizer = my_optimizer, loss='mean_squared_error')
#     model.fit(X_train, Y_train, epochs = 10, callbacks = [early_stopping_monitor])
#         
#     predictions = model.predict(X_val)
#     loss_val = numpy.exp(Y_val) - 1
#     loss_pred = numpy.exp(predictions[:,0]) - 1
#     
#     error = abs(loss_val - loss_pred)
#      
#     mae_sgd[j] = numpy.mean(abs(error))
#     mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
#     mape_sgd[j]
#     mae_sgd[j]
#     j = j + 1
#==============================================================================

model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 40,callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1]]

mape[i]             
mae[i]   




#Model23
i = 13
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)

#==============================================================================
# lr_to_test = [0.001,0.0001]
# # loop over learning rates
# j = 0
# for lr in lr_to_test:
#     my_optimizer = SGD(lr=lr)
#     model.compile(optimizer = my_optimizer, loss='mean_squared_error')
#     model.fit(X_train, Y_train, epochs = 10, callbacks = [early_stopping_monitor])
#         
#     predictions = model.predict(X_val)
#     loss_val = numpy.exp(Y_val) - 1
#     loss_pred = numpy.exp(predictions[:,0]) - 1
#     
#     error = abs(loss_val - loss_pred)
#      
#     mae_sgd[j] = numpy.mean(abs(error))
#     mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
#     mape_sgd[j]
#     mae_sgd[j]
#     j = j + 1
#==============================================================================

model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 100,callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1]]

mape[i]             
mae[i]   






###########Final###########
#Model21
i = 11
model = Sequential()
n_cols = X_train.shape[1]



model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model2 = model

early_stopping_monitor = EarlyStopping(patience=2)

#==============================================================================
# lr_to_test = [0.001,0.0001]
# # loop over learning rates
# j = 0
# for lr in lr_to_test:
#     my_optimizer = SGD(lr=lr)
#     model.compile(optimizer = my_optimizer, loss='mean_squared_error')
#     model.fit(X_train, Y_train, epochs = 10, callbacks = [early_stopping_monitor])
#         
#     predictions = model.predict(X_val)
#     loss_val = numpy.exp(Y_val) - 1
#     loss_pred = numpy.exp(predictions[:,0]) - 1
#     
#     error = abs(loss_val - loss_pred)
#      
#     mae_sgd[j] = numpy.mean(abs(error))
#     mape_sgd[j] = numpy.mean(abs(error)*100/loss_val)
#     mape_sgd[j]
#     mae_sgd[j]
#     j = j + 1
#==============================================================================

model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(X_train, Y_train, epochs = 30, callbacks = [early_stopping_monitor])

predictions = model.predict(X_val)
loss_val = numpy.exp(Y_val) - 1
loss_pred = numpy.exp(predictions[:,0]) - 1
error = abs(loss_val - loss_pred)
mae[i] = [numpy.mean(abs(error)), mae_sgd[0], mae_sgd[1]]
mape[i] = [numpy.mean(abs(error)*100/loss_val), mape_sgd[0], mape_sgd[1]]

mape[i]             
mae[i]   

numpy.mean(abs(error))


