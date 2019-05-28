from sklearn import linear_model
import pandas as pd
import nltk
import keras
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score,StratifiedKFold,GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from TextProcessing import *
import numpy as np
import random as rd


def sigmoid_function(x):
    return 1/(1 + np.exp(-x))

def derivative_sigmoid(x):
    return x * (1-x)
    
def perform_grid_search_BE(model,X,Y):


    batch_sizes = [i * 50 for i in range(1,3)]
    epochs = [200 * i for i in range(1,3)]

    print "Let's try to implement the grid search operation first"
    param_grid = dict(batch_size=batch_sizes, epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs= 4)
    grid_result = grid.fit(X, Y)
    print "Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print "%f (%f) with: %r" % (mean, stdev, param)



def buildANN(X,y,epochs = 10000,learning_rate = 0.01):
    print "Welcome to ANN!"
    #need to build a NN with 2 hidden layers along with input and output layer (4 in total) which uses back propogation for updation and gradient descent
    np.random.seed(999)     #ensure consistency in random weights to start off with
    col = X.shape[1]
    row = y.shape[1]
    W0 = 2 * np.random.random_sample((col,10)) - 1  #dimensions = col(X) by 10 weight units of layer 0 and mean = 0 so that not much variance hence [-1,1] range
    W1 = 2 * np.random.random_sample((10,row)) -1   #dimensions = 10 by row(y) weight units of layer 1 and mean = 0 and again [-1,1] range
    print "weight 0's shape",W0.shape
    print "weight 1's shape",W1.shape
    for i in range(epochs): #chose 15000 as default to ensure a good prediction
        print "Epoch ",i
        Layer1 = sigmoid_function(X.dot(W0))
        Layer2 = sigmoid_function(Layer1.dot(W1))

        #the next remaining lines in the loop do the back propogation algorithm
        Layer2_error = y - Layer2
        Layer2_delta = Layer2_error * derivative_sigmoid(Layer2)

        Layer1_error = Layer2_delta.dot(W1.T)
        Layer1_delta = Layer1_error * derivative_sigmoid(Layer1)

        W1 = W1 + (learning_rate * Layer1.T.dot(Layer2_delta))
        W0 = W0 + (learning_rate * X.T.dot(Layer1_delta))

    print "Wow! GD has finished running!"
    file_handle0 = "weight1.txt"
    file_handle1 = "weight2.txt"
    f0 = open(file_handle0,"w")
    f1 = open(file_handle1,"w")
    np.savetxt(f0,W0)
    np.savetxt(f1,W1)
    print "Done writing into the files"
    f0.close()
    f1.close()

def ANNpredict(test_IP,test_OP,cutoff = 0.1):
    correct_test_pred = 0
    total = test_IP.shape[0]
    layer1_W = np.loadtxt("weight1.txt",delimiter = " ")
    layer2_W = np.loadtxt("weight2.txt",delimiter = " ")
    input_layer = test_IP
    layer1 = sigmoid_function(np.dot(input_layer,layer1_W))
    layer2 = sigmoid_function(np.dot(layer1,layer2_W))
    output_layer = layer2
    return output_layer

def linearRegression(train_IP,train_OP,test_IP,test_OP,cutoff = 0.01):
    correct_test_pred = 0
    total = test_IP.shape[0]
    lm = linear_model.LinearRegression().fit(train_IP,train_OP)
    print "The score of the Lin Reg is ",lm.score(train_IP,train_OP)
    predictions = lm.predict(test_IP)
    test_OP = test_OP.tolist()
    predictions = predictions.tolist()
    for i in range(len(predictions)):

        if abs(predictions[i][0] - test_OP[i][0]) <= cutoff:
        
            correct_test_pred += 1

    score = (100.00 * correct_test_pred)/ total
    return score


def keras_ANN(X_train,Y_train,X_test,Y_test,batch_size,epochs,optimizer = 'adam',loss = "binary_crossentropy"):
    
    print "Welcome to Keras ANN"
    num_cols_train = 8451 #hardcoded for now
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(128,input_shape=(num_cols_train,),activation = 'relu'))
    model.add(keras.layers.Dense(64,activation = "relu"))
    model.add(keras.layers.Dense(1,activation = "sigmoid"))
    print "Let's now compile the model"
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    
    

    model.fit(X_train,Y_train,shuffle = False,epochs = epochs,batch_size = batch_size)
    print "FINISHED..."

    pred_train = model.predict(X_train)
    eval_train = model.evaluate(X_train,Y_train)
    print "The training set accuracy for the ANN is: ", (eval_train[1] * 100.00), "%"

    pred_test = model.predict(X_test)
    eval_test = model.evaluate(X_test,Y_test)
    print "The test set accuracy for the ANN is: ", (eval_test[1] * 100.00), "%"
    
    return Y_train,Y_test,pred_train,pred_test


scoreLinR1 = linearRegression(train_input,train_output,test_input,test_output,0.01)
print "The linear Regression model with 1% as the difference cutoff yields us an accuracy of ",scoreLinR1," %"

scoreLinR2 = linearRegression(train_input,train_output,test_input,test_output,0.05)
print "The linear Regression model with 5% as the difference cutoff yields us an accuracy of ",scoreLinR2," %"
