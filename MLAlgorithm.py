#!/usr/bin/python
import numpy as np
import random as rd
import h5py

np.random.seed(999)
rd.seed(999)

from sklearn.model_selection import cross_val_score,StratifiedKFold,GridSearchCV
from sklearn.linear_model import LogisticRegression
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import keras
import grapher


def logisticRegression(train_IP,train_OP,test_IP,test_OP):
    correct_pred_test = 0
    correct_pred_train = 0
    total = test_IP.shape[0]
    lm = LogisticRegression(solver = "lbfgs").fit(train_IP,train_OP)
    print "The score of the Log Reg is ",lm.score(train_IP,train_OP)

    predictions_train = lm.predict(train_IP)
    predictions_train = predictions_train.tolist()
    train_OP = train_OP.tolist()

    for i in range(len(predictions_train)):
        
        if predictions_train[i] == train_OP[i]:
            correct_pred_train += 1

    total_train = len(predictions_train)
    score_train = (100.00 * correct_pred_train) / total_train
    print "The train accuracy is ",score_train,"%"


    predictions_test = lm.predict(test_IP)
    test_OP = test_OP.tolist()
    predictions_test = predictions_test.tolist()

    for i in range(len(predictions_test)):

        if predictions_test[i] == test_OP[i]:
            correct_pred_test += 1

    total_test = len(predictions_test)
    score_test = (100.00 * correct_pred_test)/ total_test
    print "The test accuracy is ",score_test,"%"

    return train_OP,test_OP,predictions_train,predictions_test


#default parameters are the most optimal parameters
def buildANN(X = None, Y = None, X_train = None,Y_train=None,X_test=None,Y_test=None,hidden_layers = 3,activation = 'softsign',optimizer = 'Adam',neurons = 100,epochs = 75,batch_sizes = 400,loss = 'binary_crossentropy',GS = False,CV = False):
    print "Welcome to Keras ANN"
    seed = 999
    num_cols_train = X_train.shape[1] #hardcoded for now
    final_op_layers = 1
    final_op_activation = "sigmoid"
    if loss == 'categorical_crossentropy':
        final_op_layers = 2
        final_op_activation = "softmax"



    if not CV:
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(neurons, input_dim=num_cols_train,activation = activation))
        
        for i in range(hidden_layers):
            model.add(keras.layers.Dense(neurons,activation = activation))


        model.add(keras.layers.Dense(final_op_layers,activation = final_op_activation))
        print "Let's now compile the model"
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        
        if GS:
            return model

        else:

            
            #save_the_weights()
            
            model.fit(X_train,Y_train,shuffle = False,epochs = epochs,batch_size = batch_sizes)
            print "FINISHED..."
            pred_train = model.predict(X_train)
            eval_train = model.evaluate(X_train,Y_train)
            print "The training set accuracy for the ANN is: ", (eval_train[1] * 100.00), "%"

            pred_test = model.predict(X_test)
            eval_test = model.evaluate(X_test,Y_test)
            print "The test set accuracy for the ANN is: ", (eval_test[1] * 100.00), "%"
            
            return Y_train,Y_test,pred_train,pred_test


    else:
        #cross validation
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        cvscores = []
        for train, test in kfold.split(X, Y):
          # create model
            print("TRAIN:", len(train.tolist()), "TEST:", len(test.tolist()))
            model = keras.models.Sequential()
            model.add(keras.layers.Dense(neurons, input_dim=num_cols_train,activation = activation))
            
            for i in range(hidden_layers):
                model.add(keras.layers.Dense(neurons,activation = activation))


            model.add(keras.layers.Dense(final_op_layers,activation = final_op_activation))
            #print "Let's now compile the model"
            model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
            model.fit(X[train], Y[train],shuffle = False,epochs = epochs,batch_size = batch_sizes)

            scores = model.evaluate(X[test], Y[test])
            #print "FINISHED..."
            print("Accuracy is %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)
        
        print("mean: %.2f%% std:(+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        return None,None,None,None



def gridsearch(X_train,Y_train):

    model = KerasClassifier(build_fn = buildANN,GS = True) #the most optimal batch_size,epochs pair
    optimizer = ['SGD', 'RMSprop',"Nadam","Adam"]
    activation = ['softsign', 'relu','softplus','softmax','tanh','sigmoid','linear','hard_sigmoid']
    neurons = [20,40,60,80,100]
    epochs = [75,150,300,450,600]
    batch_sizes = [100,200,300,400,500]
    layers = [1,2,3,4]
    loss = ['binary_crossentropy','categorical_crossentropy']
    param_grid = dict(loss = loss,optimizer = optimizer,neurons = neurons,epochs = epochs,batch_sizes = batch_sizes,layers = layers,activation = activation)
    
    grid = GridSearchCV(estimator = model,param_grid = param_grid,n_jobs = 4)
    grid_result = grid.fit(X_train, Y_train)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def load_datasets():
    input_data = np.loadtxt("input_data.txt",delimiter = ",")
    output_data = np.loadtxt("output_data.txt",delimiter = ",")
    input_data = input_data.astype(int)
    output_data = output_data.reshape((output_data.shape[0],1))
    output_data = output_data.astype(int)
    print "The dimensions of the input set are",input_data.shape
    print "The dimensions of the output set are",output_data.shape
    return input_data,output_data    


def split_data(input_data,output_data):

    #60-40% train test data
    train_rows = rd.sample(range(input_data.shape[0]),int(0.60 * input_data.shape[0]))
    train_rows.sort()
    train_input = []
    train_output = []
    test_input = []
    test_output = []



    for i in range(input_data.shape[0]):
        if i in train_rows:
            train_input.append(input_data[i])
            train_output.append(output_data[i][0])

        else:
            test_input.append(input_data[i])
            test_output.append(output_data[i][0])

    train_input = np.asarray(train_input)
    train_output = np.asarray(train_output)
    test_input = np.asarray(test_input)
    test_output = np.asarray(test_output)


    return train_input,train_output,test_input,test_output
    

def main(GS = False):
    
    input_data,output_data = load_datasets()
    
    train_input,train_output,test_input,test_output = split_data(input_data,output_data)
    
    

    if (GS):
        gridsearch(train_input,train_output)
        print "GridSearch is over"
        return

    _,_,pred_train,pred_test= buildANN(X = input_data,Y = output_data,X_train = train_input, Y_train = train_output, X_test = test_input, Y_test = test_output,GS = GS)
    
    

    # print "Logistic Regression gives us these results:"
    # A,B,C,D = logisticRegression(train_input,train_output,test_input,test_output) 

    return train_output,test_output,pred_train,pred_test
    
    


    
    



y_train,y_test,pred_train,pred_test = main()

grapher.graph_pred_acc(y_train,pred_train,"Train Accuracy")
grapher.graph_pred_acc(y_test,pred_test,"Test Accuracy")


