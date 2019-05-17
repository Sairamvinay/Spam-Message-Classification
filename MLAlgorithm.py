import numpy as np
import random as rd


np.random.seed(999)
rd.seed(999)

from sklearn.model_selection import cross_val_score,StratifiedKFold,GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import keras
import matplotlib.pyplot as plt


def graph_optimize(x,y,gx_title,gy_title,g_title):
    plt.plot(x,y)
    plt.xlabel(gx_title)
    plt.ylabel(gy_title)
    plt.title(g_title)
    plt.show()
    plt.close('all')


def graph_hyp():
    batch_sizes = [100,200,300,400,500]
    epochs = [150,300,450,600]
    train_acc_BS = [0.99865375,0.9985793,0.9985044,0.9985044,0.9984296]
    train_acc_EP = [0.998744,0.9983847,0.9985044,0.9985044]

    test_acc_BS = [0.976446,0.977568,0.97712,0.977793,0.978385006]
    test_acc_EP = [0.975774,0.9780,0.97783737,0.978376]
    graph_optimize(batch_sizes,test_acc_BS,"Batch Size","Test Accuracy Averaged over different epochs","Test Accuracy vs Batch size")
    graph_optimize(epochs,test_acc_EP,"Epochs","Test Accuracy Averaged over different Batch Size","Test Accuracy vs Epochs")




def grapher(x,y,g_title):
    fig, ax = plt.subplots()
    plt.xlabel("Actual Output")
    plt.ylabel("Predicted Output")
    plt.title(g_title)
    ax.scatter(x,y,c = 'b')

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.savefig(g_title +  '.png')
    plt.close('all')
    

def logisticRegression(train_IP,train_OP,test_IP,test_OP):
    correct_pred_test = 0
    correct_pred_train = 0
    total = test_IP.shape[0]
    lm = linear_model.LogisticRegression(solver = "lbfgs").fit(train_IP,train_OP)
    print "The score of the Log Reg is ",lm.score(train_IP,train_OP)

    predictions_train = lm.predict(train_IP)
    predictions_train = predictions_train.tolist()
    train_OP = train_OP.tolist()

    for i in range(len(predictions_train)):
        train_OP[i] = train_OP[i][0]
        if predictions_train[i] == train_OP[i]:
            correct_pred_train += 1

    total_train = len(predictions_train)
    score_train = (100.00 * correct_pred_train) / total_train
    print "The train accuracy is ",score_train,"%"


    predictions_test = lm.predict(test_IP)
    test_OP = test_OP.tolist()
    predictions_test = predictions_test.tolist()

    for i in range(len(predictions_test)):
        test_OP[i] = test_OP[i][0]
        if predictions_test[i] == test_OP[i]:
            correct_pred_test += 1

    total_test = len(predictions_test)
    score_test = (100.00 * correct_pred_test)/ total_test
    print "The test accuracy is ",score_test,"%"

    return train_OP,test_OP,predictions_train,predictions_test




def keras_ANN(X_train,Y_train,X_test,Y_test,batch_size,epochs,optimizer = 'adam'):
    
    print "Welcome to Keras ANN"
    num_cols_train = 8451 #hardcoded for now
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(128,input_shape=(num_cols_train,),activation = 'relu'))
    model.add(keras.layers.Dense(64,activation = "relu"))
    model.add(keras.layers.Dense(1,activation = "sigmoid"))
    print "Let's now compile the model"
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    model.fit(X_train,Y_train,shuffle = False,epochs = epochs,batch_size = batch_size)
    print "FINISHED..."

    pred_train = model.predict(X_train)
    eval_train = model.evaluate(X_train,Y_train)
    print "The training set accuracy for the ANN is: ", (eval_train[1] * 100.00), "%"

    pred_test = model.predict(X_test)
    eval_test = model.evaluate(X_test,Y_test)
    print "The test set accuracy for the ANN is: ", (eval_test[1] * 100.00), "%"
    
    return Y_train,Y_test,pred_train,pred_test



def main():
    
    input_data = np.loadtxt("input_data.txt",delimiter = ",")
    output_data = np.loadtxt("output_data.txt",delimiter = ",")
    input_data = input_data.astype(int)
    output_data = output_data.reshape((output_data.shape[0],1))
    output_data = output_data.astype(int)
    print "The dimensions of the input set are",input_data.shape
    print "The dimensions of the output set are",output_data.shape
    
    
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
            train_output.append(output_data[i])

        else:
            test_input.append(input_data[i])
            test_output.append(output_data[i])

    train_input = np.asarray(train_input)
    train_output = np.asarray(train_output)
    test_input = np.asarray(test_input)
    test_output = np.asarray(test_output)

    


    A,B,C,D = keras_ANN(train_input,train_output,test_input,test_output,batch_size = 100,epochs = 150,optimizer = 'Adam')


    '''

    print "Logistic Regression gives us these results:"
    A,B,C,D = logisticRegression(train_input,train_output,test_input,test_output) 
    
    '''


    return A,B,C,D
    



y_train,y_test,pred_train,pred_test = main()
#graph_hyp()
