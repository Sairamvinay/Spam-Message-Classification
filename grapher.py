import matplotlib.pyplot as plt
import numpy as np


#for the pred vs acc graphs
def graph_pred_acc(x,y,g_title):
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
    

#for the regular plots (free hand joining)
def graph_reg(x,y,gx_title,gy_title,g_title):
    plt.plot(x,y)
    plt.xlabel(gx_title)
    plt.ylabel(gy_title)
    plt.title(g_title)
    plt.show()
    plt.close('all')

#for the bar graphs
def graph_bar(x,y,gx_title,gy_title,g_title):

    x_pos = np.arange(len(x))
    plt.bar(x_pos, y, align='center', alpha=0.5)
    plt.xticks(x_pos, x)
    plt.ylim(min(y) - np.std(y),max(y) + np.std(y))
    plt.xlabel(gx_title)
    plt.ylabel(gy_title)
    plt.title(g_title)
    plt.show()
    plt.close('all')


#for the hyperparameters
def graph_hyp():
    
    batch_sizes = [100,200,300,400,500]
    epochs = [75,150,300,450,600]
    neurons = [20,40,60,80,100]
    layers = [1,2,3,4]
    optimizer = ['SGD', 'RMSprop',"Nadam","Adam"]
    activation = ['softsign', 'relu','softplus','softmax','tanh','sigmoid','linear',(5 * ' ') + "hard_sigmoid"]
    loss = ['binary_crossentropy','categorical_crossentropy']


    train_acc_BS = np.array([97.2061,97.30782,97.34968,97.38558,97.36164])/100
    train_acc_EP = np.array([97.45736,97.39754,97.30782,97.27192,97.17618])/100    
    train_acc_neurons =np.array( [97.3078,97.3078,97.3377,97.3078,97.4574])/100
    train_acc_layers = np.array([97.2181,97.2480,97.3078,97.1882])/100
    train_acc_activ = np.array([97.5172,97.4574,97.2480,96.6497,97.4873,96.9189,97.3676,96.8890])/100
    train_acc_opt = np.array([95.9916,97.3676,97.4574,97.5770])/100
    train_acc_loss = np.array([97.3377,97.3078])/100

    
    graph_bar(batch_sizes,train_acc_BS.tolist(),"Batch Size","Train Accuracy Averaged over different epochs","Train Accuracy vs Batch size")
    graph_bar(epochs,train_acc_EP.tolist(),"Epochs","Train Accuracy Averaged over different Batch Size","Train Accuracy vs Epochs")
    graph_bar(neurons,train_acc_neurons.tolist(),"Neurons","Train Accuracy","Train Accuracy vs Neurons")
    graph_bar(layers,train_acc_layers.tolist(),"Layers","Train Accuracy", "Train Accuracy vs Layers")
    graph_bar(optimizer,train_acc_opt.tolist(),"Optimizer","Train Accuracy", "Train Accuracy vs Optimizer")
    graph_bar(activation,train_acc_activ.tolist(),"Activation function","Train Accuracy", "Train Accuracy vs Activation function")
    graph_bar(loss,train_acc_loss.tolist(),"Loss function","Train Accuracy", "Train Accuracy vs Loss function")

