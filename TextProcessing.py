import numpy as np
import random as rd
import tensorflow as tf
import nltk
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn import linear_model


#to eliminate binary words
def check_ascii(word):
    for w in word:
        if ord(w) not in range(127):
            return False

    return True


def create_word_set(df):
    print "We are now creating the word_set"
    types = list(df["type"])
    messages = list(df["message"])

    N = len(messages)

    training = []
    print "First the list of dictionaries"
    for i in range(N):

        training.append({"class":types[i],"message": messages[i]})

    
    classes = list(set(types))
    word_set = []
    print "Next the list of tokenized words"
    for i in range(N):
        words = nltk.word_tokenize(messages[i])
        word_set += words

    stop_words = ['?','!']

    #need to only store the basic roots of the words in the word_set
    print "getting the full word_set which needs to be recorded"
    for i in range(len(word_set)):
        if word_set[i] not in stop_words:
            if check_ascii(word_set[i]):
                word_set[i] = stemmer.stem(word_set[i].lower())

    word_set = list(set(word_set)) #this contains the entire word list for prediction
    print "Phew! done with it :)"
    return (word_set,training)

        


def one_hot_encoding_words(word_set,training):
    x_train,y = list(),list() #y_train is collection of one hot encoded vectors of the types of the training set
    print "Starting the one hot encoding of words, Will take some time ....."
    for data in training:
        x_vector = []           #contains the one hot encoded vector for the inputs
        msg = data["message"]   
        type_msg = data["class"]
        pat = []
        for word in msg:
            if check_ascii(word):
                pat.append(stemmer.stem(word.lower()))
        
        for word in word_set:
        
            if word in pat:
                x_vector.append(1)

            else:
                x_vector.append(0)


        x_train.append(x_vector)
        if type_msg == 'ham':
            y.append(0)   

        else:
            y.append(1)

    print "Done creating the vectors. Was quite long...."
    print "Let's save them for future"
    train = np.asarray(x_train)
    output = np.asarray(y)
    output = np.reshape(output,(len(y),1))
    np.savetxt("input_data.txt",train.astype(int),delimiter = ",",fmt = "%i")
    np.savetxt("output_data.txt",output.astype(int),delimiter = ",",fmt = "%i")
    print "Hooray we saved them!"
    return train.astype(int),output.astype(int)

def process():
    
    stemmer = PorterStemmer()   #Porter has most of the roots in english, will be perfect for this data set
    df = pd.read_table("SMSSpamCollection.txt",header = None,names = ["type","message"])
    word_set,training = create_word_set(df)
    train,output = one_hot_encoding_words(word_set,training)
    return train,output


