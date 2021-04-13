import os
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from tensorflow.python.ops.gen_array_ops import Concat
import tensorflow as tf
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
from keras import models
from keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.gen_math_ops import select_eager_fallback


def loadGloveModel():
    File='xxx_local_path_xxx\\glove.840B.300d.txt'
    print("Running loading Glove model step")
    gloveModel={}
    f = open(File, encoding="utf-8")#,'r'
    punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    for line in f:
        '''
        for ele in line:  
            if ele in punc:  
                line = line.replace(ele, "") 
        splitLines = line.split(' ')
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        '''
        splitLines = line.split(' ')
        word = splitLines[0] ## The first entry is the word
        wordEmbedding = np.asarray(splitLines[1:], dtype='float32')
        gloveModel[word] = wordEmbedding
    print(len(gloveModel)," Glove words are loaded!")
    return gloveModel

def article_callback(data,labels):
    print("Running CNN model step")
    print('size of data : ',data.shape)#(10295, 30, 300)
    print('size of labels : ',labels.shape)
    data_train,data_test,y_train,y_test = train_test_split(data,labels,test_size=0.25,random_state=1000)
    input_shape = data_train.shape[1:]
    inputs=keras.Input(shape=input_shape)
    y2 = keras.layers.Conv1D(1, 2, activation=None)(inputs)#(None, 29, 1)
    z2=keras.layers.MaxPool1D(29,strides=None)(y2)
    y3 = keras.layers.Conv1D(1, 3, activation=None)(inputs)#(None, 28, 1)
    z3=keras.layers.MaxPool1D(28,strides=None)(y3)
    y4 = keras.layers.Conv1D(1, 4, activation=None)(inputs)#(None, 27, 1)
    z4=keras.layers.MaxPool1D(27,strides=None)(y4)
    y5 = keras.layers.Conv1D(1, 5, activation=None)(inputs)#(None, 26, 1)
    z5=keras.layers.MaxPool1D(26,strides=None)(y5)
    concat=keras.layers.concatenate([z2,z3,z4,z5],axis=1)
    res=keras.layers.Dense(2,activation='softmax')(concat)
    model=keras.Model(inputs=inputs,outputs=res,name="OffensiveDetectionModel")
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer=keras.optimizers.RMSprop(),metrics=['accuracy'])
    from sklearn.metrics import precision_score,recall_score
    class Metrics(keras.callbacks.Callback):
        def __init__(self, val_data):
            super().__init__()
            self.validation_data=val_data

        def on_train_begin(self, logs={}):
            self.precision = []
            self.recall = []

        def on_epoch_end(self, epoch, logs={}):
            pred = np.round(np.asarray(self.model.predict(self.validation_data[0])).argmax(axis=1))
            predict = np.argmax(pred, axis=1)
            targ = np.argmax(self.validation_data[1], axis=1)
            self.precision.append(precision_score(targ, predict,average='micro'))
            self.recall.append(recall_score(targ, predict,average='micro'))
            print('Precision: ',precision_score(targ, predict,average='micro'))
            print('Recall: ',recall_score(targ, predict,average='micro'))

        def avg_precision_score(self):
            return np.mean(self.precision)

        def avg_recall_score(self):
            return np.mean(self.recall)
 
    data_train2,data_val,y_train2,y_val = train_test_split(data_train,y_train,test_size=0.2,random_state=1000)
    val_data=(data_val, y_val)
    my_call_back=Metrics(val_data)
    history=model.fit(data_train2,y_train2,validation_data=val_data,batch_size=64,epochs=10, callbacks=[my_call_back])
    test_scores=model.evaluate(data_test,y_test,verbose=0)#loss, accuracy, f1_score, precision, recall
    print('F1-score : ',(2 * my_call_back.avg_precision_score()*my_call_back.avg_recall_score())/(my_call_back.avg_precision_score()+my_call_back.avg_recall_score()))
    print("Running CNN model step is done")
    return test_scores


def main():
    gloveModel =loadGloveModel()
    #print(gloveModel['hello'])
    train=pd.read_csv("D:\\DiversityChallenge\\Twitter-API-v2-sample-code-master\\data_gathered\\complete_data.csv",header=0)
    stop = set(stopwords.words('english'))
    ## Iterate over the data to preprocess by removing stopwords
    lines_without_stopwords=[] 
    for line in train['text'].values: 
        line = line.lower()
        line_by_words = re.findall(r'(?:\w+)', line, flags = re.UNICODE) # remove punctuation and split
        new_line=[]
        for word in line_by_words:
            if word not in stop:
                new_line.append(word)
        lines_without_stopwords.append(new_line)
    texts = lines_without_stopwords
    
    seq_txt=[]
    a=0
    for x in texts:
        inner_list=[]
        for y in x:
            inner_list.append((np.asarray(gloveModel[y])*pow(10,5)).astype(int) if y in gloveModel else np.zeros((300,)))
            a=a+1
        seq_txt.append(inner_list)

    print('the first sample of data : ',texts[0])
    print('number of words in twitter text: ',a)##2196016  Glove words are loaded!
    #######################################
    data = pad_sequences(seq_txt)
    labels = to_categorical(np.asarray(train['label_id']))
    test_score_ref=article_callback(data,labels)
    print("Test loss:", test_score_ref[0])
    print("Test accuracy:", test_score_ref[1])


if __name__ == "__main__":
    main()



############################################
#End Result:
#F1-score :  0.6785760517799353
#Test loss: 32.83867645263672
#Test accuracy: 0.8654817342758179
############################################