import numpy as np
import operator
import time
import os
import struct
from array import array
import math
import sys
import random
from matplotlib.pyplot import *
import csv
from tabulate import tabulate


# majority vote for a little bit optimized worker
def majority_vote(knn, labels):
    knn = [k[0, 0] for k in knn]
    a = {}
    for idx in knn:
        if labels[idx] in a.keys():
            a[labels[idx]] = a[labels[idx]] + 1
        else:
            a[labels[idx]] = 1
    return sorted(a.iteritems(), key=operator.itemgetter(1), reverse=True)[0][0]


def k_nearest_neighbours(train, test, labels):
    k = 7
    train_mat = np.mat(train)
    idx = 0
    size = len(test)
    prediction_list = []

    for test_sample in test:
        idx += 1
        knn = np.argsort(np.sum(np.power(np.subtract(train_mat, test_sample), 2), axis=1), axis=0)[:k]
        prediction = majority_vote(knn, labels)
        print prediction
        prediction_list.append(prediction)

    return prediction_list    

def load(path_img, path_lbl):
    with open(path_lbl, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049,''got %d' % magic)
        labels = array("B", file.read())

    with open(path_img, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051,''got %d' % magic)
        image_data = array("B", file.read())

    images = []
    for i in xrange(size):
        images.append([0]*rows*cols)
    for i in xrange(size):
        images[i][:] = image_data[i*rows*cols : (i+1)*rows*cols]
    return images, labels

#Calculate the ratio of the total correct predictions out of all predictions made : classification accuracy
def calculate_accuracy(test_labels, prediction_list):
    correct=0
    length_test_sample = len(test_labels)
    for i in range(len(prediction_list)):
        if test_labels[i] == prediction_list[i]:
            correct = correct+1
    print correct
    accuracy_percentage = (float(correct)/float(len(prediction_list))) * 100
    return accuracy_percentage
    
def confusion_matrix(predictions, actual_classes):
    #Fetching the name of the classes to dictionary and then to the list
    c =[0, 1, 2, 3, 4, 5 ,6, 7, 8, 9]
    length = len(c) 
    

    #Creating confusion matrix as list -> empty list and hence comparing and increasing the count
    confusion_matrix=[]
    for i in range(length):
        for j in range(length):
            confusion_matrix.append(0)

    count = 0
    for i in range(len(actual_classes)):
        for j in range(length):
            for k in range(length):
                if actual_classes[i] == c[j] and predictions[i] == c[k]:
                    count = count +1
                    confusion_matrix[j*length+k] = confusion_matrix[j*length+k]+1

    print "\t\t"+'PREDICTED'
    table = []
    
    #Append Classes name
    L=[]
    L.append('\t')
    L.append('\t')
    for i in range(length):
        L.append(c[i])
    table.append(L)

    #Create Empty Table
    L=[]
    for i in range(length):
        for j in range(length+2):
            if i==length/2:
                if j==0:
                    L.append('ACTUAL')
                elif j==1:
                    L.append(c[i])
                else:
                    L.append('\t')
            else:
                if j==1:
                    L.append(c[i])
                else:
                    L.append('\t')
        table.append(L)
        L=[]

    #Populate value to the confusion matrix/empty table
    value_index=0
    for i in range(1, length+1):
        for j in range(2, length+2):
            table[i][j] = confusion_matrix[value_index]
            value_index+=1

    print tabulate(table, tablefmt="grid")  



if __name__ == '__main__':
    path='.'
    test_img_fname = 't10k-images.idx3-ubyte'
    test_lbl_fname = 't10k-labels.idx1-ubyte'
    train_img_fname = 'train-images.idx3-ubyte'
    train_lbl_fname = 'train-labels.idx1-ubyte'

    test_images, test_labels, train_data, train_label = [],[],[],[]
    test_images, test_labels = load(os.path.join(path, test_img_fname),os.path.join(path, test_lbl_fname))
    train_data, train_label = load(os.path.join(path, train_img_fname),os.path.join(path, train_lbl_fname))

    prediction_list = k_nearest_neighbours(train_data, test_images, train_label)
    accuracy = calculate_accuracy(test_labels, prediction_list)
    print "ACCURACY ",
    print accuracy,
    print "%"

    confusion_matrix(prediction_list, test_labels)




