
import numpy as np
import operator
import time
import os
import struct
from array import array
import math
import sys
import random
import matplotlib.pyplot as plt
import csv
from tabulate import tabulate
import cPickle
import gzip


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


def load_data_wrapper(train_data, train_label, test_data, test_labels):
    if os.path.exists('mnist.pkl.gz'):
        f = gzip.open('mnist.pkl.gz', 'rb')
    else:
        while(1):
            x=1
    tr_d, va_d, te_d = cPickle.load(f)
    f.close()

    train_data = np.vstack((tr_d[0], va_d[0]))
    training_inputs = [np.reshape(x, (784, 1)) for x in train_data] 
    training_results = [vectorized_result(y) for y in train_label]    
    
    training_data = zip(training_inputs, training_results)   
    
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, test_labels)

    return (training_inputs, training_results, test_inputs, test_labels)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def CreateConfusionMatrix(predictions, actual_classes, noOfClasses):
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
    confusionMatrix = [[0 for i in xrange(noOfClasses)] for i in xrange(noOfClasses)]
    for x in range(len(actual_classes)):
        if predictions[x] == actual_classes[x]:
            confusionMatrix[actual_classes[x]][actual_classes[x]] = confusionMatrix[actual_classes[x]][actual_classes[x]] + 1 
        else:
            confusionMatrix[actual_classes[x]][predictions[x]] = confusionMatrix[actual_classes[x]][predictions[x]] + 1
    return confusionMatrix

def CalculatePrecisionAndRecall(confusionMatrix, noOfClasses, noOfTestSamples):
    totalRecall = 0.0
    totalPrecision = 0.0
    precision, specificity, recall = [], [], []
    totalSpecificity = 0.0
    
    for i in range(10):
        classPrecision = 0.0
        for j in range(10):
            classPrecision = classPrecision + confusionMatrix[j][i]
        if classPrecision != 0.0:
            classPrecision = (confusionMatrix[i][i] / float(classPrecision)) * 100
        else:
            classPrecision = 0.     
        precision.append(classPrecision)
        totalPrecision = totalPrecision + classPrecision
  
    for i in range(10):
        classRecall = 0.0
        for j in range(10):
            classRecall = classRecall + confusionMatrix[i][j]
        if classRecall != 0.0:
            classRecall = (confusionMatrix[i][i] / float(classRecall)) * 100
        else:
            classRecall = 0.0
        recall.append(classRecall)
        totalRecall = totalRecall + classRecall

    for i in range(10):
        numerator = noOfTestSamples - confusionMatrix[i][i]
        denominator = numerator
        for j in range(10):
            if i != j:
                denominator = denominator + confusionMatrix[j][i]
        classSpecificity = (numerator / float(denominator))
        classSpecificity =  classSpecificity * 100
        totalSpecificity = totalSpecificity + classSpecificity
        specificity.append(classSpecificity)

    avgRecall = (totalRecall / float(noOfClasses))
    avgPrecision = (totalPrecision / float(noOfClasses))
    avgSpecificity = (totalSpecificity / float(noOfClasses))
    return avgPrecision, avgRecall, avgSpecificity, precision, recall, specificity


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def PrintResults(confusionMatrix, avgPrecision, avgRecall, avgSpecificity, precision, recall, specificity):
    for i in range(0, len(precision)):
        print "Class", i
        print "----------------------------------"
        print "Precision :", precision[i]
        print "Recall :", recall[i]
        print "Specificity :", specificity[i]
        print "\n"
    print "______________________________________"
    print "Average Recall:", avgRecall
    print "Average Precision:", avgPrecision
    print "Average Specificity:", avgSpecificity
  

def GetAccuracy(testLabels, predictions):
    correct = 0
    for x in range(len(testLabels)):
        if testLabels[x] == predictions[x]:
                  correct += 1                                         
    return (correct/float(len(testLabels))) * 100.0 

       
def feedforward(a, biases, weights):
    for b, w in zip(biases, weights):
        a = sigmoid(np.dot(w, a)+b)
    return a
  
def cross_validation_division(iteration_no, training_data, training_label, testing_label):

    if iteration_no==1:
        train_data = training_data[:50000]
        train_label = training_label[:50000]
    
        test_inputs = training_data[50000:60000]
        test_labels = testing_label[50000:60000]
    
    elif iteration_no==2:
        train_data = training_data[10000:60000]
        train_label = training_label[10000:60000]
    
        test_inputs = training_data[:10000]
        test_labels = testing_label[:10000]
        

    elif iteration_no==3:
        train_data1 = training_data[:10000]
        train_label1 = training_label[:10000]
        train_data2 = training_data[20000:60000]
        train_label2 = training_label[20000:60000]
        train_data = np.vstack((train_data1, train_data2))
        train_label = np.vstack((train_label1, train_label2))
        test_inputs = training_data[10000:20000]
        test_labels = testing_label[10000:20000]
    

    elif iteration_no==4:
        train_data1 = training_data[:20000]
        train_label1 = training_label[:20000]
        train_data2 = training_data[30000:60000]
        train_label2 = training_label[30000:60000]
        train_data = np.vstack((train_data1, train_data2))
        train_label = np.vstack((train_label1, train_label2))
    
        test_inputs = training_data[20000:30000]
        test_labels = testing_label[20000:30000]
    
    elif iteration_no==5:
        train_data1 = training_data[:30000]
        train_label1 = training_label[:30000]
        train_data2 = training_data[40000:60000]
        train_label2 = training_label[40000:60000]
        train_data = np.vstack((train_data1, train_data2))
        train_label = np.vstack((train_label1, train_label2))


        test_inputs = training_data[30000:40000]
        test_labels = testing_label[30000:40000]
    
    elif iteration_no==6:
        train_data1 = training_data[:40000]
        train_label1 = training_label[:40000]
        train_data2 = training_data[50000:60000]
        train_label2 = training_label[50000:60000]
        train_data = np.vstack((train_data1, train_data2))
        train_label = np.vstack((train_label1, train_label2))
    
        test_inputs = training_data[40000:50000]
        test_labels = testing_label[40000:50000]
    

    test_data = zip(test_inputs, test_labels)
    training_data = zip(train_data, train_label)  
    return training_data, test_data


def SGD(training_inputs, training_label, testing_label, testing_final_inputs, testing_final_labels, epochs, mini_batch_size, eta, noOfClasses, biases, weights):  
    
    iteration_no = 1;
    accuracy_list=[] 
    error_list=[]

    epoch_iteration = 0
    for iteration in range(6):
        plot_error_list=[]
        plot_epoch_list=[]
        print
        print "***********************************"
        print "============FOLD NO {0}============".format(iteration_no)
        training_data, test_data = cross_validation_division(iteration_no, training_inputs, training_label, testing_label)
        iteration_no = iteration_no + 1
        n = len(training_data)
        n_test = len(test_data)

        for j in xrange(epochs):
            random.shuffle(training_data)
            epoch_iteration = epoch_iteration + 1
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                biases, weights = update_mini_batch(mini_batch, eta, biases, weights)
            sum,test_results=evaluate(test_data, biases, weights)
            print "Training Iteration {0}: {1} / {2}".format(j, sum, n_test)
            error = (1-(float(sum)/float(n_test)))
            plot_error_list.append(error)
            plot_epoch_list.append(j)

        
        #print test_results
        predictions = []
        testLabels=[]
        for i in test_results:
            predictions.append(int(i[0]))
            testLabels.append(int(i[1]))

        confusionMatrix = CreateConfusionMatrix(testLabels,predictions, noOfClasses)
        avgPrecision, avgRecall, avgSpecificity, precision, recall, specificity = CalculatePrecisionAndRecall(confusionMatrix, noOfClasses, len(testLabels))
        PrintResults(confusionMatrix, avgPrecision, avgRecall, avgSpecificity, precision, recall, specificity)
        
        accuracy = GetAccuracy(predictions, testLabels)
        accuracy_list.append(accuracy)
        print "Accuracy :", accuracy
        error = 100.00 - accuracy
        error_list.append(error)
        print "Error Rate :", error
        print "______________________________________"
        #plt.plot(plot_epoch_list, plot_error_list)
        #plt.show()

    print 
    print '*******************************************************'
    print '**************Average Values of FOLDS******************'
    print '------------------------------------------------------'               
    accuracy_list = np.array(accuracy_list)             
    print "Average Accuracy :", np.mean(accuracy_list, axis=0)
    error_list = np.array(error_list)             
    print "Average Error Rate :", np.mean(error_list, axis=0)
    print "Standard Deviation of Error Rate :", np.std(error_list, axis=0)
    print '*******************************************************'
    print

    print '========================================================'
    print '****************TESTING the Network********************'
    print '========================================================'
    print  
   
    test_data = zip(testing_final_inputs, testing_final_labels)
    n_test = len(test_data)
    sum,test_results=evaluate(test_data, biases, weights)
    print "Sum :", sum
    print "/",
    print n_test
    
    predictions = []
    testLabels=[]
    for i in test_results:
        predictions.append(int(i[0]))
        testLabels.append(int(i[1]))
    confusionMatrix = CreateConfusionMatrix(testLabels,predictions, noOfClasses)
    avgPrecision, avgRecall, avgSpecificity, precision, recall, specificity = CalculatePrecisionAndRecall(confusionMatrix, noOfClasses, len(testLabels))
    PrintResults(confusionMatrix, avgPrecision, avgRecall, avgSpecificity, precision, recall, specificity)
    accuracy = GetAccuracy(predictions, testLabels)
    print "Accuracy :", accuracy
    error = 100.00 - accuracy
    print "Error Rate :", error
    print '*******************************************************'
    print




def update_mini_batch(mini_batch, eta, biases, weights):
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = backprop(x, y, biases, weights)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(weights, nabla_w)]
    biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(biases, nabla_b)]

    return biases, weights

def backprop(x, y, biases, weights):
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    activation = x
    activations = [x] 
    zs = [] 
    for b, w in zip(biases, weights):
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)
    delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    for l in xrange(2, num_layers):
        z = zs[-l]
        spv = sigmoid_prime(z)
        delta = np.dot(weights[-l+1].transpose(), delta) * spv
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (nabla_b, nabla_w)

def evaluate(test_data, biases, weights):
    test_results = [(np.argmax(feedforward(x, biases, weights)), y) 
                    for (x, y) in test_data]
    #print test_results
    return sum(int(x == y) for (x, y) in test_results),test_results
        
def cost_derivative(output_activations, y):
    return (output_activations-y)

def binarisation(img):
    for i in range(len(img)):
        if img[i]>0:
            img[i]/= 255.0
        else:
            img[i]=0
    return img

if __name__ == '__main__':
    path='.'
    test_img_fname = 't10k-images.idx3-ubyte'
    test_lbl_fname = 't10k-labels.idx1-ubyte'
    train_img_fname = 'train-images.idx3-ubyte'
    train_lbl_fname = 'train-labels.idx1-ubyte'

    test_data, test_labels, train_data, train_label = [],[],[],[]
    test_data, test_labels = load(os.path.join(path, test_img_fname),os.path.join(path, test_lbl_fname))
    train_data, train_label = load(os.path.join(path, train_img_fname),os.path.join(path, train_lbl_fname))
    for i in range(len(train_data)):
        train_data[i] = binarisation(train_data[i])
    for i in range(len(test_data)):
        test_data[i] = binarisation(test_data[i])

    test_data = np.array(test_data)
    train_data = np.array(train_data)
    test_data = test_data.astype(float)
    train_data = train_data.astype(float)
    train_label = np.array(train_label)
    test_labels = np.array(test_labels)    
    
    training_data, training_results, test_inputs, test_labels =  load_data_wrapper(train_data, train_label, test_data, test_labels)
    sizes = [784, 10, 10]
    num_layers = len(sizes)
    biases = [np.random.randn(y, 1) for y in sizes[1:]]
    weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    SGD(training_data, training_results, train_label, test_inputs, test_labels, 20, 10, 1.0, 10, biases, weights)
