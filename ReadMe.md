#Digit Recogniser
____________________________________________________________________________________________________________________________________________
##Preamble
____________________________________________________________________________________________________________________________________________
The aim of this assignment is to experiment with Multilayer Feedforward Neural Network (MLFNN) with Backpropagation (BP) learning we learned as part of Chapter 6 on real world problems. 


____________________________________________________________________________________________________________________________________________
##Implementation - Question 1:
-> Load the dataset into train images, train labels, test images, test labels.
-> Divide the train images in a ratio of (1:5) i.e. 10000 for testing and remaining 50000 for training the network.
-> Using 6 folds, train the network using mini_batches of train data and hence update biases and weights.
-> The biases and weights so obtained are feed forwarded to verify the remaining data for testing, and hence the confusion matrix and rest parameters are printed.
-> The plot of epochs vs error rate is plotted for each fold.
-> The same process is repeated for 6 folds.
-> The final biases and weights are recored, and average values of 6 folds are printed.

-> The final biases and weights so obtained are used to test the trained network for the given test data.
-> The accuracy and error value is noted.


____________________________________________________________________________________________________________________________________________
##Implementation - Question 2:
->KNN is implemented for the given dataset.
-> The outputs for k=1,3,5 are recorded.


____________________________________________________________________________________________________________________________________________
##Implementation - Question 3:
-> Adding Noise - It is implemented by addding noise to the train data
-> Weight Decay - It is implemented by changing the updatation of weight/biases function by adding a decay factor

____________________________________________________________________________________________________________________________________________
##Instructions to use the code:

-> The code is run using python filename.py without giving any command line arguments.
-> The mnist test data should be present in the same folder as of the code file.

__________________________________________________________________________________________________________________________________________
##Libraries/Packages Needed
These are all the packages that are required so as to the run the code:
	
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
		import cPickle
		import gzip

********************************************************************************************************************************************

