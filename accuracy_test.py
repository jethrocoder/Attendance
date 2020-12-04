import numpy as np
import pandas as pd
import os
from face_recognition import distance,knn

skip = 0
dataset_path = './val_data/'

face_data = [] 
labels = []     # we will use roll number as labels

# class_id = 0 # Labels for the given file
names = {} #Mapping btw rollno - name

# Data Preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        #Create a mapping btw class_id and name
        rollNo = fx[:-4]
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        #Create Labels for the class
        target = np.array([rollNo for _ in range(data_item.shape[0])],dtype=object)
        labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))
testset = np.concatenate((face_dataset,face_labels),axis=1)

def knn_accuracy():
    print('Accuracy: ')
    for k in range(1,6):
        total_count = testset.shape[0]
        correct_count=0
        for (i,row) in enumerate(testset):
            test_face=row[:-1]
            pred_out=knn(testset,test_face,k)
            actual_out=row[-1]
            correct_count+=(actual_out==pred_out)
        print("For k = ",k,": ",(correct_count/total_count)*100)

knn_accuracy()