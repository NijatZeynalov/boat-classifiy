#import modules
import os
import zipfile
import random
import tensorflow as tf
import shutil
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd
import numpy as np
import os
from shutil import copyfile

os.mkdir('split_boats')
os.mkdir('split_boats//training')
os.mkdir('split_boats//testing')

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    file_contents=os.listdir(SOURCE)
    files=[]
    for i in range(len(file_contents)):
        if(os.path.getsize(SOURCE+file_contents[i])==0):
            print(file_contents[i]+" is zero length, so ignoring")
        else:
            files.append(file_contents[i])
    training_length=int(len(files)*SPLIT_SIZE)
    testing_length=int(len(files)-training_length)
    shuffled_set=random.sample(files,len(files))
    training_set=shuffled_set[0:training_length]
    testing_set=shuffled_set[-testing_length:]
    for file in training_set:
        copyfile(SOURCE+file,TRAINING+file)
    for file in testing_set:
        copyfile(SOURCE+file,TESTING+file)

patth = 'boats/'
for i in os.listdir(patth):
    os.mkdir('split_boats///training/'+i)
    os.mkdir('split_boats//testing/'+ i)

    print(i+' consists of: '+str(len(os.listdir(patth+str(i)))))

    SOURCE_DIR = patth+"/"+i+"//"
    TRAINING_DIR = "split_boats//training/"+i+"/"
    TESTING_DIR = "split_boats//testing/"+i+"/"
    split_size = .9
    split_data(SOURCE_DIR, TRAINING_DIR, TESTING_DIR, split_size)