#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import cv2


# In[2]:


#Source paths for image folders
train_src_path = r'F:\IIT-J\ML\train-20230427T182925Z-001\\'
val_src_path = r'F:\IIT-J\ML\val-20230427T162630Z-001\\'


# In[3]:


# Define path to the folder containing the 'train' data
data_dir = train_src_path

# Image Sizing
img_size = (32, 32)

# Variable/lists for Images & labels
images = []
labels = []


# In[4]:


# Looping over each folder from starting from '0' to '9'
for label in range(10):
    folder_path = os.path.join(data_dir, 'train', str(label))
    # Looping over each image in the folder
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file_path.endswith(('.tiff','.bmp')):
            # Loading and resizing the image
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size)
            # Appending images and lables to the list
            images.append(img)
            labels.append(label)
            


# In[5]:


# Convertig list to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Saving the arrays in NumPy format
np.save('x_train.npy', images)
np.save('y_train.npy', labels)


# In[6]:


# Define path to the folder containing the 'val' data
data_dir_val = val_src_path
# Image Sizing
img_size_val = (32, 32)
# Variable/lists for Images & labels
images_val = []
labels_val = []
# Looping over each folder from starting from '0' to '9'
for label in range(10):
    folder_path = os.path.join(data_dir_val, 'val\\', str(label))
    # Looping over each image in the folder
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file_path.endswith(('.tiff','.bmp')):
            # Loading and resizing the image
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size_val)
            # Appending images and lables to the list
            images_val.append(img)
            labels_val.append(label)
# Convert the lists to NumPy arrays
images_val = np.array(images_val)
labels_val = np.array(labels_val)
# Save the arrays in NumPy format
np.save('x_test.npy', images_val)
np.save('y_test.npy', labels_val)


# In[7]:


#loading the dataset
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')


# In[8]:


# Test if images are loaded correctly
print(len(x_train))
print(len(x_test))
x_train[0].shape
x_train[0]
plt.matshow(x_train[0])
plt.matshow(x_train[999])
print(x_train.shape)
print(x_test.shape)
y_train
y_test
plt.matshow(x_test[150])


# In[8]:


# Simple NN Model
# creating a dense layer where each input is connected to each output, input count 1000 ouput count 10
# sigmoid is taken as activation function here
model = keras.Sequential([
 keras.layers.Flatten(),
 keras.layers.Dense(10, input_shape=(1024,),activation = 'sigmoid')
])
# compile the nn
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy']
)
# train the model
# 10 iteration are performed here
model.fit(x_train, y_train,epochs= 10, validation_data=(x_test, y_test))  


# In[9]:


# Observation  from the above iterations : we could see a better accuracy from the 2nd iteration onwards

# After scaling and dividing dat set by 255 we get:
x_train_scaled = x_train/255
x_test_scaled = x_test/255
model.fit(x_train_scaled, y_train,epochs= 10, validation_data=(x_test_scaled, y_test))


# In[10]:


# Observation :After scaling we get better accuracy result 


# In[12]:


# Now evaluating test data
model.evaluate(x_test_scaled,y_test)


# In[17]:


# Observation from above : result almost same as training dataset


# In[11]:


# 1st image prediction

#plt.matshow(x_test[0])

y_predicted = model.predict(x_test_scaled)
y_predicted[0]

# this showing the 10 results for the input '0', we need to look for the value which is max

print('Predicted Value is ',np.argmax(y_predicted[0]))

# test some more values

#plt.matshow(x_test[88])
print('Predicted Value is ',np.argmax(y_predicted[88]))

#plt.matshow(x_test[177])
print('Predicted Value is ',np.argmax(y_predicted[177]))




# In[12]:


# Building confusion matrix to see how our prediction appears

y_predicted_labels=[np.argmax(i) for i in y_predicted]

print(y_predicted_labels, len(y_predicted_labels))

conf_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
conf_mat


# In[10]:


import seaborn as sn
plt.figure(figsize = (10,10))
sn.heatmap(conf_mat,annot=True,fmt='d') 
plt.xlabel('Predicted')
plt.ylabel('Actual')


# In[12]:


# we need to modify our nn, we add some layers in the above model and different activation function
#1st Dense layer,the input is 32 x 32 i.e. 1024 neurons, which will give 10 output(numbers from 0 to 9)
# 2nd Dense layer,the input is 10 neurons from above layers
# Adding more layers will give better accuracy
model2 = keras.Sequential([
 keras.layers.Flatten(),
 keras.layers.Dense(1024,input_shape=(1024,), activation='relu'),
 keras.layers.Dense(10, activation='softmax')
])
# compile the nn
model2.compile(optimizer='adam',
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy']
 )
# train the model
# some 10 iterations done here
history = model2.fit(x_train_scaled, y_train,epochs= 10, validation_data=(x_test_scaled, y_test))


# In[16]:


# Observation : More layers means more accuracy howver more compliling time
model2.evaluate(x_test_scaled,y_test)


# In[13]:


y_predicted = model2.predict(x_test_scaled)
y_predicted[0]
y_predicted_labels=[np.argmax(i) for i in y_predicted]
print(y_predicted_labels, len(y_predicted_labels))
conf_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
conf_mat


# In[ ]:


plt.figure(figsize = (10,10))
sn.heatmap(conf_mat,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# In[1]:


# Observatoin : We see less errors in the updated model


# In[ ]:


# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[ ]:




