#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import cv2
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten,GlobalAveragePooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


#Source paths for image folders
train_val_src = r'F:\IIT-J\ML\\'train_val"
test_src = r"F:\\IIT-J\ML\\test"
train_path_src = r"F:\\ IIT-J\ML\\train_val.csv”
train_val_labels = pd.read_csv(train_path_src)


# In[3]:


# loading training datasets

images = [] labels = []

for filename in os.listdir(train_val_src):
if filename.endswith('.png'):
# Load the images and resize them to (128, 128) with 3 color channels
img = cv2.imread(os.path.join(train_val_src, filename)) img = cv2.resize(img, (128, 128))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#  img = Image.open(os.path.join(train_val_dir, filename))
img_array = np.array(img)
# Append the array to the list of images
images.append(img_array) labels.append(filename)

# Convert the string to numerical
le = LabelEncoder()
labels = le.fit_transform(labels)

# Converting the lists to NumPy arrays
images = np.array(images) labels = np.array(labels)
# Save the arrays in NumPy format
np.save('x_train.npy', images) np.save('y_train.npy', labels)
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')



# In[4]:


x_train.shape


# In[5]:


x_train[:5] y_train[:5]


# In[6]:


# loading test dataset in numpy array

images = [] labels = []

for filename in os.listdir(test_src):
if filename.endswith('.png'):
# Loadig image and resizing
img = cv2.imread(os.path.join(test_src, filename)) img = cv2.resize(img, (128, 128))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    img = Image.open(os.path.join(test_src, filename))
img_array = np.array(img)
# Append the array to the list of images
images.append(img_array)
labels.append(filename)

# String to numericals
le = LabelEncoder()
labels = le.fit_transform(labels)

# Convert the lists to NumPy arrays
images = np.array(images) labels = np.array(labels)

# Save the arrays in NumPy format
np.save('x_test.npy', images) np.save('y_test.npy', labels)

x_test = np.load('x_test.npy') y_test = np.load('y_test.npy')


# In[7]:


x_test.shape


# In[8]:


# Verify images loaded
plt.figure(figsize = (10,2)) plt.imshow(x_train[10])
plt.imshow(x_train[208]) plt.imshow(x_train[444])


# In[9]:


# Class definition from the images
image_classes = ['line', 'dot_line', 'hbar_categorical', 'vbar_categorical', 'pie'] image_classes[0]

# Mapping the categories to the labels array i.e y_train
label_map = {'line': 0, 'dot_line': 1, 'hbar_categorical': 2, 'vbar_categorical': 3, 'pie': 4} y_train = np.array([label_map[label] for label in train_val_labels['type']])
y_train
y_train.shape y_test.shape


# In[10]:


#mapping the lables from csv to the images 

def image_sample(x, y, index):
plt.figure(figsize = (10,2)) plt.imshow(x[index])
#	image_label = train_val_labels.iloc[index]['type'] #	plt.xlabel(image_label)
plt.xlabel(image_classes[y[index]])


# In[ ]:


image_sample(x_train,y_train,0)
image_sample(x_train,y_train,208) 
image_sample(x_train,y_train,444)


# In[11]:


# we have mapped the corresponding labels to the image


# In[12]:


# normalize the image # x_train[0]/255
x_train=x_train /255
x_test=x_train /255


# In[ ]:


# label for train data from csv file

y_train_index = train_val_labels['image_index']
y_train_type = train_val_labels['type']


# In[ ]:


y_train_type[:5]


# In[ ]:


# simple nn to test first

#model architecture
model = Sequential([
Flatten(input_shape=(128,128,3)), Dense(3000, activation='relu'),
Dense(1000, activation='relu'), Dense(5, activation='softmax')
])

# Compile the model
model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10)


# In[ ]:


# Split the training images and labels into training and validation sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


# In[ ]:


model.evaluate(x_test,y_test)


# In[ ]:


y_pred = model.predict(x_test) y_pred
y_pred_classes = [np.argmax(ele) for ele in y_pred]
# print("classificaton report : \n",classification_report(y_test,y_pred_classes))


# In[ ]:


#verify that they loaded correctly
print("Train Images Shape:", x_train.shape) print("Train Labels Shape:", y_train.shape) print("Test Images Shape:", x_test.shape)
print("Test Labels Shape:", y_test.shape)


# In[ ]:


# Update architecture to cnn
cnn_model = Sequential([
Conv2D(filters=16 ,kernel_size=(3,3), activation='relu', input_shape=(128,128,3)), MaxPooling2D(pool_size=(2,2)),
Conv2D(32, (3,3), activation='relu'),
MaxPooling2D(pool_size=(2,2)),
Conv2D(64, (3,3), activation='relu'), MaxPooling2D(pool_size=(2,2)),
Flatten(),
Dense(128, activation='relu'), Dense(5, activation='softmax')
])

# Compile the model
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = cnn_model.fit(x_train, y_train, batch_size=1000, epochs=50,validation_data=(x_test, y_test))

# Plot the obtained loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss']) plt.title('Model Loss')
plt.ylabel('Loss') plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right') plt.show()


# In[ ]:


image_sample(x_test,y_test,1) 
image_sample(x_test,y_test,50) 
image_sample(x_test,y_test,25) 
image_sample(x_test,y_test,30)


# In[ ]:


y_pred = cnn_model.predict(x_test) 
y_pred[:5]


# In[ ]:


y_classes = [np.argmax(element) 
    for element in y_pred] y_classes[:5]


# In[ ]:


y_test[:5]


# In[ ]:


image_sample(x_test,y_test,15) #actual
image_classes[y_classes[15]] #predicted


# In[ ]:


# Few values are not matching


# In[ ]:


print("classification report: \n", classification_report(y_test,y_classes))


# In[ ]:


# Generate the
conf_mat = confusion_matrix(y_test, y_classes)

print('Confusion Matrix:') print(conf_mat)


# In[ ]:


#confusion matrix
import seaborn as sn
plt.figure(figsize = (10,10))
sn.heatmap(conf_mat,annot=True,fmt='d') plt.xlabel('Predicted')
plt.ylabel('Actual')


# In[ ]:


# for 50 iterations, accuracy is good


# In[ ]:


from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the pre-trained model
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[ ]:


# Replace the final layer with new layer
x = vgg16_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)

predictions = Dense(5, activation='softmax')(x)
pt_model = tf.keras.Model(inputs=vgg16_model.input, outputs=predictions)


# In[ ]:


# 
for layer in pt_model.layers: layer.trainable = False


# In[ ]:


# Compile the model with categorical crossentropy loss and Adam optimizer
pt_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


# Print the summary of the model architecture
pt_model.summary()


# In[ ]:


# Setting up datagenerators for image augmentation and feeding data to the model

train_datagen = ImageDataGenerator( rescale=1./255,
rotation_range=20,
width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True, fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)


# In[ ]:


# flow method helpsin generating of augmented data
train_generator = train_datagen.flow(x_train, y_train, batch_size=32) 
test_generator = train_datagen.flow(x_test, y_test, batch_size=32)


# In[ ]:


# Train the model 

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True) 
history = pt_model.fit(train_generator, epochs=100, validation_data=test_generator, callbacks=[es])


# In[ ]:




