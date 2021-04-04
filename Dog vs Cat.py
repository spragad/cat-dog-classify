import numpy as np
import os      
import cv2                                           
from tqdm import tqdm
from sklearn.utils import shuffle
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import metrics
import os

class_names = ['dogs', 'cats']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)
IMAGE_SIZE = (200, 200)

datasets=['dataset/training_set','dataset/test_set']

output=[]
def load_data():
    for dataset in datasets:
       images = []
       labels = []
       print("\n Loading {}".format(dataset))
       
       # Iterate through each folder corresponding to a category
       for folder in os.listdir(dataset):
           label = class_names_label[folder]
           
           # Iterate through each image in our folder
           for file in tqdm(os.listdir(os.path.join(dataset, folder))):
               
               # Get the path name of the image
               img_path = os.path.join(os.path.join(dataset, folder), file)
               
               # Open and resize the img
               image = cv2.imread(img_path)
               image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
               image = cv2.resize(image, IMAGE_SIZE)
               
               # Append the image and its corresponding label to the output
               images.append(image)
               labels.append(label)
               
       images = np.array(images, dtype = 'float32')
       labels = np.array(labels, dtype = 'int32')  
       
       output.append((images, labels))

    return output  

(train_images, train_labels), (test_images, test_labels) = load_data()

train_images, train_labels = shuffle(train_images, train_labels, random_state=25)

n_train = train_labels.shape[0]
n_test = test_labels.shape[0]

print ("Number of training examples: {}".format(n_train))
print ("Number of testing examples: {}".format(n_test))
print ("Each image is of size: {}".format(IMAGE_SIZE))
      

_, train_counts = np.unique(train_labels, return_counts=True)
_, test_counts = np.unique(test_labels, return_counts=True)
pd.DataFrame({'train': train_counts,
                    'test': test_counts}, 
             index=class_names
            ).plot.bar()
plt.show()

train_images = train_images / 255.0 
test_images = test_images / 255.0

def display_examples(class_names, images, labels):
    fig = plt.figure(figsize=(10,10))
    fig.suptitle("Some examples of images of the dataset", fontsize=16)
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()

display_examples(class_names, train_images, train_labels)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', input_shape = (200, 200, 3)), 
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    
    
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    
    
    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    
    
    tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.35),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['acc'])

print(model.summary())

mod2 = model.fit(train_images, train_labels, batch_size=128, verbose=1, epochs=20, validation_split = 0.2)

def plot_accuracy_loss(history):

    fig = plt.figure(figsize=(10,5))

    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['acc'],'bo--', label = "acc")
    plt.plot(history.history['val_acc'], 'ro--', label = "val_acc")
    plt.title("train_acc vs val_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    # Plot loss function
    plt.subplot(222)
    plt.plot(history.history['loss'],'bo--', label = "loss")
    plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")

    plt.legend()
    plt.show()
    
plot_accuracy_loss(mod2)

model.evaluate(test_images, test_labels)

result = np.round(model.predict(test_images),0)

classes=metrics.classification_report(test_labels,result,target_names=class_names)

print(classes)

#Saving a Keras model (73%)
model.save('Trained Model/')

#Loading the model back:

model = tf.keras.models.load_model('Trained Model/')

