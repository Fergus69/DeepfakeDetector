
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from tensorflow import keras
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf
import os
import cv2
# Height and width refer to the size of the image
# Channels refers to the amount of color channels (red, green, blue)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def get_images_and_labels(folder_path):
    # Initialize empty lists to store image data and corresponding labels
    images = []
    labels = []

    # Iterate over all subfolders in the given folder
    for subfolder in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, subfolder)):
            # Iterate over all images in the subfolder
            for img_file in os.listdir(os.path.join(folder_path, subfolder)):
                # Load the image using OpenCV
                img = cv2.imread(os.path.join(folder_path, subfolder, img_file))
                img = cv2.resize(img, (256, 256))
                # Convert the image to a NumPy array
                img_array = np.array(img)
                # Append the image data to the list
                images.append(img_array)
                # Get the label from the folder name
                if subfolder=='DeepFake':
                    label = 0
                else:
                    label = 1
                # Append the label to the list
                labels.append(label)

    # Convert the lists to NumPy arrays
    images_array = np.array(images)
    labels_array = np.array(labels)

    return images_array, labels_array

image_dimensions = {'height':256, 'width':256, 'channels':3}
# Create a Classifier class

class Classifier:
    def __init__():
        self.model = 0
    
    def predict(self, x):
        return self.model.predict(x)
    
    def fit(self, generator, epochs=1):
        # Adaugarea unui callback pentru oprire timpurie
        early_stopping = EarlyStopping(monitor='accuracy', patience=5, verbose=1, mode='max')
        return self.model.fit(generator, epochs=epochs, steps_per_epoch=len(generator.labels)//generator.batch_size, callbacks=[early_stopping])
    
    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)
    
    def load(self, path):
        self.model.load_weights(path)
# Create a MesoNet class using the Classifier

class Meso4(Classifier):
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate)
        self.model.compile(optimizer = optimizer,
                           loss = 'mean_squared_error',
                           metrics = ['accuracy'])
    
    def init_model(self): 
        x = Input(shape = (image_dimensions['height'],
                           image_dimensions['width'],
                           image_dimensions['channels']))
        
        x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        
        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        x5 = Conv2D(32, (3, 3), padding='same', activation = 'relu')(x4)
        x5 = BatchNormalization()(x5)
        x5 = MaxPooling2D(pool_size=(4, 4), padding='same')(x5)

        x6 = Conv2D(32, (3, 3), padding='same', activation = 'relu')(x5)
        x6 = BatchNormalization()(x6)
        x6 = MaxPooling2D(pool_size=(4, 4), padding='same')(x6)


        y = Flatten()(x6)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return Model(inputs = x, outputs = y)
# Instantiate a MesoNet model with pretrained weights
meso = Meso4()
#meso.load('./aplicatie/configurations/original/mesonet.h5')
# Prepare image data

# Rescaling pixel values (between 1 and 255) to a range between 0 and 1
dataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Instantiating generator to feed images through the network
generator = dataGenerator.flow_from_directory(
    './algoritm/data/',
    target_size=(256, 256),
    batch_size=20,
    class_mode='binary')

# Checking class assignment
generator.class_indices
# '.ipynb_checkpoints' is a *hidden* file Jupyter creates for autosaves
# It must be removed for flow_from_directory to work.


# Rendering image X with label y for MesoNet
X, y = next(generator)


history=meso.fit(generator , epochs=15)


# Evaluating prediction
print(f"Predicted likelihood: {meso.predict(X)[0][0]:.4f}")
print(f"Actual label: {int(y[0])}")
print(f"\nCorrect prediction: {round(meso.predict(X)[0][0])==y[0]}")
# Showing image
plt.imshow(np.squeeze(X[0]))


# Creating separate lists for correctly classified and misclassified images
correct_real = []
correct_real_pred = []

correct_deepfake = []
correct_deepfake_pred = []

misclassified_real = []
misclassified_real_pred = []

misclassified_deepfake = []
misclassified_deepfake_pred = []
# Generating predictions on validation set, storing in separate lists
for i in range(len(generator.labels)):
    
    # Loading next picture, generating prediction
    X, y = next(generator)
    pred = meso.predict(X)[0][0]
    
    # Sorting into proper category
    if round(pred)==y[0] and y[0]==1:
        correct_real.append(X)
        correct_real_pred.append(pred)
    elif round(pred)==y[0] and y[0]==0:
        correct_deepfake.append(X)
        correct_deepfake_pred.append(pred)
    elif y[0]==1:
        misclassified_real.append(X)
        misclassified_real_pred.append(pred)
    else:
        misclassified_deepfake.append(X)
        misclassified_deepfake_pred.append(pred)   
        
    # Printing status update
    if i % 1000 == 0:
        print(i, ' predictions completed.')
    
    if i == len(generator.labels)-1:
        print("All", len(generator.labels), "predictions completed")

# Add these lines after the last loop in the code
total_predictions = len(correct_real) + len(correct_deepfake) + len(misclassified_real) + len(misclassified_deepfake)

correct_predictions = len(correct_real) + len(correct_deepfake)

percentage_correct = (correct_predictions / total_predictions) * 100

print(f"Percentage of correct guesses: {percentage_correct:.2f}%")
def plotter(images,preds):
    fig = plt.figure(figsize=(16,9))
    subset = np.random.randint(0, len(images)-1, 12)
    for i,j in enumerate(subset):
        fig.add_subplot(3,4,i+1)
        plt.imshow(np.squeeze(images[j]))
        plt.xlabel(f"Model confidence: \n{preds[j]:.4f}")
        plt.tight_layout()
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
    plt.show()
    return

# Save the trained model using the HDF5 format
meso.model.save('./configurations/mesonet.h5', save_format='h5')

# Optionally, save using the TensorFlow SavedModel format
# meso.model.save('mesonet', save_format='tf')

#plotter(correct_real, correct_real_pred)

#plotter(misclassified_real, misclassified_real_pred)

#plotter(correct_deepfake, correct_deepfake_pred)

#plotter(misclassified_deepfake, misclassified_deepfake_pred)
#plt.show(block=True)
# Crearea graficului de acuratețe
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Acuratețe', marker='x')
for i, acc in enumerate(history.history['accuracy']):
    plt.text(i, acc, f"{acc:.2f}", ha='center', va='bottom')

plt.title('Acuratețea pe epoci')
plt.xlabel('Epoci')
plt.ylabel('Acuratețe')
plt.legend()
plt.grid(True)  # Adăugarea unei grile pentru ușurința citirii
plt.show()