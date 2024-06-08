
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping,LearningRateScheduler
from tensorflow import keras
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import Callback
import tensorflow as tf
import os
import cv2
# Height and width refer to the size of the image
# Channels refers to the amount of color channels (red, green, blue)

class StepLogger(Callback):
    def __init__(self, every=5):
        self.every = every
        self.accuracy = []
        self.loss = []

    def on_train_begin(self, logs=None):
        self.accuracy = []
        self.loss = []

    def on_batch_end(self, batch, logs=None):
        if batch % self.every == 0:
            self.accuracy.append(logs['accuracy'])
            self.loss.append(logs['loss'])

step_logger = StepLogger(every=5)

class ValidationAccuracyLogger(Callback):
    def __init__(self):
        super().__init__()
        self.validation_accuracy = []
        self.steps_per_epoch = 0
        self.epoch=0

    def set_params(self, params):
        self.steps_per_epoch = params['steps']

    def on_epoch_begin(self, epoch, logs=None):
        self.validation_logs = [False, False, False, False]
        self.epoch=self.epoch+1

    def on_batch_end(self, batch, logs=None):
        validation_intervals = [
            self.steps_per_epoch // 4,
            self.steps_per_epoch // 2,
            3 * self.steps_per_epoch // 4,
            self.steps_per_epoch
        ]

        for i, interval in enumerate(validation_intervals):
            if not self.validation_logs[i] and batch >= interval:
                accuracy = validation_func()
                self.validation_accuracy.append(accuracy / 100)
                print(f'Validation accuracy at interval {i+1} of epoch {self.epoch} is {accuracy:.2f}%')
                self.validation_logs[i] = True

    def on_epoch_end(self, epoch, logs=None):
        # Optionally, validate at the end of the epoch if not already done
        if not self.validation_logs[-1]:
            accuracy = validation_func()
            self.validation_accuracy.append(accuracy / 100)
            print(f'Validation accuracy at end of epoch {self.epoch} is {accuracy:.2f}%')

def lr_schedule(epoch,initial_lr=0.001):
    initial_lr = 0.001
    if epoch < 5:
        return initial_lr
    elif epoch < 10:
        return initial_lr / 2
    else:
        return initial_lr / 4

lr_scheduler = LearningRateScheduler(lr_schedule)

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
    
    def fit(self, train_dataset, val_dataset, steps_per_epoch, validation_steps, epochs=1):
    # Add a callback for early stopping
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='max',restore_best_weights=True,)
    
        # Train the model using the training dataset and validate using the validation dataset
        return self.model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            shuffle=True,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            callbacks=[early_stopping,step_logger,validation_logger,lr_scheduler]
            ), epochs

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
dataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Instantiating generator to feed images through the network
train_generator = dataGenerator.flow_from_directory(
    './algoritm/data/',
    target_size=(256, 256),
    batch_size=40,
    class_mode='binary',
    subset='training',
    shuffle=True)  # Enable shuffling for the training generator

validation_generator = dataGenerator.flow_from_directory(
    './algoritm/data/',
    target_size=(256, 256),
    batch_size=40,
    class_mode='binary',
    subset='validation',
    shuffle=True)  # Enable shuffling for the validation generator

train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 256, 256, 3], [None])
)
train_dataset.shuffle(buffer_size=len(train_generator.labels),reshuffle_each_iteration=True)  # Shuffle the dataset with a buffer size

validation_dataset = tf.data.Dataset.from_generator(
    lambda: validation_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 256, 256, 3], [None])
)

validation_dataset.shuffle(buffer_size=len(validation_generator.labels),reshuffle_each_iteration=True)  # Shuffle the dataset with a buffer size

# Apply the repeat() function to the datasets
#train_dataset = train_dataset.repeat()
#validation_dataset = validation_dataset.repeat()

# Calculate steps per epoch
steps_per_epoch = len(train_generator.labels) // train_generator.batch_size
validation_steps = len(validation_generator.labels) // validation_generator.batch_size
# Rendering image X with label y for MesoNet
X, y = next(validation_generator)
validation_logger=ValidationAccuracyLogger()
def validation_func():
    # Creating separate lists for correctly classified and misclassified images
    correct_real = []
    correct_real_pred = []

    correct_deepfake = []
    correct_deepfake_pred = []

    misclassified_real = []
    misclassified_real_pred = []

    misclassified_deepfake = []
    misclassified_deepfake_pred = []

    # Iterate over the validation dataset
    total_samples = 0
    correct_predictions = 0
    
    for X, y in validation_dataset.take(validation_steps):
        preds = meso.predict(X)
        total_samples += len(y)
        
        for j in range(len(y)):
            pred = preds[j][0]

            # Sorting into proper category
            if round(pred) == y[j] and y[j] == 1:
                correct_real.append(X[j])
                correct_real_pred.append(pred)
                correct_predictions += 1
            elif round(pred) == y[j] and y[j] == 0:
                correct_deepfake.append(X[j])
                correct_deepfake_pred.append(pred)
                correct_predictions += 1
            elif y[j] == 1:
                misclassified_real.append(X[j])
                misclassified_real_pred.append(pred)
            else:
                misclassified_deepfake.append(X[j])
                misclassified_deepfake_pred.append(pred)

        # Printing status update
        if total_samples % 100 == 0:  # Print status every 100 samples
            print(f"{total_samples} predictions completed.")

    print(f"All {total_samples} predictions completed")

    # Calculate percentage of correct predictions
    percentage_correct = (correct_predictions / total_samples) * 100

    print(f"Percentage of correct guesses: {percentage_correct:.2f}%")
    return percentage_correct
history,epochs=meso.fit(train_dataset, validation_dataset, steps_per_epoch, validation_steps, epochs=40)
last_epoch=len(history.history['accuracy'])

# Evaluating prediction
print(f"Predicted likelihood: {meso.predict(X)[0][0]:.4f}")
print(f"Actual label: {int(y[0])}")
print(f"\nCorrect prediction: {round(meso.predict(X)[0][0])==y[0]}")


# Save the trained model using the HDF5 format
meso.model.save('./configurations/mesonet.h5', save_format='h5')


# Plotting accuracy from StepLogger and ValidationAccuracyLogger
training_x_values = range(0 + step_logger.every, last_epoch * steps_per_epoch + step_logger.every, step_logger.every)
validation_x_values = range(0 + steps_per_epoch // 4, last_epoch * steps_per_epoch + steps_per_epoch, steps_per_epoch // 4)


plt.figure(figsize=(10, 5))
plt.plot(training_x_values[:len(step_logger.accuracy)], step_logger.accuracy, label='Acuratețea la antrenare', marker='x')
plt.plot(validation_x_values[:len(validation_logger.validation_accuracy)], validation_logger.validation_accuracy, label='Acuratețea la validare', marker='^')
for i in range(0,last_epoch*steps_per_epoch,steps_per_epoch):
    plt.axvline(x=i, color='red', linestyle='--',linewidth=0.4)
    plt.text(i, min(validation_logger.validation_accuracy), f'Epoca {i // steps_per_epoch + 1}', color='red', rotation=90, verticalalignment='bottom')
plt.xlabel('Pași')
plt.ylabel('Acuratețea')
plt.title('Acuratețea la antrenare și validare')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png')
plt.show()

# Plotting loss from StepLogger
plt.figure(figsize=(10, 5))
plt.plot(range(0+step_logger.every,last_epoch*steps_per_epoch+step_logger.every,step_logger.every),step_logger.loss, label='Pierderea pe pași',marker='x')
for i in range(0,last_epoch*steps_per_epoch,steps_per_epoch):
    plt.axvline(x=i, color='red', linestyle='--',linewidth=0.4)
    plt.text(i, (max(step_logger.accuracy) + min(step_logger.accuracy))/2, f'Epoca {i // steps_per_epoch + 1}', color='red', rotation=90, verticalalignment='bottom')
plt.xlabel('Pași')
plt.ylabel('Pierderea')
plt.title('Pierderea la antrenare')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')
plt.show()