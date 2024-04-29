import os
import shutil
import random
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, BatchNormalization, Activation, MaxPooling2D, \
    GlobalAveragePooling2D, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# Define the initial path
initial_path = '/kaggle/input/aaaaaaaa/pfa'

# Define the new top-level directory names
directories = ['train', 'val', 'test']

# Define the subdirectory names for each top-level directory
subdirectories = ['Cherry Coke', 'Fanta', 'Minute Maid', 'Nestea', 'Sprite', 'aquarius',  'boga', 'irnbru',
                  'j20', 'prime','rio' ,'barrcola' ,'rubicon', 'tango']

# Define the proportions for train, validation, and test sets
props = [0.7, 0.2, 0.1]

# Loop over the top-level directories and subdirectories, and create the corresponding directories
for directory in directories:
    for subdirectory in subdirectories:
        path = os.path.join(initial_path, subdirectory)
        files = os.listdir(path)
        random.shuffle(files)
        total = len(files)
        train_end = int(total * props[0])
        val_end = train_end + int(total * props[1])
        if directory == 'train':
            new_files = files[:train_end]
        elif directory == 'val':
            new_files = files[train_end:val_end]
        else:
            new_files = files[val_end:]
        new_path = os.path.join(directory, subdirectory)
        os.makedirs(new_path, exist_ok=True)
        for file in new_files:
            old_file_path = os.path.join(path, file)
            new_file_path = os.path.join(new_path, file)
            shutil.copy(old_file_path, new_file_path)

# Define the image dimensions and batch size
img_height = 299
img_width = 299
batch_size = 32

# Define the data generators for the train, validation, and test sets
train_datagen = ImageDataGenerator(
    rescale=1 / 255.,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1 / 255.)

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb'
)

val_generator = val_datagen.flow_from_directory(
    'val',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb'
)

def create_model(input_shape, classes):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2), use_bias=False, padding='same')(x)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = Add()([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2), use_bias=False, padding='same')(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = Add()([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2), use_bias=False, padding='same')(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = Add()([x, residual])
    for _ in range(8):
        residual = x
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Add()([x, residual])
    residual = Conv2D(1024, (1, 1), strides=(2, 2), use_bias=False, padding='same')(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = Add()([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Assuming you have train_generator and val_generator defined somewhere

input_shape = (299, 299, 3)
num_classes = 14

model = create_model(input_shape, num_classes)
model.compile(optimizer=Adam(learning_rate=0.00b  02), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

epochs = 200
batch_size = 32

train_steps_per_epoch = train_generator.samples // batch_size
val_steps_per_epoch = val_generator.samples // batch_size

# Define the callback to reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-9)

# Define the path where you want to save the best model
checkpoint_path = 'best_model.keras'

# Define the callback to save the best model based on validation loss
checkpoint = ModelCheckpoint(checkpoint_path,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

# Training the model with the callback
history = model.fit(train_generator,
                    steps_per_epoch=train_steps_per_epoch,
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=val_steps_per_epoch,
                    callbacks=[reduce_lr, checkpoint])

# After training the model, you can use it to make predictions
predictions = model.predict(val_generator)

# Get class indices
class_indices = train_generator.class_indices

# Inverse the dictionary to map indices to class names
indices_to_classes = {v: k for k, v in class_indices.items()}

# Define the list of non-boycotted classes
non_boycott_classes = ['Cherry Coke', 'Fanta', 'Minute Maid', 'Nestea', 'Sprite', 'aquarius']

# Iterate through predictions and determine boycott status
for i, prediction in enumerate(predictions):
    class_index = np.argmax(prediction)
    class_name = indices_to_classes[class_index]
    boycott_status = "boycott" if class_name not in non_boycott_classes else "non boycott"
    print(f"Prediction {i+1}: Class: {class_name}, Boycott Status: {boycott_status}")



from tensorflow.keras.models import load_model
# Load the trained model
model = load_model('model/model.h5')
# Plotting the training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


#Confusion Matrix
model = load_model('model/model.h5')
validation_data_dir = 'val'
img_height = 299
img_width = 299
batch_size = 32

# Créer un générateur de données pour les données de validation
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Ne pas mélanger les données pour l'évaluation
)

# Obtenir les prédictions du modèle sur les données de validation
predictions = model.predict(val_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes

# Générer la matrice de confusion
conf_matrix = confusion_matrix(y_true, y_pred)

# Afficher la matrice de confusion
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.colorbar()
plt.title('Matrice de confusion')
plt.xlabel('Classe prédite')
plt.ylabel('Classe réelle')
plt.xticks(ticks=np.arange(len(val_generator.class_indices)), labels=val_generator.class_indices.keys(), rotation=45)
plt.yticks(ticks=np.arange(len(val_generator.class_indices)), labels=val_generator.class_indices.keys())
plt.tight_layout()
plt.show()
print(classification_report(y_true, y_pred, target_names=val_generator.class_indices.keys()))

