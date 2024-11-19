import numpy as np # type: ignore
import os
import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import VGG16, ResNet50 # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Dropout # type: ignore
from tensorflow.keras.models import Model, Sequential # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from sklearn.metrics import classification_report, confusion_matrix # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import pandas as pd # type: ignore


BATCH_SIZE = 32
EPOCHS = 100
IMG_SIZE = (224, 224)


train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_datagen.flow_from_directory('output_images',
                                               target_size=IMG_SIZE,
                                               batch_size=BATCH_SIZE,
                                               class_mode='categorical',
                                               subset='training')

val_data = train_datagen.flow_from_directory('output_images',
                                             target_size=IMG_SIZE,
                                             batch_size=BATCH_SIZE,
                                             class_mode='categorical',
                                             subset='validation')


def build_vgg16(input_shape, num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def build_resnet50(input_shape, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def build_alexnet(input_shape, num_classes):
    model = Sequential()
    model.add(tf.keras.layers.Conv2D(96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(tf.keras.layers.Conv2D(256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(tf.keras.layers.Conv2D(384, kernel_size=(3,3), activation='relu', padding="same"))
    model.add(tf.keras.layers.Conv2D(384, kernel_size=(3,3), activation='relu', padding="same"))
    model.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding="same"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def compile_and_train(model, train_data, val_data, model_name):
    
    checkpoint_filepath = f'{model_name}_checkpoint.keras'
    
    
    initial_epoch = 0
    if os.path.exists(checkpoint_filepath):
        print(f"Checkpoint found. Loading from {checkpoint_filepath}...")
        
        model.build(input_shape=(None, 224, 224, 3))

        model.load_weights(checkpoint_filepath)

        initial_epoch = 3 
    else:
        print(f"No checkpoint found. Training {model_name} from scratch.")

    checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, verbose=1)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, epochs=EPOCHS, initial_epoch=initial_epoch, 
                        validation_data=val_data, verbose=1, callbacks=[checkpoint])

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f'{model_name}_history.csv', index=False)

    model.save(f'{model_name}_bharatanatyam_classifier.h5')
    return history

def plot_metrics(history, model_name):
    epochs_range = range(EPOCHS)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']


    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title(f'{model_name} - Training and Validation Accuracy')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.legend(loc='upper right')

    plt.show()

# Model Training
input_shape = (224, 224, 3)
num_classes = train_data.num_classes

# VGG16 Training
vgg16_model = build_vgg16(input_shape, num_classes)
history_vgg16 = compile_and_train(vgg16_model, train_data, val_data, "VGG16")
plot_metrics(history_vgg16, "VGG16")

# ResNet50 Training
resnet_model = build_resnet50(input_shape, num_classes)
history_resnet = compile_and_train(resnet_model, train_data, val_data, "ResNet50")
plot_metrics(history_resnet, "ResNet50")

# AlexNet Training
alexnet_model = build_alexnet(input_shape, num_classes)
history_alexnet = compile_and_train(alexnet_model, train_data, val_data, "AlexNet")
plot_metrics(history_alexnet, "AlexNet")


def evaluate_model(model, val_data, model_name):
    val_data.reset()
    y_true = val_data.classes
    y_pred = model.predict(val_data)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print(f"{model_name} Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=val_data.class_indices.keys()))

evaluate_model(vgg16_model, val_data, "VGG16")
evaluate_model(resnet_model, val_data, "ResNet50")
evaluate_model(alexnet_model, val_data, "AlexNet")


def compare_accuracies(histories, model_names):
    epochs = range(20, 101, 20)
    data = {name: [history.history['val_accuracy'][e-1] for e in epochs] for name, history in zip(model_names, histories)}
    df = pd.DataFrame(data, index=epochs)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, markers=True)
    plt.title('Validation Accuracy Comparison at Epochs 20, 40, 60, 80, 100')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend(loc='lower right')
    plt.show()

compare_accuracies([history_vgg16, history_resnet, history_alexnet], ['VGG16', 'ResNet50', 'AlexNet'])
