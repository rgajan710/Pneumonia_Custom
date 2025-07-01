import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ==========================
# GPU Config (Optional)
# ==========================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ==========================
# Image and Data Preparation
# ==========================
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    r'C:\Users\nmit\Desktop\archive\chest_xray\train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    r'C:\Users\nmit\Desktop\archive\chest_xray\val',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    r'C:\Users\nmit\Desktop\archive\chest_xray\test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# ==========================
# Model Blocks
# ==========================
def depthwise_separable_conv_block(x, filters, kernel_size=(3, 3), strides=1):
    """MobileNet-style block"""
    x = layers.SeparableConv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x

def residual_block(x, filters):
    """ResNet-style residual block"""
    shortcut = x
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

def vgg_block(x, filters, convs):
    """VGG-style stack of conv layers"""
    for _ in range(convs):
        x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    return x

def inception_module(x, filters):
    """Inception-style module with parallel paths"""
    path1 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    
    path2 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    path2 = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(path2)
    
    path3 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    path3 = layers.Conv2D(filters, (5, 5), padding='same', activation='relu')(path3)
    
    path4 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    path4 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(path4)
    
    return layers.concatenate([path1, path2, path3, path4], axis=-1)

def xception_block(x, filters):
    """Xception-style depthwise + residual"""
    shortcut = layers.Conv2D(filters, (1, 1), padding='same')(x)
    x = layers.SeparableConv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.SeparableConv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.add([shortcut, x])
    x = layers.BatchNormalization()(x)
    return x

# ==========================
# Model Definition
# ==========================
def build_hybrid_model():
    inputs = layers.Input(shape=(150, 150, 3))
    
    x = depthwise_separable_conv_block(inputs, 32)      # MobileNet
    x = vgg_block(x, 64, 2)                             # VGG
    x = residual_block(x, 64)                           # ResNet
    x = inception_module(x, 32)                         # Inception
    x = xception_block(x, 64)                           # Xception
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

model = build_hybrid_model()

# ==========================
# Model Training
# ==========================
epochs = 10
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# ==========================
# Save and Evaluate
# ==========================
model.save('hybrid_cnn_model.h5')

y_pred = model.predict(test_generator)
y_pred = np.round(y_pred).flatten()
y_true = test_generator.classes

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# ==========================
# Visualization
# ==========================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss Over Epochs')

plt.show()
