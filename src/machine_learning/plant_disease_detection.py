import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def build_model(input_shape=(128,128,3), num_classes=38):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_disease_model(train_dir=r"C:\Users\JEEVITH\OneDrive\Documents\precision-agriculture\data\PlantVillage\train", val_dir=r"C:\Users\JEEVITH\OneDrive\Documents\precision-agriculture\data\PlantVillage\val", save_path=r"C:\Users\JEEVITH\OneDrive\Documents\precision-agriculture\models\plant_disease_model.h5"):
    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_directory(train_dir, target_size=(128,128), batch_size=32, class_mode='categorical')
    val_gen = datagen.flow_from_directory(val_dir, target_size=(128,128), batch_size=32, class_mode='categorical')

    model = build_model(num_classes=len(train_gen.class_indices))

    history = model.fit(train_gen, validation_data=val_gen, epochs=10)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"✅ Model saved at {save_path}")

    return history, model, train_gen.class_indices

if __name__ == "__main__":
    train_disease_model()
