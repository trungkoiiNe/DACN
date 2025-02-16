import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

class FaceRecognitionTrainer:
    def __init__(self):
        self.input_shape = (160, 160, 3)
        self.data_dir = 'face_data'
        
    def load_data(self):
        X = []
        y = []
        
        for employee_id in os.listdir(self.data_dir):
            employee_dir = os.path.join(self.data_dir, employee_id)
            if not os.path.isdir(employee_dir):  # Skip if not a directory
                continue
                
            for img_name in os.listdir(employee_dir):
                # Skip non-image files and metadata
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(employee_dir, img_name)
                img = cv2.imread(img_path)
                
                # Validate image was loaded properly
                if img is None:
                    print(f"Warning: Could not load image {img_path}")
                    continue
                    
                try:
                    img = cv2.resize(img, (160, 160))
                    X.append(img)
                    y.append(int(employee_id))
                except Exception as e:
                    print(f"Error processing image {img_path}: {str(e)}")
                    continue
                
        if len(X) == 0:
            raise ValueError("No valid images found in the dataset")
                
        return np.array(X), np.array(y)
    
    def create_model(self, num_classes):
        base_model = MobileNetV2(input_shape=self.input_shape,
                               include_top=False,
                               weights='imagenet')
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        return model
    
    def train(self):
        X, y = self.load_data()
        num_classes = len(np.unique(y))
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Normalize the data
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        
        # Create and compile the model
        model = self.create_model(num_classes)
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Train the model
        history = model.fit(X_train, y_train,
                          batch_size=32,
                          epochs=20,
                          validation_data=(X_test, y_test))
        
        # Save the model
        model.save('face_recognition_model.h5')
        return history

if __name__ == "__main__":
    trainer = FaceRecognitionTrainer()
    history = trainer.train()
