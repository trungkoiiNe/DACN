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
        valid_employees = []
        
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory '{self.data_dir}' does not exist")
        
        # First pass: count valid images per employee
        employee_image_counts = {}
        for employee_id in os.listdir(self.data_dir):
            employee_dir = os.path.join(self.data_dir, employee_id)
            if not os.path.isdir(employee_dir):
                continue
            
            image_count = len([f for f in os.listdir(employee_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            if image_count > 0:
                employee_image_counts[employee_id] = image_count
                valid_employees.append(employee_id)
        
        if len(valid_employees) < 2:
            raise ValueError(
                f"Found only {len(valid_employees)} employees with images. "
                f"Need at least 2 employees. "
                f"Current employees: {', '.join(valid_employees)}"
            )
        
        # Second pass: load images for valid employees
        for employee_id in valid_employees:
            employee_dir = os.path.join(self.data_dir, employee_id)
            for img_name in os.listdir(employee_dir):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                img_path = os.path.join(employee_dir, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    img = cv2.resize(img, (160, 160))
                    img = img.astype('float32')
                    img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
                    X.append(img)
                    y.append(int(employee_id))
                except Exception as e:
                    print(f"Error processing image {img_path}: {str(e)}")
                    continue
        
        if len(X) == 0:
            raise ValueError("No valid images found in the dataset")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        print(f"Loaded {len(X)} images from {len(valid_employees)} employees")
        for emp_id in valid_employees:
            print(f"Employee {emp_id}: {employee_image_counts[emp_id]} images")
        
        return X, y
    
    def validate_training_data(self):
        """Check if there's enough data for training"""
        try:
            if not os.path.exists(self.data_dir):
                return False, "No training data directory found"
            
            # Count valid employees with images
            valid_employees = []
            for employee_id in os.listdir(self.data_dir):
                employee_dir = os.path.join(self.data_dir, employee_id)
                if not os.path.isdir(employee_dir):
                    continue
                
                image_count = len([f for f in os.listdir(employee_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if image_count > 0:
                    valid_employees.append(employee_id)
            
            if len(valid_employees) < 2:
                return False, (f"Found only {len(valid_employees)} employees with images. "
                             f"Need at least 2 employees. "
                             f"Current employees: {', '.join(valid_employees)}")
            
            return True, f"Found {len(valid_employees)} employees with images. Ready for training."
            
        except Exception as e:
            return False, f"Error validating training data: {str(e)}"
    
    def create_model(self, num_classes):
        base_model = MobileNetV2(input_shape=self.input_shape,
                               include_top=False,
                               weights='imagenet')
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Fine-tune the top layers
        for layer in base_model.layers[-30:]:
            layer.trainable = True
            
        return model
    
    def train(self):
        try:
            X, y = self.load_data()
            num_classes = len(np.unique(y))
            
            # Convert labels to categorical
            y = tf.keras.utils.to_categorical(y, num_classes)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
            
            # Create and compile the model
            model = self.create_model(num_classes)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
            
            # Add callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
            ]
            
            # Train the model
            history = model.fit(X_train, y_train,
                              batch_size=32,
                              epochs=20,
                              validation_data=(X_test, y_test),
                              callbacks=callbacks)
            
            # Save the model
            model.save('face_recognition_model.h5')
            return history
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            raise

if __name__ == "__main__":
    trainer = FaceRecognitionTrainer()
    history = trainer.train()
