import cv2
import os
import json
import numpy as np
from camera_utils import take_photo
import platform

class FaceCollector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.output_dir = 'face_data'
        os.makedirs(self.output_dir, exist_ok=True)
        self.is_colab = 'google.colab' in str(get_ipython())
        
    def collect_faces(self, employee_id, name, num_samples=50):
        employee_dir = os.path.join(self.output_dir, str(employee_id))
        os.makedirs(employee_dir, exist_ok=True)
        
        count = 0
        if self.is_colab:
            while count < num_samples:
                try:
                    # Take photo using Colab camera
                    temp_file = take_photo(filename='temp.jpg')
                    frame = cv2.imread(temp_file)
                    
                    if frame is None:
                        print("Failed to capture image")
                        continue
                        
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.detector.detectMultiScale(gray, 1.1, 4)
                    
                    for (x, y, w, h) in faces:
                        face_img = frame[y:y+h, x:x+w]
                        face_img = cv2.resize(face_img, (160, 160))
                        
                        cv2.imwrite(os.path.join(employee_dir, f'{count}.jpg'), face_img)
                        count += 1
                        print(f"Captured face {count}/{num_samples}")
                        
                    os.remove(temp_file)  # Clean up temporary file
                    
                except Exception as e:
                    print(f"Error capturing image: {str(e)}")
                    continue
        else:
            # Original webcam code
            cap = cv2.VideoCapture(0)
            while count < num_samples:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    face_img = frame[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (160, 160))
                    
                    cv2.imwrite(os.path.join(employee_dir, f'{count}.jpg'), face_img)
                    count += 1
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f'Captured: {count}', (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                cv2.imshow('Face Collection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            cap.release()
            cv2.destroyAllWindows()
        
        # Save metadata
        metadata = {
            'employee_id': employee_id,
            'name': name,
            'num_samples': count
        }
        with open(os.path.join(employee_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        print(f"Completed collecting {count} faces for {name}")

if __name__ == "__main__":
    collector = FaceCollector()
    # Example usage:
    collector.collect_faces(1, "John Doe", num_samples=5)  # Reduced samples for testing
