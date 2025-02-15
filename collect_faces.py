import cv2
import os
import json
from mtcnn import MTCNN
import numpy as np

class FaceCollector:
    def __init__(self):
        self.detector = MTCNN()
        self.output_dir = 'face_data'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def collect_faces(self, employee_id, name, num_samples=50):
        employee_dir = os.path.join(self.output_dir, str(employee_id))
        os.makedirs(employee_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        count = 0
        
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
                
            faces = self.detector.detect_faces(frame)
            
            for face in faces:
                if face['confidence'] > 0.95:
                    x, y, w, h = face['box']
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

if __name__ == "__main__":
    collector = FaceCollector()
    # Example usage:
    collector.collect_faces(0, "John Doe")
