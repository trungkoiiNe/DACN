import cv2
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf
from datetime import datetime
import json
import os
from tensorflow.keras.models import load_model

class AttendanceSystem:
    def __init__(self):
        self.detector = MTCNN()
        self.face_recognizer = load_model('face_recognition_model.h5')
        self.employees = self.load_employees()
        
    def load_employees(self):
        with open('employees.json', 'r') as f:
            return json.load(f)
    
    def mark_attendance(self, employee_id):
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")
        
        attendance_file = f'attendance_{date}.json'
        
        if os.path.exists(attendance_file):
            with open(attendance_file, 'r') as f:
                attendance = json.load(f)
        else:
            attendance = {}
        
        if employee_id not in attendance:
            attendance[employee_id] = {
                "status": "present",
                "time_in": time
            }
            
        with open(attendance_file, 'w') as f:
            json.dump(attendance, f)

    def run(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            faces = self.detector.detect_faces(frame)
            
            for face in faces:
                x, y, w, h = face['box']
                confidence = face['confidence']
                
                if confidence > 0.95:
                    face_img = frame[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (160, 160))
                    face_img = np.expand_dims(face_img, axis=0)
                    
                    predictions = self.face_recognizer.predict(face_img)
                    employee_id = np.argmax(predictions)
                    confidence = predictions[0][employee_id]
                    
                    if confidence > 0.8:
                        name = self.employees[str(employee_id)]["name"]
                        self.mark_attendance(str(employee_id))
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            cv2.imshow('Attendance System', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = AttendanceSystem()
    system.run()
