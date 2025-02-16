import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json
import os
from datetime import datetime
from collect_faces import FaceCollector
from train_model import FaceRecognitionTrainer
from tensorflow.keras.models import load_model

class StreamlitAttendance:
    def __init__(self):
        # Use OpenCV's face detector instead of MTCNN
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.employees = self.load_employees()
        if os.path.exists('face_recognition_model.h5'):
            self.face_recognizer = load_model('face_recognition_model.h5')
        else:
            self.face_recognizer = None

    def load_employees(self):
        if os.path.exists('employees.json'):
            with open('employees.json', 'r') as f:
                return json.load(f)
        return {}

    def save_employee(self, employee_id, name, department):
        if not os.path.exists('employees.json'):
            employees = {}
        else:
            with open('employees.json', 'r') as f:
                employees = json.load(f)
        
        employees[str(employee_id)] = {
            "name": name,
            "department": department,
            "employee_id": f"EMP{employee_id:03d}"
        }
        
        with open('employees.json', 'w') as f:
            json.dump(employees, f, indent=4)

    def collect_faces_page(self):
        st.subheader("Register New Employee")
        
        employee_id = st.number_input("Employee ID", min_value=0, step=1)
        name = st.text_input("Employee Name")
        department = st.text_input("Department")
        
        if st.button("Register & Collect Face Data"):
            self.save_employee(employee_id, name, department)
            collector = FaceCollector()
            st.write("Please look at the camera. Press 'q' to stop collection.")
            collector.collect_faces(employee_id, name)
            st.success("Face data collection completed!")

    def train_model_page(self):
        st.subheader("Train Face Recognition Model")
        
        trainer = FaceRecognitionTrainer()
        can_train, message = trainer.validate_training_data()
        
        st.info(message)
        
        if not can_train:
            st.warning("Please register at least 2 employees before training")
            if st.button("Go to Registration"):
                st.session_state.menu_choice = "Register Employee"
                st.experimental_rerun()
            return
        
        if st.button("Start Training"):
            try:
                with st.spinner("Training in progress..."):
                    history = trainer.train()
                st.success("Training completed!")
                if history is not None and hasattr(history, 'history'):
                    st.line_chart(history.history['accuracy'])
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

    def attendance_page(self):
        st.subheader("Live Attendance System")
        
        if self.face_recognizer is None:
            st.error("Please train the model first!")
            return
        
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        stop_button = st.button("Stop")
        
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (160, 160))
                face_img = np.expand_dims(face_img, axis=0)
                
                predictions = self.face_recognizer.predict(face_img)
                employee_id = np.argmax(predictions)
                confidence = predictions[0][employee_id]
                
                if confidence > 0.8:
                    name = self.employees[str(employee_id)]["name"]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame)
        
        cap.release()

    def view_attendance_page(self):
        st.subheader("View Attendance Records")
        
        date = st.date_input("Select Date")
        attendance_file = f'attendance_{date}.json'
        
        if os.path.exists(attendance_file):
            with open(attendance_file, 'r') as f:
                attendance = json.load(f)
            
            for emp_id, data in attendance.items():
                emp_info = self.employees.get(emp_id, {})
                st.write(f"Name: {emp_info.get('name', 'Unknown')}")
                st.write(f"Department: {emp_info.get('department', 'Unknown')}")
                st.write(f"Time In: {data['time_in']}")
                st.write("---")
        else:
            st.write("No attendance records for this date.")

def main():
    st.title("Face Recognition Attendance System")
    
    if 'menu_choice' not in st.session_state:
        st.session_state.menu_choice = "Register Employee"
    
    system = StreamlitAttendance()
    
    menu = ["Register Employee", "Train Model", "Take Attendance", "View Attendance"]
    choice = st.sidebar.selectbox("Select Action", menu, 
                                index=menu.index(st.session_state.menu_choice))
    st.session_state.menu_choice = choice
    
    if choice == "Register Employee":
        system.collect_faces_page()
    elif choice == "Train Model":
        system.train_model_page()
    elif choice == "Take Attendance":
        system.attendance_page()
    else:
        system.view_attendance_page()

if __name__ == "__main__":
    main()
