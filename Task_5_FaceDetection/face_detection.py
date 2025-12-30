import cv2
import numpy as np
import os

class FaceDetector:
    def __init__(self):
        """
        Initialize Face Detector using Haar Cascade Classifiers
        """
        # Load pre-trained Haar Cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
    
    def detect_faces_in_image(self, image_path, scale_factor=1.1, min_neighbors=5):
        """
        Detect faces in a single image
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None, None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(30, 30)
        )
        
        return faces, image
    
    def detect_eyes_and_smiles(self, image, faces):
        """
        Detect eyes and smile within detected faces
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_data = []
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            smiles = self.smile_cascade.detectMultiScale(roi_gray)
            
            face_data.append({
                'face': (x, y, w, h),
                'eyes': eyes,
                'smiles': smiles
            })
        
        return face_data
    
    def draw_faces(self, image, faces, face_data=None):
        """
        Draw rectangles around detected faces, eyes, and smiles
        """
        result_image = image.copy()
        
        for idx, (x, y, w, h) in enumerate(faces):
            # Draw face rectangle
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(result_image, f'Face {idx+1}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            if face_data:
                data = face_data[idx]
                # Draw eyes
                for (ex, ey, ew, eh) in data['eyes']:
                    cv2.rectangle(result_image, (x+ex, y+ey), 
                                 (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
                
                # Draw smiles
                for (sx, sy, sw, sh) in data['smiles']:
                    cv2.rectangle(result_image, (x+sx, y+sy), 
                                 (x+sx+sw, y+sy+sh), (0, 0, 255), 2)
        
        return result_image
    
    def process_image(self, image_path, output_path=None):
        """
        Complete face detection pipeline for an image
        """
        faces, image = self.detect_faces_in_image(image_path)
        
        if image is None:
            return None
        
        face_data = self.detect_eyes_and_smiles(image, faces)
        result = self.draw_faces(image, faces, face_data)
        
        if output_path:
            cv2.imwrite(output_path, result)
            print(f"Result saved to {output_path}")
        
        return result, faces, face_data
    
    def detect_from_webcam(self):
        """
        Real-time face detection from webcam
        """
        cap = cv2.VideoCapture(0)
        
        print("Starting webcam. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading from webcam")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # Detect eyes within face
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
            
            cv2.imshow('Face Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    detector = FaceDetector()
    print("Face Detection System Ready!")
    print("You can use this system to:")
    print("1. Detect faces in images")
    print("2. Detect eyes and smiles within faces")
    print("3. Process images with detected faces drawn")
    print("4. Real-time detection from webcam")
