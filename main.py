from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import cvzone
import pyttsx3
import threading
import urllib.request
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Initialize pyttsx3 for offline text-to-speech
engine = pyttsx3.init()

# Create a lock for thread safety
tts_lock = threading.Lock()

def play_sound(text):
    """ Function to convert text to speech using pyttsx3. """
    with tts_lock:
        engine.say(text)
        engine.runAndWait()

def play_sound_async(text):
    """ Run play_sound in a separate thread to avoid blocking. """
    thread = threading.Thread(target=play_sound, args=(text,))
    thread.start()

# Load COCO class names
with open("coco.txt", "r") as f:
    class_names = f.read().splitlines()

# Load the YOLOv8 models
model_webcam = YOLO("yolo11s.pt")
model_rgb = YOLO("yolo11s.pt")  # Use the same model for RGB

# Webcam capture initialization (but don't open it here)
cap = None

# Track spoken IDs to avoid repeating
spoken_ids = set()

def initialize_webcam():
    """Initialize the webcam if not already initialized."""
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)

def release_webcam():
    """Release the webcam capture."""
    global cap
    if cap and cap.isOpened():
        cap.release()

def generate_frames_webcam():
    initialize_webcam()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (1020, 500))
        
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model_webcam.track(frame, persist=True)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
            class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
            track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
            confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score
            
            current_frame_counter = {}

            for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
                c = class_names[class_id]
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
                cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                
                if track_id not in spoken_ids:
                    spoken_ids.add(track_id)
                    
                    if c not in current_frame_counter:
                        current_frame_counter[c] = 0
                    current_frame_counter[c] += 1

            for class_name, count in current_frame_counter.items():
                if count > 0:
                    count_text = f"{count} {class_name}" if count > 1 else f"One {class_name}"
                    play_sound_async(count_text)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release the webcam when done
    release_webcam()

def generate_frames_rgb():
    url = 'http://192.168.0.106:8080/shot.jpg'
    
    while True:
        imgResp = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgNp, -1)
        
        if frame is None:
            continue
        
        frame = cv2.resize(frame, (1020, 500))
        
        # Run YOLOv8 tracking on the RGB frame
        results = model_rgb.track(frame, persist=True)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
            class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
            track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
            confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score
            
            current_frame_counter = {}

            for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
                c = class_names[class_id]
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
                cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                
                if track_id not in spoken_ids:
                    spoken_ids.add(track_id)
                    
                    if c not in current_frame_counter:
                        current_frame_counter[c] = 0
                    current_frame_counter[c] += 1

            for class_name, count in current_frame_counter.items():
                if count > 0:
                    count_text = f"{count} {class_name}" if count > 1 else f"One {class_name}"
                    play_sound_async(count_text)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    release_webcam()  # Ensure webcam is released when navigating to the home page
    return render_template('index.html')

@app.route('/video_feed_webcam')
def video_feed_webcam():
    return Response(generate_frames_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_rgb')
def video_feed_rgb():
    return Response(generate_frames_rgb(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')  # Ensure this HTML template exists

@app.route('/rgb')
def rgb():
    return render_template('rgb.html')  # Ensure this HTML template exists

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/service')
def service():
    return render_template('service.html')
@app.route('/blog')
def blog():
    return render_template('blog.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/detail')
def detail():
    return render_template('detail.html')
@app.route('/price')
def price():
    return render_template('price.html')
@app.route('/team')
def team():
    return render_template('team.html')
@app.route('/testimonial')
def testimonial():
    return render_template('testimonial.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
