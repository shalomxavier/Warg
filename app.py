from flask import Flask, render_template, Response, jsonify, request
import cv2
from playsound import playsound
import threading
from twilio.rest import Client
import os
from datetime import datetime, timedelta
from threading import Lock, Semaphore
import time

app = Flask(__name__)

# Load multiple pre-trained Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")
full_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

member_count = 0
detection_enabled = True
alarm_enabled = False
sms_enabled = False
last_sms_sent = None

member_count_lock = Lock()
sms_semaphore = Semaphore(1)
sms_cooldown = 60  # in seconds

# Twilio credentials
TWILIO_ACCOUNT_SID = 'YOUR_SID'
TWILIO_AUTH_TOKEN = 'YOUR_TOKEN'
TWILIO_MESSAGING_SERVICE_SID = 'YOUR_MSG_SERVICE_SID'
TWILIO_TO_NUMBER = '+91XXXXXXXXXX'  # Replace with your verified or valid recipient number

def play_alarm():
    playsound('static/alarm.mp3')  # Ensure this path and file exist

def send_sms_alert():
    global last_sms_sent

    if last_sms_sent and datetime.now() - last_sms_sent < timedelta(seconds=sms_cooldown):
        return

    if not sms_semaphore.acquire(blocking=False):
        return

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            messaging_service_sid=TWILIO_MESSAGING_SERVICE_SID,
            body=f"Human presence detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            to=TWILIO_TO_NUMBER
        )
        last_sms_sent = datetime.now()
        print(f"SMS sent! SID: {message.sid}")

    except Exception as e:
        print(f"Failed to send SMS: {str(e)}")

    finally:
        sms_semaphore.release()

def detect_human_parts(frame):
    global member_count, alarm_enabled, sms_enabled
    if not detection_enabled:
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    upper_bodies = upper_body_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    full_bodies = full_body_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    detected_regions = list(faces) + list(upper_bodies) + list(full_bodies)

    with member_count_lock:
        member_count = len(detected_regions)

    if member_count > 0:
        if alarm_enabled:
            threading.Thread(target=play_alarm).start()
        if sms_enabled:
            threading.Thread(target=send_sms_alert, daemon=True).start()

    for (x, y, w, h) in detected_regions:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

def gen_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Camera not accessible.")
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Error: Unable to read from camera.")
                break
            frame = detect_human_parts(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/member_count')
def member_count_endpoint():
    return jsonify({"member_count": member_count})

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global detection_enabled
    detection_enabled = request.json.get("enabled", True)
    return jsonify({"detection_enabled": detection_enabled})

@app.route('/toggle_alarm', methods=['POST'])
def toggle_alarm():
    global alarm_enabled
    alarm_enabled = request.json.get("enabled", True)
    return jsonify({"alarm_enabled": alarm_enabled})

@app.route('/toggle_sms', methods=['POST'])
def toggle_sms():
    global sms_enabled
    sms_enabled = request.json.get("enabled", True)
    return jsonify({"sms_enabled": sms_enabled})

if __name__ == "__main__":
    app.run(debug=True)
