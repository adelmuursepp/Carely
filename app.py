from email import message
from flask import Flask, render_template, request, redirect, url_for, Response, session, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import numpy as np
import mediapipe as mp
import os
from dotenv import load_dotenv
import openai
from openai import OpenAI  
import requests 
import logging
import requests
import sounddevice as sd
import numpy as np
import whisper
import queue
import threading
import time
from scipy.io.wavfile import write
from io import BytesIO
import soundfile as sf  # To process audio formats like WAV
import tempfile
from resemble import Resemble
import websockets
import asyncio
import base64
from groq import Groq

client = Groq()

Resemble.api_key('JtsTeaqzfiBJuvxFYh9uIgtt')

# Load environment variables from .env
load_dotenv()


model = whisper.load_model("base")


HUME_API_KEY = os.getenv("HUME_API_KEY")
HUME_WS_URL = f"wss://api.hume.ai/v0/stream/models?apiKey={HUME_API_KEY}"


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key

# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

socketio = SocketIO(app, engineio_logger=True, logger=True, cors_allowed_origins="*")

HEYGEN_API_KEY = os.getenv('HEYGEN_API_KEY')

# Endpoint for generating videos
HEYGEN_VIDEO_GENERATE_URL = 'https://api.heygen.com/v2/video/generate'
HEYGEN_VIDEO_STATUS_URL = 'https://api.heygen.com/v1/video_status.get'


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


counter_right = 0
counter_left = 0
stage_right = None
stage_left = None
 
# Camera object for video capturing
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
 
# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point (shoulder)
    b = np.array(b)  # Second point (elbow)
    c = np.array(c)  # Third point (wrist)
   
    # Calculate the angle using arctangent
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
   
    # Ensure the angle is within [0, 180] range
    if angle > 180.0:
        angle = 360 - angle
    return angle
 
# Generate frames to send to the web platform
def generate_frames():
    global counter_right, counter_left, stage_right, stage_left
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Recolor the image to RGB as required by MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
 
            # Make pose detection
            results = pose.process(image)
 
            # Recolor back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
            # Extract landmarks and calculate angles
            try:
                landmarks = results.pose_landmarks.landmark
 
                # Get coordinates for right arm
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
 
                # Get coordinates for left arm
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
 
                # Calculate the angles at the elbow joints
                angle_right = calculate_angle(right_shoulder, right_elbow, right_wrist)
                angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
 
                # Visualize the angles on the screen for both arms
                cv2.putText(image, f'Right Elbow Angle: {int(angle_right)}',
                            tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
 
                cv2.putText(image, f'Left Elbow Angle: {int(angle_left)}',
                            tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
 
                # Curl counter logic for the right arm
                if angle_right > 160:
                    stage_right = "down"
                if angle_right < 30 and stage_right == "down":
                    stage_right = "up"
                    counter_right += 1
                    print(f"Right Reps: {counter_right}")
 
                # Curl counter logic for the left arm
                if angle_left > 160:
                    stage_left = "down"
                if angle_left < 30 and stage_left == "down":
                    stage_left = "up"
                    counter_left += 1
                    print(f"Left Reps: {counter_left}")
 
                # Display feedback for right arm based on form
                feedback_right = "Keep Curling"
                if angle_right > 160:
                    feedback_right = "Right Arm Fully Extended"
                elif angle_right < 30:
                    feedback_right = "Right Arm Fully Contracted"
 
                # Display feedback for left arm based on form
                feedback_left = "Keep Curling"
                if angle_left > 160:
                    feedback_left = "Left Arm Fully Extended"
                elif angle_left < 30:
                    feedback_left = "Left Arm Fully Contracted"
 
                # Show feedback on the frame for both arms
                cv2.putText(image, feedback_right,
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
 
                cv2.putText(image, feedback_left,
                            (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
 
                # Display the rep count on the frame for both arms
                cv2.putText(image, f'Right Reps: {counter_right}',
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
 
                cv2.putText(image, f'Left Reps: {counter_left}',
                            (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
 
            except:
                pass
 
            # Render the pose annotations on the frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
 
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
 
            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Configure the database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # SQLite database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)

# Define Profile model
class Profile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    surgery_type = db.Column(db.String(100), nullable=False)

# Create the database tables
with app.app_context():
    db.create_all()
# Landing Page route
@app.route('/')
def index():
    return render_template('index.html')

# Sign-Up route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password == confirm_password:
            # Hash the password
            hashed_password = generate_password_hash(password)

            # Save the user in the database
            new_user = User(email=email, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            session['user'] = new_user.email  # Save user info in the session
            return redirect(url_for('create_profile'))
        else:
            error = "Passwords do not match."
            return render_template('signup.html', error=error)

    return render_template('signup.html')

# Sign-In route
# Sign-In route
@app.route('/signin', methods=['GET', 'POST'])
def signin():
    print("session is", session)
    error = None  # Initialize error message variable
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Find the user in the database
        user = User.query.filter_by(email=email).first()
        
        if user is None:
            # If the user does not exist
            error = "No account found."
        elif not check_password_hash(user.password, password):
            # If the password does not match
            error = "Wrong email or password."
        else:
            # Successful sign-in
            session['user'] = user.email  # Save user info in the session
            return redirect(url_for('home'))
    
    return render_template('signin.html', error=error)


# Create Profile route
@app.route('/create_profile', methods=['GET', 'POST'])
def create_profile():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']
        surgery_type = 'default'
        
        if 'user' in session:
        # Find the user based on the email in the session
            user = User.query.filter_by(email=session['user']).first()
        else:
            redirect(url_for('signin'))
        if user:
            new_profile = Profile(user_id=user.id, name=name, age=age, gender=gender, surgery_type=surgery_type)
            db.session.add(new_profile)
            db.session.commit()

        return redirect(url_for('home'))

    return render_template('create_profile.html')

@app.route('/leaderboard')
def leaderboard():
    return render_template('leaderboard.html')

@app.route('/excercise')
def excercise():
    return render_template('excercise.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Fixed thread ID and Assistant ID
FIXED_THREAD_ID = "thread_LRBdUm1StIQS7nvb4Osww3R9"
ASSISTANT_ID = "asst_T7NJHUhQr1AY4X7d4MwPF1X1"


@app.route('/new_chat', methods=['GET', 'POST'])
def new_chat():
    if request.method == 'GET':
        # Render the chat interface with the fixed thread_id
        return render_template('new_chat.html', thread_id=FIXED_THREAD_ID)

    elif request.method == 'POST':
        user_message = request.form.get('message')
        thread_id = FIXED_THREAD_ID  # Use the fixed thread_id directly

        if not user_message:
            return jsonify({'error': 'Missing message.'}), 400


        # Step 3: Add user message to the thread
        openai.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_message
        )


        # Step 4: Create a run with the Assistant
        run = openai.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID,
            instructions="Answer as Ishika."
        )

# print(thread_messages.÷data)

        if run.status == 'completed': 


            messages_cursor = openai.beta.threads.messages.list(
                thread_id=thread_id,
                order='desc',
                limit=1
            )
            # print("=======message cursor", messages_cursor)
            messages = list(messages_cursor)
            if messages:
                last_message = messages[0]
                if last_message.role == 'assistant':
                    print("Last message\n\n")
                    assistant_reply = last_message.content[0].text.value

        else:
            print(run.status)



    # Get your default Resemble project.
    project_uuid = Resemble.v2.projects.all(1, 10)['items'][0]['uuid']

    # Get your Voice uuid. In this example, we'll obtain the first.
    voice_uuid = Resemble.v2.voices.all(1, 10)['items'][0]['uuid']

    # Let's create a clip!
    response = Resemble.v2.clips.create_sync(project_uuid,
                                            voice_uuid,
                                            assistant_reply,
                                            title=None,
                                            sample_rate=None,
                                            output_format=None,
                                            precision=None,
                                            include_timestamps=None,
                                            is_archived=None,
                                            raw=None)
    if response['success']:
        # # Get the audio URL from the response
        # audio_url = response['item']['audio_src']
        # print(f"Audio URL: {audio_url}")

        # # Fetch the audio file from the URL
        # audio_response = requests.get(audio_url)

        # # Use a temporary file to store the audio and play it
        # audio_file = BytesIO(audio_response.content)
        # audio_file.seek(0)

        # return send_file(audio_file, mimetype='audio/wav', download_name='audio.wav')

    #     print("assistant reply", assistant_reply)
        return jsonify({'reply': assistant_reply}), 200
    else:
        return jsonify({"error": "Failed to get audio"}), 400


# Handle WebSocket messages from frontend
@socketio.on('message', namespace='video_stream')
async def handle_video_frame(data):
    app.logger.info(f"Received frame of size: {len(data)} bytes")
    # emit('response', {'status': 'Frame received'}) 
    try:
        # Forward the base64-encoded frame to the Hume API
        response = await send_to_hume(data)
        print(f"Hume API Response: {response}")

    except Exception as e:
        print(f"Error processing frame: {e}")

async def send_to_hume(base64_image):
    async with websockets.connect(HUME_WS_URL) as websocket:
        # Send the frame as JSON to Hume API
        message = {
            "models": {"language": {}},
            "data": base64_image
        }
        await websocket.send(json.dumps(message))

        # Receive and return the response from Hume API
        response = await websocket.recv()
        return json.loads(response)


# Endpoint for generating videos
HEYGEN_VIDEO_GENERATE_URL = 'https://api.heygen.com/v2/video/generate'
HEYGEN_VIDEO_STATUS_URL = 'https://api.heygen.com/v1/video_status.get'


# Endpoint for generating videos
HEYGEN_VIDEO_GENERATE_URL = 'https://api.heygen.com/v2/video/generate'
HEYGEN_VIDEO_STATUS_URL = 'https://api.heygen.com/v1/video_status.get'

# Route to generate the video
@app.route('/generate_video', methods=['POST'])
def generate_video():
    data = request.json
    input_text = data.get('input_text', "Welcome to the HeyGen API!")

    # Prepare the video generation request data
    video_data = {
        "video_inputs": [
            {
                "character": {
                    "type": "avatar",
                    "avatar_id": "af235c10c19046648979035bf8192235", 
                    "avatar_style": "friendly"
                },
                "voice": {
                    "type": "text",
                    "input_text": input_text,
                    "voice_id": "39cb1efb2549439c8ee90d14f4e08a89"  
                },
                "background": {
                    "type": "color",
                    "value": "#008000"
                }
            }
        ],
        "dimension": {
            "width": 1280,
            "height": 720
        },
        "aspect_ratio": "16:9",
        "test": True  # Using test mode for free watermarked video
    }

    # Make the request to HeyGen API
    headers = {
        'X-Api-Key': HEYGEN_API_KEY,
        'Content-Type': 'application/json'
    }
    response = requests.post(HEYGEN_VIDEO_GENERATE_URL, json=video_data, headers=headers)
    print(response)

    if response.status_code == 200:
        video_id = response.json().get('video_id')
        return jsonify({'video_id': video_id}), 200
    else:
        return jsonify({'error': 'Failed to generate video'}), 500


# @app.route('/audio_to_text', methods=['POST'])
# def audio_to_text():
#     audio_file = request.files['audio']
#     if not audio_file:
#         return jsonify({'error': 'No audio file found'}), 400

#     # Process the audio file here
#     result = model.transcribe(audio_file)
#     new_chat(result)
#     return jsonify({'message': 'Audio file received'}), 200

 # To make a POST request to the /new_chat route

# @app.route('/audio_to_text', methods=['POST'])
# def audio_to_text():
#     audio_file = request.files['audio']
#     if not audio_file:
#         return jsonify({'error': 'No audio file found'}), 400

#     # Process the audio file here (assuming you have a model object to transcribe)
    
#     audio_bytes = BytesIO(audio_file.read())
#     audio, sample_rate = sf.read(audio_bytes)

    # Check if sample rate needs to be adjusted for the model
    # If using Whisper, it resamples automatically
    # But if you're using another model, you may need to resample

    # Assuming model accepts raw audio data for transcription
    # result = model.transcribe(audio)
    # transcribed_text = result['text']  
    # # Make a POST request to /new_chat with the transcribed text
    # response = requests.post('http://localhost:5000/new_chat', data={'msg': transcribed_text})

    # # Check if the /new_chat route responded successfully
    # if response.status_code == 200:
    #     return jsonify({'message': 'Audio file transcribed and processed successfully'}), 200
    # else:
    #     return jsonify({'error': 'Failed to process the transcribed text'}), 500



@app.route('/audio_to_text', methods=['POST'])
def audio_to_text():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file found'}), 400

    audio_file = request.files['audio']

    # Save the audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        temp_audio_file.write(audio_file.read())
        temp_audio_path = temp_audio_file.name

    # # Transcribe the audio using Whisper
    # result = model.transcribe(temp_audio_path)

        result = client.audio.transcriptions.create(
        file=(temp_audio_path, temp_audio_path.read()),
        model="whisper-large-v3",
        language="en",
        response_format="verbose_json",
        )
        print(result.text)
    transcription = result.text
    # transcription = result['text']

    # Clean up the temporary audio file
    os.remove(temp_audio_path)

    # Make the POST request to another endpoint
    response = requests.post('http://localhost:5000/new_chat', data={'message': transcription})

    if response.status_code != 200:
        return jsonify({'error': 'Failed to process chat.'}), response.status_code

    # Assuming the assistant's reply is received in the response
    assistant_reply = response.json().get('reply')

    # Use Resemble to generate the audio response from the assistant's reply
    Resemble.api_key('JtsTeaqzfiBJuvxFYh9uIgtt')
    project_uuid = Resemble.v2.projects.all(1, 10)['items'][0]['uuid']
    voice_uuid = Resemble.v2.voices.all(1, 10)['items'][0]['uuid']

    # Create the voice clip using Resemble
    resemble_response = Resemble.v2.clips.create_sync(
        project_uuid, voice_uuid, assistant_reply
    )

    if resemble_response['success']:
        # Fetch the audio file from the generated URL
        audio_url = resemble_response['item']['audio_src']
        audio_response = requests.get(audio_url)

        # Use BytesIO to handle the file in memory
        audio_file = BytesIO(audio_response.content)
        audio_file.seek(0)

        # Send the audio file as a response
        return send_file(
            audio_file,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='response_audio.wav'
        )
    else:
        return jsonify({'error': 'Failed to generate audio'}), 400


# Route to check the status of the video
@app.route('/video_status', methods=['GET'])
def video_status():
    video_id = request.args.get('video_id')

    if not video_id:
        return jsonify({'error': 'Missing video ID'}), 400

    # Check the video status
    status_url = f'{HEYGEN_VIDEO_STATUS_URL}?video_id={video_id}'
    headers = {
        'X-Api-Key': HEYGEN_API_KEY
    }
    response = requests.get(status_url, headers=headers)

    if response.status_code == 200:
        status = response.json().get('status')
        video_url = response.json().get('video_url')
        return jsonify({'status': status, 'video_url': video_url}), 200
    else:
        return jsonify({'error': 'Failed to check video status'}), 500

async def test_connection():
    try:
        async with websockets.connect('ws://127.0.0.1:5000/video_stream') as websocket:
            print("Connected to WebSocket!")
            await websocket.send("Test message")
            response = await websocket.recv()
            print(f"Received response: {response}")
    except Exception as e:
        print(f"WebSocket connection failed: {e}")


   
# Home route
@app.route('/home')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', debug=True)
    
