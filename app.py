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
from twilio.rest import Client
from cartesia import Cartesia
import json

   


from groq import Groq



Resemble.api_key('JtsTeaqzfiBJuvxFYh9uIgtt')

# Load environment variables from .env
load_dotenv()

groq_client = Groq()


twilio_account_sid = os.environ["TWILIO_ACCOUNT_SID"]
twilio_auth_token = os.environ["TWILIO_AUTH_TOKEN"]
twilio_client = Client(twilio_account_sid, twilio_auth_token)

cartesia_client = Cartesia(api_key=os.environ.get("CARTESIA_API_KEY"))


# model = whisper.load_model("base")


HUME_API_KEY = os.getenv("HUME_API_KEY")
HUME_WS_URL = f"wss://api.hume.ai/v0/stream/models?apiKey={HUME_API_KEY}"


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key

# Initialize OpenAI Client
client = OpenAI()

socketio = SocketIO(app, engineio_logger=True, logger=True, cors_allowed_origins="*")

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

@app.route('/send-whatsapp', methods=['POST'])
def send_whatsapp():
    try:
        data = request.json
        message_body = data.get('message', "Signs of sadness. It would be a good idea to check on her.")  



        # Send WhatsApp message
        message = twilio_client.messages.create(
            body=message_body,
            from_="whatsapp:+14155238886",
            to="whatsapp:+14377883703"
        )

        return jsonify({
            "message_sid": message.sid,
            "status": message.status,
            "body": message.body
        }), 200

    except Exception as e:
        print("Error sending whatsapp", e)
        return jsonify({"error": str(e)}), 500

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

def generate_audio_with_cartesia(transcript, voice_id, experimental_controls):
    """
    Generates audio from text using the Cartesia API.

    Parameters:
        transcript (str): The text to convert to speech.

    Returns:
        bytes: The audio data in bytes if successful.
        None: If there was an error during the API call.
    """
    try:
        url = "https://api.cartesia.ai/tts/bytes"
        payload = {
            "model_id": "sonic-english",
            "transcript": transcript,
            "voice": {
                "mode": "id",
                "id": voice_id
            },
            "__experimental_controls": experimental_controls,
            "output_format": {
                "container": "wav",
                "encoding": "pcm_s16le",
                "sample_rate": 44100
            }
        }
        headers = {
            "Cartesia-Version": "2024-06-10",
            "X-API-Key": os.environ.get("CARTESIA_API_KEY"),
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)
        print("Sent request to Cartesia, response status:", response.status_code)

        if response.status_code != 200:
            print(f"Cartesia API Error: {response.status_code} {response.text}")
            return None

        # The response.content contains the audio data
        audio_data = response.content
        return audio_data

    except Exception as e:
        print("Error generating audio with Cartesia:", e)
        return None



# Fixed thread ID and Assistant ID
FIXED_THREAD_ID = "thread_UCyjG6b6PvuvMnB9IVXMAJnr"
ASSISTANT_ID = "asst_4atbjWem0a1P5EUSEVeUBWam"

FIXED_THREAD_ID2 = "thread_s77TwY60UVroCUpKdXkKWazQ"
ASSISTANT_ID2= "asst_o6T1dAedek46toiAMEODve6Q"

ALIS_VOICE_ID="29b344f6-1387-4056-a46d-0ec33d53cb81"  
ADEL_VOICE_ID="81a47897-6b62-42f8-85b4-0689d37e6511"

@app.route('/new_chat', methods=['GET', 'POST'])
def new_chat():
    print("Request received in new_chat")
    if request.method == 'GET':
        # Render the chat interface with the fixed thread_id
        return render_template('new_chat.html', thread_id=FIXED_THREAD_ID)

    elif request.method == 'POST':
        print("Request received in new_chat POST")
        user_message = request.form.get('message')
        print("user message in new chat is ", user_message)
        thread_id = FIXED_THREAD_ID  # Use the fixed thread_id directly

        if not user_message:
            return jsonify({'error': 'Missing message.'}), 400
        
        try:
            thread = client.beta.threads.retrieve(FIXED_THREAD_ID)
            print("Thread exists:", thread)
        except Exception as e:
            print("Error accessing thread:", e)


        # Step 3: Add user message to the thread
        # openai.beta.threads.messages.create(
        #     thread_id=thread_id,
        #     role="user",
        #     content=user_message
        # )


        # Step 4: Create a run with the Assistant
        run = openai.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID,
            instructions="Use maximum 3 sentences to tell a new concise story about a shared memory."
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
                    print(assistant_reply)

        else:
            print(run.status)

        # **Use the separate function to generate audio**
        experimental_controls = {
            "speed": "normal",
            "emotion": [
            "positivity:high",
            "curiosity"
            ]
        }
        audio_data = generate_audio_with_cartesia(assistant_reply, ALIS_VOICE_ID, experimental_controls)

        if audio_data is None:
            return jsonify({'error': 'Failed to generate audio'}), 500

        # Use BytesIO to handle the file in memory
        audio_io = BytesIO(audio_data)
        audio_io.seek(0)

        # Send the audio file as a response
        return send_file(
            audio_io,
            mimetype='audio/wav',
            as_attachment=False,
            download_name='response_audio.wav'
        )



@app.route('/new_chat2', methods=['GET', 'POST'])
def new_chat2():
    if request.method == 'GET':
        # Render the chat interface with the fixed thread_id
        return render_template('new_chat.html', thread_id=FIXED_THREAD_ID2)

    elif request.method == 'POST':
        print("I am in the newchat2 request")
        user_message = request.form.get('message')
        thread_id = FIXED_THREAD_ID2  # Use the fixed thread_id directly


        # Create a run with the Assistant
        run = openai.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID2,
            instructions="Use maximum 3 sentences to tell me a new concise story about a shared memory."
        )


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

        # **Use the separate function to generate audio**
        experimental_controls = {
            "speed": "normal",
            "emotion": [
            "positivity:high",
            "curiosity"
            ]
        }
        audio_data = generate_audio_with_cartesia(assistant_reply, ALIS_VOICE_ID, experimental_controls)

        if audio_data is None:
            return jsonify({'error': 'Failed to generate audio'}), 500

        # Use BytesIO to handle the file in memory
        audio_io = BytesIO(audio_data)
        audio_io.seek(0)

        # Send the audio file as a response
        return send_file(
            audio_io,
            mimetype='audio/wav',
            as_attachment=False,
            download_name='response_audio.wav'
        )



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



@app.route('/audio_to_text', methods=['POST'])
def audio_to_text():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file found'}), 400

    audio_file = request.files['audio']
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as temp_audio_file:
            print("Writing to temp file")  # Debug log
            temp_audio_file.write(audio_file.read())
            temp_audio_path = temp_audio_file.name
            print(f"Temp file path: {temp_audio_path}")  # Debug log

        with open(temp_audio_path, "rb") as file:
            print("Starting transcription")  # Debug log
            transcription = groq_client.audio.transcriptions.create(
                file=(temp_audio_path, file.read()),
                model="whisper-large-v3",
                language="en",
                response_format="verbose_json",
            )
            print("Transcription completed")  
    except:
        print("Error completing transcription")
    # Clean up the temporary audio file
    os.remove(temp_audio_path)

    try:
        thread = client.beta.threads.retrieve(FIXED_THREAD_ID)
        print("Thread exists")
    except Exception as e:
        print("Error accessing thread:", e)

    openai.beta.threads.messages.create(
        thread_id=FIXED_THREAD_ID ,
        role="user",
        content=str(transcription.text)
    )

    run = openai.beta.threads.runs.create_and_poll(
        thread_id=FIXED_THREAD_ID ,
        assistant_id=ASSISTANT_ID,
        instructions="Answer as Mom."
    )


    if run.status == 'completed': 


        messages_cursor = openai.beta.threads.messages.list(
            thread_id=FIXED_THREAD_ID ,
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
                print(assistant_reply)

    else:
        print(run.status)

    # **Use the separate function to generate audio**
    experimental_controls = {
            "speed": "fast",
            "emotion": [
            "positivity:highest",
            "curiosity:high",
            "surprise:high"
            ]
        }
    audio_data = generate_audio_with_cartesia(assistant_reply, ADEL_VOICE_ID, experimental_controls)

    if audio_data is None:
        return jsonify({'error': 'Failed to generate audio'}), 500

    # Use BytesIO to handle the file in memory
    audio_io = BytesIO(audio_data)
    audio_io.seek(0)

    # Send the audio file as a response
    return send_file(
        audio_io,
        mimetype='audio/wav',
        as_attachment=False,
        download_name='response_audio.wav'
    )


    

@app.route('/audio_to_text2', methods=['POST'])
def audio_to_text2():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file found'}), 400

    audio_file = request.files['audio']
    try:
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as temp_audio_file:
            print("Writing to temp file")  # Debug log
            temp_audio_file.write(audio_file.read())
            temp_audio_path = temp_audio_file.name
            print(f"Temp file path: {temp_audio_path}")  # Debug log

        with open(temp_audio_path, "rb") as file:
            print("Starting transcription in audio to text 2")  # Debug log
            transcription = groq_client.audio.transcriptions.create(
                file=(temp_audio_path, file.read()),
                model="whisper-large-v3",
                language="en",
                response_format="verbose_json",
            )
            print("Transcription completed")  
    except:
        print("Error completing transcription")
    # Clean up the temporary audio file
    os.remove(temp_audio_path)

    try:
        thread = client.beta.threads.retrieve(FIXED_THREAD_ID2)
        print("Thread exists")
    except Exception as e:
        print("Error accessing thread:", e)

    openai.beta.threads.messages.create(
        thread_id=FIXED_THREAD_ID2,
        role="user",
        content=str(transcription.text)
    )

    run = openai.beta.threads.runs.create_and_poll(
        thread_id=FIXED_THREAD_ID2,
        assistant_id=ASSISTANT_ID2,
        instructions="Answer as Adel."
    )


    if run.status == 'completed': 


        messages_cursor = openai.beta.threads.messages.list(
            thread_id=FIXED_THREAD_ID2,
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
                print(assistant_reply)

    else:
        print(run.status)

    # **Use the separate function to generate audio**
    experimental_controls = {
            "speed": "fast",
            "emotion": [
            "positivity:highest",
            "curiosity:high",
            "surprise:high"
            ]
        }
    audio_data = generate_audio_with_cartesia(assistant_reply, ADEL_VOICE_ID, experimental_controls)

    if audio_data is None:
        return jsonify({'error': 'Failed to generate audio'}), 500

    # Use BytesIO to handle the file in memory
    audio_io = BytesIO(audio_data)
    audio_io.seek(0)

    # Send the audio file as a response
    return send_file(
        audio_io,
        mimetype='audio/wav',
        as_attachment=False,
        download_name='response_audio.wav'
    )




   
# Home route
@app.route('/home')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', debug=True)
    
