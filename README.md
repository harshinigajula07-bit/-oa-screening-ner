# OA Screening NER

**AI-Assisted Early Detection System for Osteoarthritis (OA) Risk Markers in the North Eastern Region (NER)**

OA Screening NER is a Streamlit-based research/prototype application for early screening of movement-related risk markers associated with osteoarthritis. The application combines camera-based pose analysis, gait/sensor parameters, user profile information, local speech recognition, and a local AI question-answering component.

> **Important:** This is a research/prototype screening application. It does not provide a medical diagnosis and should not be treated as a replacement for professional medical evaluation.

## Features

### 1. Phone Login and Language Selection
- English and Hindi can be selected on the login page.
- Phone numbers are normalized for Indian `+91` numbers and international formats.
- OTP authentication is designed for phone verification.
- Twilio SMS credentials can be supplied through environment variables.
- The current source also contains a local fallback when SMS credentials are not configured; this fallback is intended for prototype testing and is **not real phone verification**.

### 2. User Profile
The profile page stores:
- Name
- Sex / Gender
- Height
- Weight
- Blood group
- Previous injury details
- Profile picture

Profile information is stored locally in SQLite.

### 3. MediaPipe Pose
The application uses the MediaPipe Tasks Pose Landmarker model stored locally as:

```text
pose_landmarker_lite.task
```

It supports:
- Camera/captured photo analysis
- Uploaded photo analysis
- Uploaded movement video analysis
- 33 MediaPipe body landmarks
- Landmark X, Y, Z coordinates
- Landmark visibility
- Green landmark points
- Blue body-joint connections
- Knee and hip angle calculations
- Pose result/history storage

The 33 landmarks are:

1. Nose  
2. Left Eye Inner  
3. Left Eye  
4. Left Eye Outer  
5. Right Eye Inner  
6. Right Eye  
7. Right Eye Outer  
8. Left Ear  
9. Right Ear  
10. Mouth Left  
11. Mouth Right  
12. Left Shoulder  
13. Right Shoulder  
14. Left Elbow  
15. Right Elbow  
16. Left Wrist  
17. Right Wrist  
18. Left Pinky  
19. Right Pinky  
20. Left Index  
21. Right Index  
22. Left Thumb  
23. Right Thumb  
24. Left Hip  
25. Right Hip  
26. Left Knee  
27. Right Knee  
28. Left Ankle  
29. Right Ankle  
30. Left Heel  
31. Right Heel  
32. Left Foot Index  
33. Right Foot Index

### 4. Gait / Sensor Risk Assessment
The Sensor Data page accepts four prototype gait parameters and classifies each parameter as **LOW / NORMAL**, **MODERATE**, or **RISK** using the thresholds implemented in the application.

| Parameter | LOW / NORMAL | MODERATE | RISK |
|---|---|---|---|
| Symmetry Index (SI) | 0–10% | >10–20% | >20% |
| Cadence | ≥110 steps/min | 70–<110 steps/min | <70 steps/min |
| Peak Knee Flexion Angle | ≥60° | 45–<60° | <45° |
| Gait Speed | >1.2 m/s | 0.8–1.2 m/s | <0.8 m/s |

The cadence handling follows the prototype implementation because the supplied reference contained an ambiguous upper range around `90–110+`; the code therefore uses `70–<110` as Moderate and `≥110` as LOW / NORMAL.

The Sensor Data page also retains:
- IMU input: accelerometer X/Y/Z and gyroscope X/Y/Z
- FSR input: overall FSR value plus left and right FSR values
- Sensor history stored locally in SQLite

### 5. Knee Analysis
The application provides knee-focused analysis using pose-derived measurements, including left and right knee angles and related movement information.

### 6. Ask a Question
The application includes a local question-answering workflow:
- Text questions
- Microphone input using `streamlit-mic-recorder`
- Offline speech-to-text using Vosk when a local Vosk model is installed
- Local LLM responses using Ollama
- Previous injury information is supplied to the local AI assistant when relevant
- Questions and answers are saved locally

The Ollama endpoint used by the application is:

```text
http://127.0.0.1:11434
```

### 7. Results
The Results page combines available information from:
- User profile
- Latest MediaPipe Pose result
- Gait/sensor assessment
- Recent question history

The prototype combines available pose/sensor status values using the application's status logic. Question/answer history is displayed as supporting information and is not used to assign the risk category.

### 8. Local Data Storage
The application uses SQLite:

```text
oa_screening.db
```

Stored information includes:
- User records
- Verification timestamps
- Profile data
- Pose results
- Sensor data
- Question/answer history

The database migration logic adds missing columns to older databases so existing stored data can be preserved when the schema changes.

## Technology Stack

- **Python**
- **Streamlit** — web application interface
- **MediaPipe Tasks / Pose Landmarker** — pose and 33-landmark detection
- **OpenCV** — image/video processing and landmark drawing
- **NumPy** — numerical calculations
- **Pandas** — result/history tables
- **SQLite** — local data storage
- **Vosk** — local/offline speech-to-text
- **Ollama** — local LLM inference
- **Pillow** — image handling
- **Requests** — communication with local Ollama and configured online services
- **streamlit-mic-recorder** — microphone recording
- **Twilio** — optional online phone OTP delivery when configured

## Project Structure

```text
oa-screening-ner/
│
├── app.py
├── pose_landmarker_lite.task
├── stt_model/
│   └── <local Vosk model files>
├── assets/
│   └── profile/
├── oa_screening.db        # created/used by the application
└── README.md
```

## Requirements

Create and activate a Python virtual environment, then install the required packages.

```bash
pip install streamlit numpy pandas requests pillow opencv-python mediapipe vosk streamlit-mic-recorder
```

You also need:
- The local `pose_landmarker_lite.task` file in the project directory.
- A compatible Vosk model extracted into `stt_model/` for offline speech recognition.
- Ollama installed and running locally for the Ask a Question AI feature.
- At least one Ollama model installed locally.
- Twilio credentials only when real online SMS OTP is enabled.

## Run the Application

From the project directory:

### Windows PowerShell

```powershell
cd "C:\Users\USER\OneDrive\Desktop\oa-screening-ner"
venv\Scripts\activate
streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

## Offline and Online Components

The following components are designed to run locally after their required files/models are installed:

- MediaPipe Pose
- 33-landmark extraction
- OpenCV image/video processing
- SQLite storage
- Vosk speech-to-text
- Ollama local AI
- Results generation

Phone OTP through an external SMS/voice provider requires an internet connection.

## Environment Variables for Real SMS OTP

When using Twilio SMS, keep credentials outside the source code. For example:

```powershell
$env:TWILIO_ACCOUNT_SID="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
$env:TWILIO_AUTH_TOKEN="YOUR_REAL_AUTH_TOKEN"
$env:TWILIO_FROM_NUMBER="+1XXXXXXXXXX"
```

Never commit the real Auth Token or other secrets to GitHub.

> Twilio account permissions, phone-number availability, trial restrictions, and verification requirements are controlled by Twilio and may differ by account and region.

## Navigation

After login, the application provides navigation for:

```text
Home
Profile
MediaPipe Pose
Sensor Data
Knee Analysis
Ask a Question
Results
Language
Logout
```

## Risk Interpretation in This Prototype

The labels **LOW / NORMAL**, **MODERATE**, and **RISK** are prototype screening categories based on the threshold logic implemented in the application. They are not clinical diagnoses.

The application can use more than one source of information, including pose and gait/sensor status. A final status is generated from the available status values according to the application's combination logic.

## Data and Privacy

The application is designed to store user and analysis data locally in SQLite and profile assets locally on the device. If Twilio is enabled, phone-number OTP information is transmitted to Twilio as required for the external SMS service.

Do not commit:

```text
venv/
__pycache__/
oa_screening.db
*.pyc
secrets
API tokens
passwords
private credentials
```

A basic `.gitignore` is recommended for future commits.

## Current Prototype Limitations

- Pose-based status logic is a prototype heuristic, not a validated clinical model.
- The four gait thresholds are the thresholds implemented for this prototype and should not be interpreted as universal clinical cut-offs.
- Sensor values currently can be entered through the UI; direct hardware/Bluetooth acquisition depends on the hardware integration developed separately.
- Offline voice recognition depends on having the appropriate Vosk model locally.
- Ollama responses require Ollama and a local model to be installed and running.
- Real SMS/voice OTP requires an enabled external service and internet connectivity.

## Project Purpose

The project is intended as a low-cost, accessible research prototype for early identification of movement-related OA risk markers, with an emphasis on local processing, multilingual UI support, and use in settings where access to specialized screening resources may be limited.

## Disclaimer

This software is a student/research prototype. It is intended to support screening and demonstration workflows only. It does not diagnose osteoarthritis, determine a medical treatment plan, or replace evaluation by a qualified healthcare professional.
