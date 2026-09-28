import os
import io
import json
import wave
import math
import sqlite3
import tempfile
import secrets
from datetime import datetime, timedelta

import streamlit as st
import numpy as np
import pandas as pd
import requests
from PIL import Image

# ---------------------------------------------------------
# OPTIONAL / EXTERNAL PACKAGES
# ---------------------------------------------------------

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

try:
    from streamlit_mic_recorder import mic_recorder
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "pose_landmarker_lite.task"
)

STT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "stt_model"
)

DB_PATH = os.path.join(
    BASE_DIR,
    "oa_screening.db"
)

ASSETS_DIR = os.path.join(
    BASE_DIR,
    "assets"
)

PROFILE_DIR = os.path.join(
    ASSETS_DIR,
    "profile"
)

os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(PROFILE_DIR, exist_ok=True)


# =========================================================
# OLLAMA
# =========================================================

OLLAMA_URL = "http://127.0.0.1:11434"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="OA Screening NER",
    page_icon="🦵",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LANGUAGE TEXT
# =========================================================

TEXT = {

    "English": {

        "home": "Home",
        "profile": "Profile",
        "pose": "MediaPipe Pose",
        "sensor": "Sensor Data",
        "knee": "Knee Analysis",
        "question": "Ask a Question",
        "language": "Language",
        "logout": "Logout",

        "welcome": "Welcome to the OA Screening System",
        "subtitle": "AI-Assisted Early Screening for Osteoarthritis Risk Markers",

        "login": "Login",
        "phone": "Phone Number",
        "continue": "Continue",

        "profile_title": "User Profile",
        "name": "Name",
        "sex": "Sex / Gender",
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "blood": "Blood Group",
        "injury": "Previous Injury Details",
        "save": "Save Profile",

        "capture": "Capture Photo",
        "upload_photo": "Upload Photo",
        "upload_video": "Upload Video",

        "process": "Process",
        "results": "Results",
        "landmarks": "Detected Landmarks",

        "imu": "IMU Data",
        "fsr": "FSR Data",

        "question_placeholder": "Type your question here...",
        "ask": "Ask",

        "start_recording": "🎙️ Start Recording",
        "stop_recording": "⏹️ Stop Recording",

        "speech_text": "Speech converted to text",
        "listening": "Converting speech to text...",

        "no_diagnosis":
            "This is a research/prototype screening system. "
            "It does not provide a medical diagnosis.",

        "risk": "Possible movement marker",
        "normal": "No obvious movement marker",
        "insufficient": "Insufficient pose quality",

        "pose_history": "Pose History",
        "question_history": "Question History",

        "ollama": "Local AI",
        "model": "Model",

        "language_saved": "Language selected",

        "results_page": "Results",
        "sensor_parameters": "Gait / Sensor Parameters",
        "symmetry_index": "Symmetry Index (SI)",
        "cadence": "Cadence",
        "peak_knee_flexion": "Peak Knee Flexion Angle",
        "gait_speed": "Gait Speed",
        "normal_status": "LOW / NORMAL",
        "moderate_status": "MODERATE",
        "risk_status": "RISK",
        "save_gait": "Analyze & Save Gait Parameters",
        "parameter_results": "Parameter Results",
        "overall_result": "Overall Screening Result",
        "source_thresholds": "Reference Thresholds",
        "profile_summary": "Profile Summary",
        "pose_summary": "MediaPipe Pose Summary",
        "latest_questions": "Latest Questions",
        "no_sensor_assessment": "No gait/sensor assessment has been saved yet.",
        "enter_parameters": "Enter the four parameters measured from the prototype sensors / gait analysis.",
        "result_note": "This is a prototype risk-marker screening result, not a medical diagnosis.",
        "cadence_note": "The supplied reference shows 110+ as normal and below 70 as risk; values from 70 to below 110 are treated as moderate in this prototype.",

    },

    "Hindi": {

        "home": "होम",
        "profile": "प्रोफ़ाइल",
        "pose": "MediaPipe Pose",
        "sensor": "सेंसर डेटा",
        "knee": "घुटने का विश्लेषण",
        "question": "प्रश्न पूछें",
        "language": "भाषा",
        "logout": "लॉगआउट",

        "welcome": "OA स्क्रीनिंग सिस्टम में आपका स्वागत है",
        "subtitle": "ऑस्टियोआर्थराइटिस जोखिम संकेतकों की AI-सहायता प्राप्त प्रारंभिक स्क्रीनिंग",

        "login": "लॉगिन",
        "phone": "फोन नंबर",
        "continue": "जारी रखें",

        "profile_title": "उपयोगकर्ता प्रोफ़ाइल",
        "name": "नाम",
        "sex": "लिंग",
        "height": "ऊंचाई (cm)",
        "weight": "वजन (kg)",
        "blood": "ब्लड ग्रुप",
        "injury": "पिछली चोट का विवरण",
        "save": "प्रोफ़ाइल सेव करें",

        "capture": "फोटो लें",
        "upload_photo": "फोटो अपलोड करें",
        "upload_video": "वीडियो अपलोड करें",

        "process": "प्रोसेस करें",
        "results": "परिणाम",
        "landmarks": "पता लगाए गए लैंडमार्क",

        "imu": "IMU डेटा",
        "fsr": "FSR डेटा",

        "question_placeholder": "अपना प्रश्न यहां लिखें...",
        "ask": "पूछें",

        "start_recording": "🎙️ रिकॉर्डिंग शुरू करें",
        "stop_recording": "⏹️ रिकॉर्डिंग रोकें",

        "speech_text": "भाषण से टेक्स्ट",
        "listening": "भाषण को टेक्स्ट में बदला जा रहा है...",

        "no_diagnosis":
            "यह एक शोध/प्रोटोटाइप स्क्रीनिंग सिस्टम है। "
            "यह चिकित्सा निदान प्रदान नहीं करता है।",

        "risk": "संभावित मूवमेंट मार्कर",
        "normal": "कोई स्पष्ट मूवमेंट मार्कर नहीं",
        "insufficient": "पोज़ गुणवत्ता अपर्याप्त",

        "pose_history": "पोज़ इतिहास",
        "question_history": "प्रश्न इतिहास",

        "ollama": "लोकल AI",
        "model": "मॉडल",

        "language_saved": "भाषा चुनी गई",

        "results_page": "परिणाम",
        "sensor_parameters": "चाल / सेंसर पैरामीटर",
        "symmetry_index": "समरूपता सूचकांक (SI)",
        "cadence": "कैडेंस",
        "peak_knee_flexion": "अधिकतम घुटना फ्लेक्शन कोण",
        "gait_speed": "चाल की गति",
        "normal_status": "कम / सामान्य",
        "moderate_status": "मध्यम",
        "risk_status": "जोखिम",
        "save_gait": "चाल पैरामीटर का विश्लेषण और सेव करें",
        "parameter_results": "पैरामीटर परिणाम",
        "overall_result": "समग्र स्क्रीनिंग परिणाम",
        "source_thresholds": "संदर्भ सीमाएं",
        "profile_summary": "प्रोफ़ाइल सारांश",
        "pose_summary": "MediaPipe पोज़ सारांश",
        "latest_questions": "हाल के प्रश्न",
        "no_sensor_assessment": "अभी तक कोई चाल/सेंसर मूल्यांकन सेव नहीं किया गया है।",
        "enter_parameters": "प्रोटोटाइप सेंसर / चाल विश्लेषण से मापे गए चार पैरामीटर दर्ज करें।",
        "result_note": "यह एक प्रोटोटाइप जोखिम-संकेत स्क्रीनिंग परिणाम है, चिकित्सा निदान नहीं।",
        "cadence_note": "दिए गए संदर्भ में 110+ को सामान्य और 70 से कम को जोखिम बताया गया है; इस प्रोटोटाइप में 70 से 110 से कम मानों को मध्यम माना गया है।",

    }
}


# =========================================================
# SESSION STATE
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "phone" not in st.session_state:
    st.session_state.phone = ""

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "language" not in st.session_state:
    st.session_state.language = "English"

if "voice_question_text" not in st.session_state:
    st.session_state.voice_question_text = ""

if "last_pose" not in st.session_state:
    st.session_state.last_pose = None

if "last_sensor" not in st.session_state:
    st.session_state.last_sensor = None

# OTP login state
if "otp_pending" not in st.session_state:
    st.session_state.otp_pending = False

if "otp_phone" not in st.session_state:
    st.session_state.otp_phone = ""

if "otp_code" not in st.session_state:
    st.session_state.otp_code = ""

if "otp_created_at" not in st.session_state:
    st.session_state.otp_created_at = None

if "otp_attempts" not in st.session_state:
    st.session_state.otp_attempts = 0

if "otp_demo_mode" not in st.session_state:
    st.session_state.otp_demo_mode = False


def T(key):
    return TEXT[st.session_state.language].get(
        key,
        key
    )


# =========================================================
# DATABASE
# =========================================================

def get_db():

    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False
    )

    conn.row_factory = sqlite3.Row

    return conn


def init_database():

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            phone TEXT PRIMARY KEY,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            phone TEXT PRIMARY KEY,
            name TEXT,
            sex TEXT,
            height REAL,
            weight REAL,
            blood_group TEXT,
            previous_injury TEXT,
            profile_image_path TEXT,
            updated_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS pose_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT,
            timestamp TEXT,
            input_type TEXT,
            left_knee_angle REAL,
            right_knee_angle REAL,
            left_hip_angle REAL,
            right_hip_angle REAL,
            landmarks_count INTEGER,
            marker_status TEXT,
            details TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT,
            timestamp TEXT,
            sensor_type TEXT,
            data TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT,
            timestamp TEXT,
            input_type TEXT,
            question TEXT,
            answer TEXT
        )
    """)

    # -----------------------------------------------------
    # DATABASE MIGRATION
    # Existing databases may have been created by an older
    # version of the app. CREATE TABLE IF NOT EXISTS does not
    # add newly introduced columns, so add them explicitly.
    # This preserves all existing user/history data.
    # -----------------------------------------------------

    def ensure_columns(table_name, columns):
        cur.execute(f"PRAGMA table_info({table_name})")
        existing = {row[1] for row in cur.fetchall()}

        for column_name, column_type, default_sql in columns:
            if column_name not in existing:
                sql = (
                    f"ALTER TABLE {table_name} "
                    f"ADD COLUMN {column_name} {column_type}"
                )
                if default_sql is not None:
                    sql += f" DEFAULT {default_sql}"
                cur.execute(sql)

    ensure_columns(
        "users",
        [
            ("verified_at", "TEXT", "NULL"),
        ]
    )

    ensure_columns(
        "pose_results",
        [
            ("phone", "TEXT", "NULL"),
            ("timestamp", "TEXT", "NULL"),
            ("input_type", "TEXT", "NULL"),
            ("left_knee_angle", "REAL", "NULL"),
            ("right_knee_angle", "REAL", "NULL"),
            ("left_hip_angle", "REAL", "NULL"),
            ("right_hip_angle", "REAL", "NULL"),
            ("landmarks_count", "INTEGER", "0"),
            ("marker_status", "TEXT", "NULL"),
            ("details", "TEXT", "NULL"),
        ]
    )

    ensure_columns(
        "profiles",
        [
            ("name", "TEXT", "NULL"),
            ("sex", "TEXT", "NULL"),
            ("height", "REAL", "NULL"),
            ("weight", "REAL", "NULL"),
            ("blood_group", "TEXT", "NULL"),
            ("previous_injury", "TEXT", "NULL"),
            ("profile_image_path", "TEXT", "NULL"),
            ("updated_at", "TEXT", "NULL"),
        ]
    )

    ensure_columns(
        "sensor_data",
        [
            ("phone", "TEXT", "NULL"),
            ("timestamp", "TEXT", "NULL"),
            ("sensor_type", "TEXT", "NULL"),
            ("data", "TEXT", "NULL"),
        ]
    )

    ensure_columns(
        "questions",
        [
            ("phone", "TEXT", "NULL"),
            ("timestamp", "TEXT", "NULL"),
            ("input_type", "TEXT", "NULL"),
            ("question", "TEXT", "NULL"),
            ("answer", "TEXT", "NULL"),
        ]
    )

    conn.commit()
    conn.close()


init_database()


# =========================================================
# USER
# =========================================================

def create_or_get_user(phone):

    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        "SELECT phone FROM users WHERE phone=?",
        (phone,)
    )

    row = cur.fetchone()

    if row is None:

        cur.execute(
            """
            INSERT INTO users(phone, created_at)
            VALUES (?, ?)
            """,
            (
                phone,
                datetime.now().isoformat()
            )
        )

        conn.commit()

    conn.close()


# =========================================================
# PROFILE
# =========================================================

def load_profile(phone):

    conn = get_db()

    row = conn.execute(
        """
        SELECT *
        FROM profiles
        WHERE phone=?
        """,
        (phone,)
    ).fetchone()

    conn.close()

    if row:
        return dict(row)

    return None


def save_profile(
    phone,
    name,
    sex,
    height,
    weight,
    blood_group,
    previous_injury,
    image_path
):

    conn = get_db()

    conn.execute(
        """
        INSERT INTO profiles
        (
            phone,
            name,
            sex,
            height,
            weight,
            blood_group,
            previous_injury,
            profile_image_path,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

        ON CONFLICT(phone)
        DO UPDATE SET
            name=excluded.name,
            sex=excluded.sex,
            height=excluded.height,
            weight=excluded.weight,
            blood_group=excluded.blood_group,
            previous_injury=excluded.previous_injury,
            profile_image_path=excluded.profile_image_path,
            updated_at=excluded.updated_at
        """,
        (
            phone,
            name,
            sex,
            height,
            weight,
            blood_group,
            previous_injury,
            image_path,
            datetime.now().isoformat()
        )
    )

    conn.commit()
    conn.close()


# =========================================================
# POSE DATABASE
# =========================================================

def save_pose_result(
    phone,
    input_type,
    left_knee,
    right_knee,
    left_hip,
    right_hip,
    landmarks_count,
    marker_status,
    details
):

    conn = get_db()

    conn.execute(
        """
        INSERT INTO pose_results
        (
            phone,
            timestamp,
            input_type,
            left_knee_angle,
            right_knee_angle,
            left_hip_angle,
            right_hip_angle,
            landmarks_count,
            marker_status,
            details
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            phone,
            datetime.now().isoformat(),
            input_type,
            left_knee,
            right_knee,
            left_hip,
            right_hip,
            landmarks_count,
            marker_status,
            details
        )
    )

    conn.commit()
    conn.close()


def get_pose_history(phone):

    conn = get_db()

    df = pd.read_sql_query(
        """
        SELECT
            timestamp,
            input_type,
            left_knee_angle,
            right_knee_angle,
            left_hip_angle,
            right_hip_angle,
            landmarks_count,
            marker_status
        FROM pose_results
        WHERE phone=?
        ORDER BY id DESC
        """,
        conn,
        params=(phone,)
    )

    conn.close()

    return df


# =========================================================
# SENSOR DATABASE
# =========================================================

def save_sensor_data(
    phone,
    sensor_type,
    data
):

    conn = get_db()

    conn.execute(
        """
        INSERT INTO sensor_data
        (
            phone,
            timestamp,
            sensor_type,
            data
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            phone,
            datetime.now().isoformat(),
            sensor_type,
            json.dumps(data)
        )
    )

    conn.commit()
    conn.close()


def get_sensor_history(phone):

    conn = get_db()

    df = pd.read_sql_query(
        """
        SELECT *
        FROM sensor_data
        WHERE phone=?
        ORDER BY id DESC
        """,
        conn,
        params=(phone,)
    )

    conn.close()

    return df


# =========================================================
# QUESTION DATABASE
# =========================================================

def save_question(
    phone,
    input_type,
    question,
    answer
):

    conn = get_db()

    conn.execute(
        """
        INSERT INTO questions
        (
            phone,
            timestamp,
            input_type,
            question,
            answer
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            phone,
            datetime.now().isoformat(),
            input_type,
            question,
            answer
        )
    )

    conn.commit()
    conn.close()


def get_question_history(phone):

    conn = get_db()

    df = pd.read_sql_query(
        """
        SELECT
            timestamp,
            input_type,
            question,
            answer
        FROM questions
        WHERE phone=?
        ORDER BY id DESC
        """,
        conn,
        params=(phone,)
    )

    conn.close()

    return df


# =========================================================
# MEDIAPIPE
# =========================================================

LANDMARK_NAMES = [

    "Nose",

    "Left Eye Inner",
    "Left Eye",
    "Left Eye Outer",

    "Right Eye Inner",
    "Right Eye",
    "Right Eye Outer",

    "Left Ear",
    "Right Ear",

    "Mouth Left",
    "Mouth Right",

    "Left Shoulder",
    "Right Shoulder",

    "Left Elbow",
    "Right Elbow",

    "Left Wrist",
    "Right Wrist",

    "Left Pinky",
    "Right Pinky",

    "Left Index",
    "Right Index",

    "Left Thumb",
    "Right Thumb",

    "Left Hip",
    "Right Hip",

    "Left Knee",
    "Right Knee",

    "Left Ankle",
    "Right Ankle",

    "Left Heel",
    "Right Heel",

    "Left Foot Index",
    "Right Foot Index"
]

LANDMARK_NAMES_HI = [
    "नाक",
    "बाईं आंख अंदर", "बाईं आंख", "बाईं आंख बाहर",
    "दाईं आंख अंदर", "दाईं आंख", "दाईं आंख बाहर",
    "बायां कान", "दायां कान",
    "मुंह बाईं ओर", "मुंह दाईं ओर",
    "बायां कंधा", "दायां कंधा",
    "बाईं कोहनी", "दाईं कोहनी",
    "बाईं कलाई", "दाईं कलाई",
    "बाईं छोटी उंगली", "दाईं छोटी उंगली",
    "बाईं तर्जनी", "दाईं तर्जनी",
    "बायां अंगूठा", "दायां अंगूठा",
    "बायां कूल्हा", "दायां कूल्हा",
    "बायां घुटना", "दायां घुटना",
    "बायां टखना", "दायां टखना",
    "बाईं एड़ी", "दाईं एड़ी",
    "बायां पैर का अगला भाग", "दायां पैर का अगला भाग"
]


# Common UI messages that occur outside the main translation dictionary.
# Keeping these here makes the most-used screens bilingual without changing
# the existing database or processing logic.
UI_TEXT = {
    "English": {
        "select_language": "Select language",
        "enter_phone": "Enter phone number",
        "valid_phone": "Please enter a valid phone number.",
        "please_phone": "Please enter your phone number.",
        "phone_format_note": "Enter a 10-digit Indian mobile number or an international number starting with +.",
        "send_otp": "Send OTP",
        "otp": "OTP",
        "enter_otp": "Enter 6-digit OTP",
        "verify_otp": "Verify OTP",
        "resend_otp": "Resend OTP",
        "change_phone": "Change Phone Number",
        "otp_sent": "OTP sent to your phone number.",
        "otp_sent_to": "OTP sent to:",
        "wrong_otp": "Incorrect OTP.",
        "otp_expired": "OTP expired. Please request a new OTP.",
        "otp_max_attempts": "Maximum OTP attempts reached. Please request a new OTP.",
        "attempts_remaining": "Attempts remaining",
        "otp_demo_notice": "Local demo OTP mode is active. The OTP is shown below; this is not real phone verification and no SMS/call is sent.",
        "demo_otp": "Demo OTP",
        "phone_verified": "Phone number verified successfully.",
        "analyze_camera": "Analyze Captured Photo",
        "take_picture": "Take a picture",
        "analyze_photo": "Analyze Uploaded Photo",
        "process_video": "Process Video",
        "processing_video": "Processing video...",
        "video_done": "Video processing completed.",
        "pose_saved": "Pose result saved.",
        "original_image": "Original Image",
        "processed_pose": "Processed Pose",
        "landmark_pose": "33 Landmark Pose",
        "upload_a_photo": "Upload a photo",
        "movement_video": "Upload movement video",
        "no_pose": "No pose detected.",
        "left_knee": "Left Knee", "right_knee": "Right Knee",
        "left_hip": "Left Hip", "right_hip": "Right Hip",
        "coordinates": "Coordinates",
        "visibility": "Visibility",
        "landmark_number": "#",
        "all_33": "All 33 MediaPipe Body Landmarks",
        "landmark_note": "MediaPipe Pose detects 33 body landmarks. The table below shows the detected coordinates for this image.",
        "no_history": "No pose history available yet.",
        "system_workflow": "System Workflow",
        "main_features": "Main Features",
        "profile_picture": "Profile Picture",
        "upload_profile": "Upload profile picture",
        "save_profile_success": "Profile saved successfully.",
        "local_ai": "Local AI",
        "select_model": "Select model",
        "answer": "Answer",
        "question_saved": "Question and answer saved locally.",
        "no_questions": "No questions asked yet.",
        "question_label": "Question",
        "answer_label": "Answer",
        "navigation": "Navigation",
        "save_imu": "Save IMU Data", "save_fsr": "Save FSR Data", "reference_table": "Reference table from your supplied parameters",
        "imu_description": "Enter prototype IMU measurements.", "fsr_value": "FSR Value", "left_fsr": "Left FSR", "right_fsr": "Right FSR",
        "save_sensor_success": "Sensor data saved locally.", "no_sensor_data": "No sensor data available.",
        "unit_si": "%", "unit_cadence": "steps/min", "unit_angle": "degrees", "unit_speed": "m/s",
        "normal_range": "NORMAL", "moderate_range": "MODERATE", "risk_range": "RISK",
        "results_navigation": "Results", "no_pose_result": "No MediaPipe Pose result is available yet.",
    },
    "Hindi": {
        "select_language": "भाषा चुनें",
        "enter_phone": "फोन नंबर दर्ज करें",
        "valid_phone": "कृपया सही फोन नंबर दर्ज करें।",
        "please_phone": "कृपया अपना फोन नंबर दर्ज करें।",
        "phone_format_note": "10 अंकों का भारतीय मोबाइल नंबर या + से शुरू होने वाला अंतरराष्ट्रीय नंबर दर्ज करें।",
        "send_otp": "OTP भेजें",
        "otp": "OTP",
        "enter_otp": "6 अंकों का OTP दर्ज करें",
        "verify_otp": "OTP सत्यापित करें",
        "resend_otp": "OTP फिर से भेजें",
        "change_phone": "फोन नंबर बदलें",
        "otp_sent": "OTP आपके फोन नंबर पर भेज दिया गया है।",
        "otp_sent_to": "OTP भेजा गया:",
        "wrong_otp": "OTP गलत है।",
        "otp_expired": "OTP की अवधि समाप्त हो गई है। कृपया नया OTP मांगें।",
        "otp_max_attempts": "OTP के अधिकतम प्रयास पूरे हो गए हैं। कृपया नया OTP मांगें।",
        "attempts_remaining": "बचे हुए प्रयास",
        "otp_demo_notice": "लोकल डेमो OTP मोड सक्रिय है। OTP नीचे दिखाया गया है; कोई SMS/कॉल नहीं भेजी जाती और यह वास्तविक फोन सत्यापन नहीं है।",
        "demo_otp": "डेमो OTP",
        "phone_verified": "फोन नंबर सफलतापूर्वक सत्यापित हो गया।",
        "analyze_camera": "कैमरा फोटो का विश्लेषण करें",
        "take_picture": "फोटो लें",
        "analyze_photo": "अपलोड की गई फोटो का विश्लेषण करें",
        "process_video": "वीडियो प्रोसेस करें",
        "processing_video": "वीडियो प्रोसेस किया जा रहा है...",
        "video_done": "वीडियो प्रोसेसिंग पूरी हो गई।",
        "pose_saved": "पोज़ परिणाम सेव किया गया।",
        "original_image": "मूल फोटो",
        "processed_pose": "प्रोसेस किया गया पोज़",
        "landmark_pose": "33 लैंडमार्क पोज़",
        "upload_a_photo": "फोटो अपलोड करें",
        "movement_video": "मूवमेंट वीडियो अपलोड करें",
        "no_pose": "कोई पोज़ नहीं मिला।",
        "left_knee": "बायां घुटना", "right_knee": "दायां घुटना",
        "left_hip": "बायां कूल्हा", "right_hip": "दायां कूल्हा",
        "coordinates": "निर्देशांक",
        "visibility": "दृश्यता",
        "landmark_number": "क्रमांक",
        "all_33": "सभी 33 MediaPipe बॉडी लैंडमार्क",
        "landmark_note": "MediaPipe Pose शरीर के 33 लैंडमार्क पहचानता है। नीचे इस फोटो में पहचाने गए निर्देशांक दिखाए गए हैं।",
        "no_history": "अभी कोई पोज़ इतिहास उपलब्ध नहीं है।",
        "system_workflow": "सिस्टम कार्यप्रवाह",
        "main_features": "मुख्य सुविधाएं",
        "profile_picture": "प्रोफ़ाइल फोटो",
        "upload_profile": "प्रोफ़ाइल फोटो अपलोड करें",
        "save_profile_success": "प्रोफ़ाइल सफलतापूर्वक सेव की गई।",
        "local_ai": "लोकल AI",
        "select_model": "मॉडल चुनें",
        "answer": "उत्तर",
        "question_saved": "प्रश्न और उत्तर स्थानीय रूप से सेव किए गए।",
        "no_questions": "अभी कोई प्रश्न नहीं पूछा गया है।",
        "question_label": "प्रश्न",
        "answer_label": "उत्तर",
        "navigation": "नेविगेशन",
        "save_imu": "IMU डेटा सेव करें", "save_fsr": "FSR डेटा सेव करें", "reference_table": "आपके दिए गए पैरामीटर की संदर्भ तालिका",
        "imu_description": "प्रोटोटाइप IMU माप दर्ज करें।", "fsr_value": "FSR मान", "left_fsr": "बायां FSR", "right_fsr": "दायां FSR",
        "save_sensor_success": "सेंसर डेटा स्थानीय रूप से सेव किया गया।", "no_sensor_data": "कोई सेंसर डेटा उपलब्ध नहीं है।",
        "unit_si": "%", "unit_cadence": "स्टेप/मिनट", "unit_angle": "डिग्री", "unit_speed": "m/s",
        "normal_range": "सामान्य", "moderate_range": "मध्यम", "risk_range": "जोखिम",
        "results_navigation": "परिणाम", "no_pose_result": "अभी कोई MediaPipe Pose परिणाम उपलब्ध नहीं है।",
    }
}

def U(key):
    return UI_TEXT[st.session_state.language].get(key, key)


POSE_CONNECTIONS = [

    (11, 12),

    (11, 13),
    (13, 15),

    (15, 17),
    (15, 19),
    (15, 21),

    (12, 14),
    (14, 16),

    (16, 18),
    (16, 20),
    (16, 22),

    (11, 23),
    (12, 24),

    (23, 24),

    (23, 25),
    (25, 27),

    (27, 29),
    (27, 31),

    (24, 26),
    (26, 28),

    (28, 30),
    (28, 32)
]


@st.cache_resource
def load_pose_landmarker():

    if not MP_AVAILABLE:
        return None

    if not os.path.exists(MODEL_PATH):
        return None

    try:

        base_options = python.BaseOptions(
            model_asset_path=MODEL_PATH
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        return vision.PoseLandmarker.create_from_options(
            options
        )

    except Exception as e:

        st.error(
            f"MediaPipe model error: {e}"
        )

        return None


# =========================================================
# ANGLE
# =========================================================

def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba) *
        np.linalg.norm(bc)
    )

    if denominator == 0:
        return None

    cosine_angle = np.dot(ba, bc) / denominator

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    angle = np.degrees(
        np.arccos(cosine_angle)
    )

    return float(angle)


def calculate_pose_angles(landmarks):

    def p(index):

        lm = landmarks[index]

        return [
            lm.x,
            lm.y,
            lm.z
        ]

    left_knee = calculate_angle(
        p(23),
        p(25),
        p(27)
    )

    right_knee = calculate_angle(
        p(24),
        p(26),
        p(28)
    )

    left_hip = calculate_angle(
        p(11),
        p(23),
        p(25)
    )

    right_hip = calculate_angle(
        p(12),
        p(24),
        p(26)
    )

    return (
        left_knee,
        right_knee,
        left_hip,
        right_hip
    )


# =========================================================
# SCREENING HEURISTIC
# =========================================================

def evaluate_pose(
    landmarks_count,
    left_knee,
    right_knee
):

    if landmarks_count < 20:

        return (
            T("insufficient"),
            "Fewer than 20 landmarks were detected."
        )

    angles = [
        x for x in [
            left_knee,
            right_knee
        ]
        if x is not None
    ]

    if not angles:

        return (
            T("insufficient"),
            "Knee landmarks could not be measured."
        )

    possible_marker = False

    for angle in angles:

        if angle < 70 or angle > 175:

            possible_marker = True

    if possible_marker:

        return (
            T("risk"),
            "One or more knee-angle measurements "
            "were outside the prototype screening range."
        )

    return (
        T("normal"),
        "No obvious movement marker was detected "
        "by this prototype heuristic."
    )


# =========================================================
# DRAW LANDMARKS
# =========================================================

def draw_landmarks(
    image,
    landmarks
):

    if not CV2_AVAILABLE:
        return image

    output = image.copy()

    height, width = output.shape[:2]

    points = []

    for lm in landmarks:

        x = int(lm.x * width)
        y = int(lm.y * height)

        points.append(
            (x, y)
        )

        cv2.circle(
            output,
            (x, y),
            4,
            (0, 255, 0),
            -1
        )

    for a, b in POSE_CONNECTIONS:

        if a < len(points) and b < len(points):

            cv2.line(
                output,
                points[a],
                points[b],
                (255, 0, 0),
                2
            )

    return output


# =========================================================
# PROCESS IMAGE
# =========================================================

def process_image(
    image_bytes
):

    if not CV2_AVAILABLE:

        return None, None, "OpenCV not installed."

    landmarker = load_pose_landmarker()

    if landmarker is None:

        return (
            None,
            None,
            "MediaPipe model not found. "
            "Make sure pose_landmarker_lite.task "
            "is in the project folder."
        )

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        rgb = np.array(image)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(
            mp_image
        )

        if not result.pose_landmarks:

            return (
                image,
                None,
                "No pose detected."
            )

        landmarks = result.pose_landmarks[0]

        processed = draw_landmarks(
            rgb,
            landmarks
        )

        processed_image = Image.fromarray(
            processed
        )

        angles = calculate_pose_angles(
            landmarks
        )

        count = len(landmarks)

        status, details = evaluate_pose(
            count,
            angles[0],
            angles[1]
        )

        result_data = {

            "landmarks": landmarks,

            "landmarks_count": count,

            "left_knee": angles[0],
            "right_knee": angles[1],

            "left_hip": angles[2],
            "right_hip": angles[3],

            "status": status,

            "details": details
        }

        return (
            processed_image,
            result_data,
            None
        )

    except Exception as e:

        return (
            None,
            None,
            f"Pose processing error: {e}"
        )


# =========================================================
# PROCESS VIDEO
# =========================================================

def process_video(
    video_bytes
):

    if not CV2_AVAILABLE:

        return None, None, "OpenCV not installed."

    landmarker = load_pose_landmarker()

    if landmarker is None:

        return (
            None,
            None,
            "MediaPipe model not found."
        )

    input_path = None
    output_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as f:

            f.write(video_bytes)

            input_path = f.name

        cap = cv2.VideoCapture(
            input_path
        )

        if not cap.isOpened():

            return (
                None,
                None,
                "Could not open video."
            )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 20

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as f:

            output_path = f.name

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        frame_results = []
        last_landmarks = None

        frame_index = 0

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect(
                mp_image
            )

            if result.pose_landmarks:

                landmarks = result.pose_landmarks[0]
                last_landmarks = list(landmarks)

                frame = draw_landmarks(
                    frame,
                    landmarks
                )

                angles = calculate_pose_angles(
                    landmarks
                )

                frame_results.append(
                    {
                        "frame": frame_index,
                        "left_knee": angles[0],
                        "right_knee": angles[1],
                        "left_hip": angles[2],
                        "right_hip": angles[3],
                        "landmarks": len(landmarks)
                    }
                )

            writer.write(frame)

            frame_index += 1

        cap.release()
        writer.release()

        if not frame_results:

            return (
                None,
                None,
                "No pose detected in video."
            )

        df = pd.DataFrame(
            frame_results
        )

        valid_left = df[
            "left_knee"
        ].dropna()

        valid_right = df[
            "right_knee"
        ].dropna()

        left_knee = (
            float(valid_left.mean())
            if not valid_left.empty
            else None
        )

        right_knee = (
            float(valid_right.mean())
            if not valid_right.empty
            else None
        )

        left_hip = (
            float(df["left_hip"].dropna().mean())
            if not df["left_hip"].dropna().empty
            else None
        )

        right_hip = (
            float(df["right_hip"].dropna().mean())
            if not df["right_hip"].dropna().empty
            else None
        )

        status, details = evaluate_pose(
            int(df["landmarks"].mean()),
            left_knee,
            right_knee
        )

        result_data = {

            "landmarks": last_landmarks,

            "landmarks_count":
                int(df["landmarks"].mean()),

            "left_knee":
                left_knee,

            "right_knee":
                right_knee,

            "left_hip":
                left_hip,

            "right_hip":
                right_hip,

            "status":
                status,

            "details":
                details,

            "frame_data":
                df
        }

        with open(
            output_path,
            "rb"
        ) as f:

            processed_video = f.read()

        return (
            processed_video,
            result_data,
            None
        )

    except Exception as e:

        return (
            None,
            None,
            f"Video processing error: {e}"
        )

    finally:

        if input_path and os.path.exists(input_path):

            try:
                os.remove(input_path)
            except:
                pass


# =========================================================
# VOSK
# =========================================================

@st.cache_resource
def load_vosk_model():

    if not VOSK_AVAILABLE:
        return None

    if not os.path.exists(
        STT_MODEL_PATH
    ):
        return None

    try:

        return Model(
            STT_MODEL_PATH
        )

    except Exception as e:

        return None


def speech_to_text(
    audio_bytes
):

    if not VOSK_AVAILABLE:

        return (
            "",
            "Vosk is not installed. "
            "Run pip install vosk."
        )

    model = load_vosk_model()

    if model is None:

        return (
            "",
            "Vosk model not found. "
            "Put the extracted speech model "
            "inside the stt_model folder."
        )

    temp_audio = os.path.join(
        BASE_DIR,
        "temp_recording.wav"
    )

    try:

        with open(
            temp_audio,
            "wb"
        ) as f:

            f.write(audio_bytes)

        wf = wave.open(
            temp_audio,
            "rb"
        )

        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()

        if channels != 1:

            wf.close()

            return (
                "",
                "Microphone audio is not mono. "
                "Please use a mono WAV recording."
            )

        if sample_width != 2:

            wf.close()

            return (
                "",
                "Microphone audio is not 16-bit PCM."
            )

        recognizer = KaldiRecognizer(
            model,
            sample_rate
        )

        while True:

            data = wf.readframes(
                4000
            )

            if not data:
                break

            recognizer.AcceptWaveform(
                data
            )

        result = json.loads(
            recognizer.FinalResult()
        )

        wf.close()

        text_result = result.get(
            "text",
            ""
        ).strip()

        if not text_result:

            return (
                "",
                "No speech was detected."
            )

        return (
            text_result,
            None
        )

    except Exception as e:

        return (
            "",
            f"Speech-to-text error: {e}"
        )

    finally:

        if os.path.exists(
            temp_audio
        ):

            try:
                os.remove(
                    temp_audio
                )
            except:
                pass


# =========================================================
# OLLAMA
# =========================================================

def get_ollama_models():

    try:

        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=3
        )

        if response.status_code != 200:
            return []

        data = response.json()

        models = []

        for item in data.get(
            "models",
            []
        ):

            name = item.get(
                "name"
            )

            if name:
                models.append(name)

        return models

    except:

        return []


def ask_ollama(
    question,
    model
):

    profile = load_profile(
        st.session_state.phone
    )

    previous_injury = ""

    if profile:

        previous_injury = (
            profile.get(
                "previous_injury"
            ) or ""
        )

    prompt = f"""
You are the local AI assistant for an
offline osteoarthritis screening research prototype.

Answer the user's question clearly and simply.

Important rules:

1. This application is a screening prototype.
2. Do not claim to diagnose osteoarthritis.
3. Do not invent medical measurements.
4. Do not claim the application replaces a doctor.
5. Explain screening results carefully.
6. If previous injury information is provided,
   consider it when relevant.
7. Do not provide dangerous medical instructions.
8. Encourage appropriate professional medical evaluation
   when the user has concerning symptoms.

Previous injury information:
{previous_injury}

User question:
{question}
"""

    try:

        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )

        if response.status_code != 200:

            return (
                "",
                f"Ollama returned HTTP "
                f"{response.status_code}: "
                f"{response.text}"
            )

        data = response.json()

        answer = data.get(
            "response",
            ""
        ).strip()

        if not answer:

            return (
                "",
                "Ollama returned an empty answer."
            )

        return (
            answer,
            None
        )

    except requests.exceptions.ConnectionError:

        return (
            "",
            "Ollama is not running. "
            "Start Ollama and make sure a local model is installed."
        )

    except Exception as e:

        return (
            "",
            f"Ollama error: {e}"
        )


# =========================================================
# OTP AUTHENTICATION - LOCAL DEMO OTP (NO TWILIO)
# =========================================================

OTP_EXPIRY_MINUTES = 5
OTP_MAX_ATTEMPTS = 5
DEMO_OTP = "123456"


def normalize_phone(phone):
    """Normalize Indian 10-digit or international phone numbers."""
    phone = str(phone).strip()
    compact = phone.replace(" ", "").replace("-", "")

    if compact.startswith("+91") and compact[3:].isdigit() and len(compact[3:]) == 10:
        return compact

    if compact.isdigit() and len(compact) == 10:
        return "+91" + compact

    if compact.startswith("+") and compact[1:].isdigit() and 10 <= len(compact[1:]) <= 15:
        return compact

    return ""


def generate_demo_otp():
    """Return the fixed OTP used only for local prototype demonstration."""
    return DEMO_OTP


def start_otp_login(phone):
    """Start a local demo OTP flow. No SMS, call, or internet is used."""
    otp = generate_demo_otp()

    st.session_state.otp_pending = True
    st.session_state.otp_phone = phone
    st.session_state.otp_code = otp
    st.session_state.otp_created_at = datetime.now().isoformat()
    st.session_state.otp_attempts = 0
    st.session_state.otp_demo_mode = True

    return True, "demo"


def otp_is_expired():
    created = st.session_state.otp_created_at

    if not created:
        return True

    try:
        created_dt = datetime.fromisoformat(created)
    except ValueError:
        return True

    return datetime.now() - created_dt > timedelta(minutes=OTP_EXPIRY_MINUTES)


def clear_otp_state():
    st.session_state.otp_pending = False
    st.session_state.otp_phone = ""
    st.session_state.otp_code = ""
    st.session_state.otp_created_at = None
    st.session_state.otp_attempts = 0
    st.session_state.otp_demo_mode = False


def mark_phone_verified(phone):
    conn = get_db()
    conn.execute(
        """
        UPDATE users
        SET verified_at=?
        WHERE phone=?
        """,
        (datetime.now().isoformat(), phone)
    )
    conn.commit()
    conn.close()


# =========================================================
# LOGIN PAGE
# =========================================================

def login_page():

    st.title("🦵 OA Screening NER")

    # Language is selected BEFORE login so the entire application
    # starts in the user's selected language.
    language = st.selectbox(
        U("select_language"),
        ["English", "Hindi"],
        index=0 if st.session_state.language == "English" else 1,
        key="login_language"
    )

    if language != st.session_state.language:
        st.session_state.language = language
        st.rerun()

    st.subheader(T("login"))
    st.info(T("subtitle"))
    st.info(U("otp_demo_notice"))

    if not st.session_state.otp_pending:

        phone = st.text_input(
            T("phone"),
            placeholder=U("enter_phone"),
            key="login_phone"
        )

        st.caption(U("phone_format_note"))

        if st.button(
            T("send_otp"),
            type="primary",
            use_container_width=True
        ):
            normalized = normalize_phone(phone)

            if not normalized:
                st.error(U("valid_phone"))
                return

            create_or_get_user(normalized)
            ok, mode = start_otp_login(normalized)

            if not ok:
                st.error(mode)
                return

            if mode == "demo":
                st.warning(U("otp_demo_notice"))
                st.success(f"{U('demo_otp')}: {st.session_state.otp_code}")

            st.rerun()

    else:

        masked = st.session_state.otp_phone
        if len(masked) >= 4:
            masked = "*" * max(0, len(masked) - 4) + masked[-4:]

        st.info(f"{U('otp_sent_to')} {masked}")

        # Demo OTP is always displayed clearly in local prototype mode.
        if st.session_state.otp_demo_mode:
            st.warning(U("otp_demo_notice"))
            st.markdown(f"### 🔐 {U('demo_otp')}")
            st.code(st.session_state.otp_code or DEMO_OTP, language=None)
            st.caption("Enter this 6-digit code in the OTP box below.")

        otp = st.text_input(
            T("otp"),
            max_chars=6,
            placeholder=U("enter_otp"),
            key="login_otp"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                T("verify_otp"),
                type="primary",
                use_container_width=True
            ):
                if otp_is_expired():
                    st.error(U("otp_expired"))
                    clear_otp_state()
                    st.rerun()

                if st.session_state.otp_attempts >= OTP_MAX_ATTEMPTS:
                    st.error(U("otp_max_attempts"))
                    clear_otp_state()
                    st.rerun()

                st.session_state.otp_attempts += 1

                if otp.strip() == st.session_state.otp_code:
                    phone = st.session_state.otp_phone
                    mark_phone_verified(phone)

                    st.session_state.phone = phone
                    st.session_state.logged_in = True
                    st.session_state.page = "Home"
                    clear_otp_state()
                    st.success(U("phone_verified"))
                    st.rerun()
                else:
                    remaining = OTP_MAX_ATTEMPTS - st.session_state.otp_attempts
                    st.error(f"{U('wrong_otp')} {U('attempts_remaining')}: {remaining}")

        with col2:
            if st.button(
                T("resend_otp"),
                use_container_width=True
            ):
                ok, mode = start_otp_login(st.session_state.otp_phone)

                if not ok:
                    st.error(mode)
                else:
                    if mode == "demo":
                        st.warning(U("otp_demo_notice"))
                        st.success(f"{U('demo_otp')}: {st.session_state.otp_code}")
                    else:
                        st.success(U("otp_sent"))
                    st.rerun()

        if st.button(
            T("change_phone"),
            use_container_width=True
        ):
            clear_otp_state()
            st.rerun()


# =========================================================
# HOME
# =========================================================

def home_page():

    st.title(
        "🦵 " + T("welcome")
    )

    st.subheader(
        T("subtitle")
    )

    st.markdown(
        """
        ### System Workflow

        **Login → Profile → MediaPipe Pose → Sensor Data
        → Knee Analysis → Ask a Question → Results**

        The system uses local processing for the prototype.
        """
    )

    st.warning(
        T("no_diagnosis")
    )

    st.markdown(
        """
        ### Main Features

        - 📷 MediaPipe Pose analysis
        - 🦴 33 body landmark detection
        - 📐 Knee and hip angle calculation
        - 📈 Pose history
        - 📡 IMU data
        - 🦶 FSR data
        - 🎤 Offline speech-to-text using Vosk
        - 🤖 Local AI using Ollama
        - 💾 Local SQLite storage
        - 🌐 English and Hindi interface
        """
    )

    st.success(
        f"Logged in as: {st.session_state.phone}"
    )


# =========================================================
# PROFILE PAGE
# =========================================================

def profile_page():

    st.title(
        "👤 " + T("profile_title")
    )

    profile = load_profile(
        st.session_state.phone
    )

    if profile is None:

        profile = {}

    col1, col2 = st.columns(
        [1, 2]
    )

    with col1:

        st.subheader(
            "Profile Picture"
        )

        current_image = profile.get(
            "profile_image_path"
        )

        if (
            current_image
            and os.path.exists(current_image)
        ):

            st.image(
                current_image,
                width=180
            )

        uploaded_image = st.file_uploader(
            "Upload profile picture",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            key="profile_image"
        )

    with col2:

        name = st.text_input(
            T("name"),
            value=profile.get(
                "name",
                ""
            )
        )

        sex = st.selectbox(
            T("sex"),
            [
                "Prefer not to say",
                "Female",
                "Male",
                "Other"
            ],
            index=0
        )

        if profile.get("sex") in [
            "Prefer not to say",
            "Female",
            "Male",
            "Other"
        ]:

            sex = st.selectbox(
                T("sex"),
                [
                    "Prefer not to say",
                    "Female",
                    "Male",
                    "Other"
                ],
                index=[
                    "Prefer not to say",
                    "Female",
                    "Male",
                    "Other"
                ].index(
                    profile.get("sex")
                )
            )

        height = st.number_input(
            T("height"),
            min_value=0.0,
            max_value=300.0,
            value=float(
                profile.get(
                    "height"
                ) or 0
            ),
            step=0.1
        )

        weight = st.number_input(
            T("weight"),
            min_value=0.0,
            max_value=500.0,
            value=float(
                profile.get(
                    "weight"
                ) or 0
            ),
            step=0.1
        )

        blood_group = st.selectbox(
            T("blood"),
            [
                "",
                "A+",
                "A-",
                "B+",
                "B-",
                "AB+",
                "AB-",
                "O+",
                "O-"
            ]
        )

        if profile.get(
            "blood_group"
        ):

            blood_group = st.selectbox(
                T("blood"),
                [
                    "",
                    "A+",
                    "A-",
                    "B+",
                    "B-",
                    "AB+",
                    "AB-",
                    "O+",
                    "O-"
                ],
                index=[
                    "",
                    "A+",
                    "A-",
                    "B+",
                    "B-",
                    "AB+",
                    "AB-",
                    "O+",
                    "O-"
                ].index(
                    profile.get(
                        "blood_group"
                    )
                )
            )

        previous_injury = st.text_area(
            T("injury"),
            value=profile.get(
                "previous_injury",
                ""
            ),
            height=120,
            placeholder=(
                "Example: previous knee injury, "
                "year, side, treatment, etc."
            )
        )

    if st.button(
        T("save"),
        type="primary"
    ):

        image_path = profile.get(
            "profile_image_path"
        )

        if uploaded_image:

            extension = os.path.splitext(
                uploaded_image.name
            )[1]

            image_path = os.path.join(
                PROFILE_DIR,
                f"{st.session_state.phone}{extension}"
            )

            with open(
                image_path,
                "wb"
            ) as f:

                f.write(
                    uploaded_image.getbuffer()
                )

        save_profile(
            st.session_state.phone,
            name,
            sex,
            height,
            weight,
            blood_group,
            previous_injury,
            image_path
        )

        st.success(
            "Profile saved successfully."
        )


# =========================================================
# POSE PAGE
# =========================================================

def display_landmark_table(result):
    """Display all 33 MediaPipe Pose landmarks with coordinates and visibility."""
    landmarks = result.get("landmarks") if result else None
    if not landmarks:
        st.info(U("no_pose"))
        return

    names = LANDMARK_NAMES_HI if st.session_state.language == "Hindi" else LANDMARK_NAMES
    rows = []

    for i in range(33):
        if i < len(landmarks):
            lm = landmarks[i]
            visibility = getattr(lm, "visibility", None)
            rows.append({
                U("landmark_number"): i + 1,
                "Landmark" if st.session_state.language == "English" else "लैंडमार्क": names[i],
                "X": round(float(lm.x), 4),
                "Y": round(float(lm.y), 4),
                "Z": round(float(lm.z), 4),
                U("visibility"): round(float(visibility), 4) if visibility is not None else None,
            })
        else:
            rows.append({
                U("landmark_number"): i + 1,
                "Landmark" if st.session_state.language == "English" else "लैंडमार्क": names[i],
                "X": None, "Y": None, "Z": None, U("visibility"): None
            })

    st.subheader(U("all_33"))
    st.caption(U("landmark_note"))
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def display_pose_result(
    result
):

    if not result:
        return

    st.subheader(
        T("results")
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            U("left_knee"),
            (
                f"{result['left_knee']:.1f}°"
                if result["left_knee"] is not None
                else "N/A"
            )
        )

    with c2:

        st.metric(
            U("right_knee"),
            (
                f"{result['right_knee']:.1f}°"
                if result["right_knee"] is not None
                else "N/A"
            )
        )

    with c3:

        st.metric(
            U("left_hip"),
            (
                f"{result['left_hip']:.1f}°"
                if result["left_hip"] is not None
                else "N/A"
            )
        )

    with c4:

        st.metric(
            U("right_hip"),
            (
                f"{result['right_hip']:.1f}°"
                if result["right_hip"] is not None
                else "N/A"
            )
        )

    st.write(
        f"**{T('landmarks')}:** "
        f"{result['landmarks_count']}"
    )

    display_landmark_table(result)

    status = result["status"]

    if status == T("risk"):

        st.warning(
            f"⚠️ {status}"
        )

    elif status == T("normal"):

        st.success(
            f"✓ {status}"
        )

    else:

        st.info(
            status
        )

    st.write(
        result["details"]
    )

    st.caption(
        T("no_diagnosis")
    )


def pose_history_section():

    st.subheader(
        T("pose_history")
    )

    df = get_pose_history(
        st.session_state.phone
    )

    if df.empty:

        st.info(
            U("no_history")
        )

        return

    st.dataframe(
        df,
        use_container_width=True
    )

    chart_df = df.copy()

    chart_df["timestamp"] = pd.to_datetime(
        chart_df["timestamp"]
    )

    chart_df = chart_df.sort_values(
        "timestamp"
    )

    chart_data = chart_df.set_index(
        "timestamp"
    )[
        [
            "left_knee_angle",
            "right_knee_angle"
        ]
    ]

    st.line_chart(
        chart_data
    )


def mediapipe_pose_page():

    st.title(
        "📷 " + T("pose")
    )

    st.warning(
        T("no_diagnosis")
    )

    if not MP_AVAILABLE:

        st.error(
            "MediaPipe is not installed."
        )

        return

    if not os.path.exists(
        MODEL_PATH
    ):

        st.error(
            f"Pose model not found:\n{MODEL_PATH}"
        )

        return

    tabs = st.tabs(
        [
            T("capture"),
            T("upload_photo"),
            T("upload_video"),
            T("pose_history")
        ]
    )

    # -----------------------------------------------------
    # CAMERA
    # -----------------------------------------------------

    with tabs[0]:

        st.subheader(
            "📷 Camera"
        )

        camera_image = st.camera_input(
            U("take_picture")
        )

        if camera_image:

            if st.button(
                U("analyze_camera"),
                key="analyze_camera"
            ):

                processed, result, error = process_image(
                    camera_image.getvalue()
                )

                if error:

                    st.error(error)

                else:

                    st.image(
                        processed,
                        caption=U("processed_pose")
                    )

                    display_pose_result(
                        result
                    )

                    save_pose_result(
                        st.session_state.phone,
                        "camera",
                        result["left_knee"],
                        result["right_knee"],
                        result["left_hip"],
                        result["right_hip"],
                        result["landmarks_count"],
                        result["status"],
                        result["details"]
                    )

                    st.session_state.last_pose = result

                    st.success(
                        U("pose_saved")
                    )

    # -----------------------------------------------------
    # PHOTO
    # -----------------------------------------------------

    with tabs[1]:

        uploaded_photo = st.file_uploader(
            U("upload_a_photo"),
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            key="pose_photo"
        )

        if uploaded_photo:

            st.image(
                uploaded_photo,
                caption=U("original_image")
            )

            if st.button(
                U("analyze_photo"),
                key="analyze_photo"
            ):

                processed, result, error = process_image(
                    uploaded_photo.getvalue()
                )

                if error:

                    st.error(error)

                else:

                    st.image(
                        processed,
                        caption=U("landmark_pose")
                    )

                    display_pose_result(
                        result
                    )

                    save_pose_result(
                        st.session_state.phone,
                        "photo",
                        result["left_knee"],
                        result["right_knee"],
                        result["left_hip"],
                        result["right_hip"],
                        result["landmarks_count"],
                        result["status"],
                        result["details"]
                    )

                    st.session_state.last_pose = result

    # -----------------------------------------------------
    # VIDEO
    # -----------------------------------------------------

    with tabs[2]:

        uploaded_video = st.file_uploader(
            U("movement_video"),
            type=[
                "mp4",
                "mov",
                "avi",
                "mkv"
            ],
            key="pose_video"
        )

        if uploaded_video:

            st.video(
                uploaded_video
            )

            if st.button(
                U("process_video"),
                key="process_video"
            ):

                with st.spinner(
                    U("processing_video")
                ):

                    processed_video, result, error = process_video(
                        uploaded_video.getvalue()
                    )

                if error:

                    st.error(error)

                else:

                    st.success(
                        U("video_done")
                    )

                    st.video(
                        processed_video
                    )

                    display_pose_result(
                        result
                    )

                    save_pose_result(
                        st.session_state.phone,
                        "video",
                        result["left_knee"],
                        result["right_knee"],
                        result["left_hip"],
                        result["right_hip"],
                        result["landmarks_count"],
                        result["status"],
                        result["details"]
                    )

                    st.session_state.last_pose = result

    # -----------------------------------------------------
    # HISTORY
    # -----------------------------------------------------

    with tabs[3]:

        pose_history_section()


# =========================================================
# SENSOR PAGE
# =========================================================
# =========================================================
# GAIT / SENSOR RISK ASSESSMENT
# =========================================================

def classify_sensor_parameter(parameter, value):
    """Classify a supplied gait parameter using the reference table."""
    if value is None:
        return "insufficient"

    value = float(value)

    if parameter == "Symmetry Index (SI)":
        # Reference: 0%-10% normal, 10%-20% moderate, >20% risk.
        if value <= 10:
            return "normal"
        if value <= 20:
            return "moderate"
        return "risk"

    if parameter == "Cadence":
        # The supplied image shows 110+ as normal, below 70 as risk,
        # and 70-90 as moderate. Because the image also contains
        # "90-110+" in the normal cell, this prototype uses 70-<110
        # as moderate so every input has one category.
        if value >= 110:
            return "normal"
        if value >= 70:
            return "moderate"
        return "risk"

    if parameter == "Peak Knee Flexion Angle":
        # Reference: 60-70+ normal, 45-60 moderate, below 45 risk.
        if value >= 60:
            return "normal"
        if value >= 45:
            return "moderate"
        return "risk"

    if parameter == "Gait Speed":
        # Reference: >1.2 m/s normal, 0.8-1.2 m/s moderate, <0.8 m/s risk.
        if value > 1.2:
            return "normal"
        if value >= 0.8:
            return "moderate"
        return "risk"

    return "insufficient"


def sensor_status_label(status):
    if status == "normal":
        return T("normal_status")
    if status == "moderate":
        return T("moderate_status")
    if status == "risk":
        return T("risk_status")
    return "N/A"


def overall_status(statuses):
    """Combine parameter statuses without claiming a medical diagnosis."""
    clean = [s for s in statuses if s in {"normal", "moderate", "risk"}]
    if not clean:
        return "insufficient"
    if "risk" in clean:
        return "risk"
    if "moderate" in clean:
        return "moderate"
    return "normal"


def get_latest_sensor_assessment(phone):
    conn = get_db()
    row = conn.execute(
        """
        SELECT timestamp, data
        FROM sensor_data
        WHERE phone=? AND sensor_type='Gait Parameters'
        ORDER BY id DESC
        LIMIT 1
        """,
        (phone,)
    ).fetchone()
    conn.close()

    if not row:
        return None

    try:
        data = json.loads(row["data"])
        data["timestamp"] = row["timestamp"]
        return data
    except Exception:
        return None


def display_status(status, label=None):
    text_label = label or sensor_status_label(status)
    if status == "risk":
        st.error("⚠️ " + text_label)
    elif status == "moderate":
        st.warning("⚠️ " + text_label)
    elif status == "normal":
        st.success("✓ " + text_label)
    else:
        st.info(text_label)


def reference_threshold_table():
    rows = [
        [T("symmetry_index"), "Percentage / Ratio difference", "0%-10%", "10%-20%", ">20%"],
        [T("cadence"), "Steps per minute", "110+", "70-<110*", "<70"],
        [T("peak_knee_flexion"), "Degree", "60+", "45-<60", "<45"],
        [T("gait_speed"), "Meters per second", ">1.2 m/s", "0.8-1.2 m/s", "<0.8 m/s"],
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "PARAMETER" if st.session_state.language == "English" else "पैरामीटर",
            "UNIT/DESCRIPTION" if st.session_state.language == "English" else "इकाई/विवरण",
            "NORMAL" if st.session_state.language == "English" else "सामान्य",
            "MODERATE" if st.session_state.language == "English" else "मध्यम",
            "RISK" if st.session_state.language == "English" else "जोखिम",
        ]
    )



def sensor_page():

    st.title("📡 " + T("sensor"))
    st.info(T("enter_parameters"))

    st.subheader(T("sensor_parameters"))

    # Reference table supplied by the user.
    st.caption(T("source_thresholds"))
    st.dataframe(reference_threshold_table(), use_container_width=True, hide_index=True)
    st.caption(T("cadence_note"))

    c1, c2 = st.columns(2)

    with c1:
        si = st.number_input(
            T("symmetry_index") + " (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.1,
            key="sensor_si"
        )

        cadence = st.number_input(
            T("cadence") + " (steps/min)",
            min_value=0.0,
            max_value=300.0,
            value=0.0,
            step=1.0,
            key="sensor_cadence"
        )

    with c2:
        peak_knee = st.number_input(
            T("peak_knee_flexion") + " (°)",
            min_value=0.0,
            max_value=180.0,
            value=0.0,
            step=1.0,
            key="sensor_peak_knee"
        )

        gait_speed = st.number_input(
            T("gait_speed") + " (m/s)",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.01,
            key="sensor_gait_speed"
        )

    if st.button(T("save_gait"), type="primary", use_container_width=True):
        values = {
            "Symmetry Index (SI)": si,
            "Cadence": cadence,
            "Peak Knee Flexion Angle": peak_knee,
            "Gait Speed": gait_speed,
        }

        statuses = {
            key: classify_sensor_parameter(key, value)
            for key, value in values.items()
        }

        assessment = {
            "symmetry_index": si,
            "cadence": cadence,
            "peak_knee_flexion": peak_knee,
            "gait_speed": gait_speed,
            "symmetry_index_status": statuses["Symmetry Index (SI)"],
            "cadence_status": statuses["Cadence"],
            "peak_knee_flexion_status": statuses["Peak Knee Flexion Angle"],
            "gait_speed_status": statuses["Gait Speed"],
            "overall_status": overall_status(list(statuses.values())),
        }

        save_sensor_data(
            st.session_state.phone,
            "Gait Parameters",
            assessment
        )

        st.session_state.last_sensor = assessment

        st.success(T("save_sensor_success"))

    assessment = st.session_state.last_sensor or get_latest_sensor_assessment(
        st.session_state.phone
    )

    if assessment:
        st.divider()
        st.subheader(T("parameter_results"))

        rows = [
            [T("symmetry_index"), assessment["symmetry_index"], "%", assessment["symmetry_index_status"]],
            [T("cadence"), assessment["cadence"], "steps/min", assessment["cadence_status"]],
            [T("peak_knee_flexion"), assessment["peak_knee_flexion"], "°", assessment["peak_knee_flexion_status"]],
            [T("gait_speed"), assessment["gait_speed"], "m/s", assessment["gait_speed_status"]],
        ]
        df = pd.DataFrame(
            rows,
            columns=["Parameter", "Value", "Unit", "Status"]
        )
        df["Status"] = df["Status"].map(sensor_status_label)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader(T("overall_result"))
        display_status(assessment["overall_status"])

    st.divider()

    # Existing IMU and FSR inputs are retained.
    tabs = st.tabs([T("imu"), T("fsr"), "History"])

    with tabs[0]:
        st.subheader(T("imu"))
        st.write(U("imu_description"))

        c1, c2 = st.columns(2)
        with c1:
            ax = st.number_input("Accelerometer X", value=0.0, step=0.01, key="imu_ax")
            ay = st.number_input("Accelerometer Y", value=0.0, step=0.01, key="imu_ay")
            az = st.number_input("Accelerometer Z", value=0.0, step=0.01, key="imu_az")
        with c2:
            gx = st.number_input("Gyroscope X", value=0.0, step=0.01, key="imu_gx")
            gy = st.number_input("Gyroscope Y", value=0.0, step=0.01, key="imu_gy")
            gz = st.number_input("Gyroscope Z", value=0.0, step=0.01, key="imu_gz")

        if st.button(U("save_imu"), key="save_imu"):
            data = {
                "accelerometer_x": ax, "accelerometer_y": ay, "accelerometer_z": az,
                "gyroscope_x": gx, "gyroscope_y": gy, "gyroscope_z": gz
            }
            save_sensor_data(st.session_state.phone, "IMU", data)
            st.session_state.last_sensor = assessment
            st.success(U("save_sensor_success"))

    with tabs[1]:
        st.subheader(T("fsr"))
        fsr_value = st.number_input(U("fsr_value"), min_value=0.0, value=0.0, step=0.1, key="fsr_value")
        fsr_left = st.number_input(U("left_fsr"), min_value=0.0, value=0.0, step=0.1, key="fsr_left")
        fsr_right = st.number_input(U("right_fsr"), min_value=0.0, value=0.0, step=0.1, key="fsr_right")

        if st.button(U("save_fsr"), key="save_fsr"):
            data = {"fsr_value": fsr_value, "left_fsr": fsr_left, "right_fsr": fsr_right}
            save_sensor_data(st.session_state.phone, "FSR", data)
            st.success(U("save_sensor_success"))

    with tabs[2]:
        df = get_sensor_history(st.session_state.phone)
        if df.empty:
            st.info(U("no_sensor_data"))
        else:
            st.dataframe(df, use_container_width=True)


# =========================================================
# KNEE ANALYSIS
# =========================================================

def knee_analysis_page():

    st.title(
        "🦵 " + T("knee")
    )

    st.warning(
        T("no_diagnosis")
    )

    profile = load_profile(
        st.session_state.phone
    )

    if profile:

        st.subheader(
            "Profile Information"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.write(
                f"**Name:** "
                f"{profile.get('name') or 'Not provided'}"
            )

        with c2:

            st.write(
                f"**Height:** "
                f"{profile.get('height') or 'Not provided'} cm"
            )

        with c3:

            st.write(
                f"**Weight:** "
                f"{profile.get('weight') or 'Not provided'} kg"
            )

        if profile.get(
            "previous_injury"
        ):

            st.info(
                "**Previous injury:** "
                + profile["previous_injury"]
            )

    pose = st.session_state.last_pose

    if pose is None:

        df = get_pose_history(
            st.session_state.phone
        )

        if not df.empty:

            row = df.iloc[0]

            pose = {

                "left_knee":
                    row["left_knee_angle"],

                "right_knee":
                    row["right_knee_angle"],

                "left_hip":
                    row["left_hip_angle"],

                "right_hip":
                    row["right_hip_angle"],

                "landmarks_count":
                    row["landmarks_count"],

                "status":
                    row["marker_status"],

                "details":
                    "Loaded from pose history."
            }

    if pose is None:

        st.info(
            "First perform a MediaPipe Pose analysis."
        )

        return

    st.subheader(
        "Latest Pose Measurements"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Left Knee Angle",
            (
                f"{pose['left_knee']:.1f}°"
                if pose["left_knee"] is not None
                else "N/A"
            )
        )

        st.metric(
            "Left Hip Angle",
            (
                f"{pose['left_hip']:.1f}°"
                if pose["left_hip"] is not None
                else "N/A"
            )
        )

    with c2:

        st.metric(
            "Right Knee Angle",
            (
                f"{pose['right_knee']:.1f}°"
                if pose["right_knee"] is not None
                else "N/A"
            )
        )

        st.metric(
            "Right Hip Angle",
            (
                f"{pose['right_hip']:.1f}°"
                if pose["right_hip"] is not None
                else "N/A"
            )
        )

    st.subheader(
        "Screening Summary"
    )

    if pose["status"] == T("risk"):

        st.warning(
            "⚠️ " + pose["status"]
        )

    elif pose["status"] == T("normal"):

        st.success(
            "✓ " + pose["status"]
        )

    else:

        st.info(
            pose["status"]
        )

    st.write(
        pose["details"]
    )

    # -----------------------------------------------------
    # SENSOR INFORMATION
    # -----------------------------------------------------

    sensor_df = get_sensor_history(
        st.session_state.phone
    )

    if not sensor_df.empty:

        st.subheader(
            "Latest Sensor Data"
        )

        st.dataframe(
            sensor_df.head(5),
            use_container_width=True
        )

    st.info(
        "The final screening result should be interpreted "
        "as a prototype risk-marker indication, not as a "
        "clinical diagnosis."
    )


# =========================================================
# ASK QUESTION PAGE
# =========================================================

def ask_question_page():

    st.title(
        "🤖 " + T("question")
    )

    st.write(
        "Ask the local AI assistant using text or your voice."
    )

    # -----------------------------------------------------
    # OLLAMA MODELS
    # -----------------------------------------------------

    models = get_ollama_models()

    if not models:

        st.warning(
            "No local Ollama model was found."
        )

        st.code(
            "ollama list"
        )

        st.write(
            "Install a local model while internet is "
            "available, for example:"
        )

        st.code(
            "ollama pull phi:latest"
        )

        st.info(
            "Speech-to-text with Vosk is separate and "
            "can work locally once its model is installed."
        )

    model = None

    if models:

        model = st.selectbox(
            T("model"),
            models
        )

        st.success(
            f"Local Ollama model available: {model}"
        )

    # -----------------------------------------------------
    # MICROPHONE
    # -----------------------------------------------------

    st.subheader(
        "🎤 Offline Speech to Text"
    )

    if not MIC_AVAILABLE:

        st.error(
            "streamlit-mic-recorder is not installed."
        )

        st.code(
            "pip install streamlit-mic-recorder"
        )

    else:

        audio = mic_recorder(

            start_prompt=T(
                "start_recording"
            ),

            stop_prompt=T(
                "stop_recording"
            ),

            just_once=True,

            format="wav",

            key="voice_question_recorder"
        )

        if audio:

            audio_bytes = audio.get(
                "bytes"
            )

            if audio_bytes:

                st.audio(
                    audio_bytes,
                    format="audio/wav"
                )

                with st.spinner(
                    T("listening")
                ):

                    voice_text, stt_error = speech_to_text(
                        audio_bytes
                    )

                if stt_error:

                    st.error(
                        stt_error
                    )

                else:

                    st.session_state[
                        "voice_question_text"
                    ] = voice_text

                    st.success(
                        T("speech_text")
                    )

                    st.info(
                        voice_text
                    )

    # -----------------------------------------------------
    # TEXT QUESTION
    # -----------------------------------------------------

    question = st.text_area(

        "Question",

        value=st.session_state.get(
            "voice_question_text",
            ""
        ),

        height=120,

        placeholder=T(
            "question_placeholder"
        )
    )

    # -----------------------------------------------------
    # PREVIOUS INJURY
    # -----------------------------------------------------

    profile = load_profile(
        st.session_state.phone
    )

    if profile:

        injury = profile.get(
            "previous_injury"
        )

        if injury:

            st.subheader(
                "Previous Injury Information"
            )

            st.info(
                injury
            )

    # -----------------------------------------------------
    # ASK
    # -----------------------------------------------------

    if st.button(
        "🤖 " + T("ask"),
        type="primary",
        use_container_width=True
    ):

        question = question.strip()

        if not question:

            st.error(
                "Please enter or speak a question."
            )

        elif not models:

            st.error(
                "No local Ollama model is installed."
            )

        else:

            with st.spinner(
                "Local AI is generating an answer..."
            ):

                answer, error = ask_ollama(
                    question,
                    model
                )

            if error:

                st.error(
                    error
                )

            else:

                st.subheader(
                    "Answer"
                )

                st.write(
                    answer
                )

                input_type = (
                    "voice"
                    if st.session_state.get(
                        "voice_question_text"
                    ) == question
                    else "text"
                )

                save_question(
                    st.session_state.phone,
                    input_type,
                    question,
                    answer
                )

                st.success(
                    "Question and answer saved locally."
                )

    # -----------------------------------------------------
    # HISTORY
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        T("question_history")
    )

    history = get_question_history(
        st.session_state.phone
    )

    if history.empty:

        st.info(
            "No questions asked yet."
        )

    else:

        for _, row in history.iterrows():

            with st.expander(
                f"{row['timestamp']} — "
                f"{row['input_type']}"
            ):

                st.write(
                    "**Question:**"
                )

                st.write(
                    row["question"]
                )

                st.write(
                    "**Answer:**"
                )

                st.write(
                    row["answer"]
                )


# =========================================================
# RESULTS PAGE
# =========================================================

def results_page():
    st.title("📊 " + T("results_page"))
    st.warning(T("result_note"))

    profile = load_profile(st.session_state.phone)
    pose = st.session_state.last_pose
    if pose is None:
        df_pose = get_pose_history(st.session_state.phone)
        if not df_pose.empty:
            row = df_pose.iloc[0]
            pose = {
                "left_knee": row["left_knee_angle"],
                "right_knee": row["right_knee_angle"],
                "left_hip": row["left_hip_angle"],
                "right_hip": row["right_hip_angle"],
                "landmarks_count": row["landmarks_count"],
                "status": row["marker_status"],
                "details": row.get("details", "")
            }

    sensor = st.session_state.last_sensor or get_latest_sensor_assessment(
        st.session_state.phone
    )

    # Profile
    st.subheader(T("profile_summary"))
    if profile:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(T("name"), profile.get("name") or "-")
        c2.metric(T("sex"), profile.get("sex") or "-")
        c3.metric(T("height"), f"{profile.get('height') or '-'} cm")
        c4.metric(T("weight"), f"{profile.get('weight') or '-'} kg")
        if profile.get("previous_injury"):
            st.info(f"{T('injury')}: {profile['previous_injury']}")
    else:
        st.info("No profile information available." if st.session_state.language == "English" else "प्रोफ़ाइल जानकारी उपलब्ध नहीं है।")

    # MediaPipe Pose
    st.divider()
    st.subheader(T("pose_summary"))
    if pose:
        c1, c2, c3 = st.columns(3)
        c1.metric(U("left_knee"), f"{pose['left_knee']:.1f}°" if pose.get("left_knee") is not None else "N/A")
        c2.metric(U("right_knee"), f"{pose['right_knee']:.1f}°" if pose.get("right_knee") is not None else "N/A")
        c3.metric(T("landmarks"), int(pose.get("landmarks_count") or 0))
        if pose.get("status"):
            st.write(f"**{T('results_page')}:** {pose['status']}")
    else:
        st.info(U("no_pose_result"))

    # Sensor / gait parameters
    st.divider()
    st.subheader(T("sensor_parameters"))
    sensor_statuses = []
    if sensor:
        rows = [
            [T("symmetry_index"), sensor["symmetry_index"], "%", sensor["symmetry_index_status"]],
            [T("cadence"), sensor["cadence"], "steps/min", sensor["cadence_status"]],
            [T("peak_knee_flexion"), sensor["peak_knee_flexion"], "°", sensor["peak_knee_flexion_status"]],
            [T("gait_speed"), sensor["gait_speed"], "m/s", sensor["gait_speed_status"]],
        ]
        sensor_statuses = [
            sensor["symmetry_index_status"],
            sensor["cadence_status"],
            sensor["peak_knee_flexion_status"],
            sensor["gait_speed_status"],
        ]
        df = pd.DataFrame(rows, columns=["Parameter", "Value", "Unit", "Status"])
        df["Status"] = df["Status"].map(sensor_status_label)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info(T("no_sensor_assessment"))

    # Overall result from the available assessment pages.
    pose_code = None
    if pose:
        # Older saved pose results contain translated strings. Interpret them conservatively.
        pose_status = str(pose.get("status", ""))
        if pose_status in {"risk", "संभावित मूवमेंट मार्कर"}:
            pose_code = "risk"
        elif pose_status in {"normal", "कोई स्पष्ट मूवमेंट मार्कर नहीं"}:
            pose_code = "normal"

    combined = list(sensor_statuses)
    if pose_code:
        combined.append(pose_code)

    final_code = overall_status(combined)

    st.divider()
    st.subheader(T("overall_result"))
    display_status(final_code)
    st.caption(T("result_note"))

    # Recent questions are shown as supporting information, not used to make a risk classification.
    st.divider()
    st.subheader(T("latest_questions"))
    questions = get_question_history(st.session_state.phone)
    if questions.empty:
        st.info(U("no_questions"))
    else:
        for _, row in questions.head(3).iterrows():
            with st.expander(str(row["timestamp"])):
                st.write(f"**{U('question_label')}:** {row['question']}")
                st.write(f"**{U('answer_label')}:** {row['answer']}")


# =========================================================
# LANGUAGE PAGE
# =========================================================

def language_page():

    st.title(
        "🌐 " + T("language")
    )

    language = st.radio(
        "Select language",
        [
            "English",
            "Hindi"
        ],
        index=(
            0
            if st.session_state.language == "English"
            else 1
        )
    )

    if language != st.session_state.language:

        st.session_state.language = language

        st.success(
            T("language_saved")
        )

        st.rerun()

    st.info(
        "The application interface supports English and Hindi."
        if st.session_state.language == "English"
        else "एप्लिकेशन इंटरफेस अंग्रेज़ी और हिंदी का समर्थन करता है।"
    )

    st.warning(
        "Speech recognition language depends on the "
        "Vosk model installed in stt_model."
    )


# =========================================================
# LOGOUT
# =========================================================

def logout():

    # IMPORTANT:
    # We DO NOT delete the SQLite database.
    # Therefore profile, pose, sensor and question
    # information remains stored locally.

    st.session_state.logged_in = False
    st.session_state.phone = ""
    st.session_state.page = "Home"
    st.session_state.last_pose = None
    st.session_state.last_sensor = None
    st.session_state.voice_question_text = ""

    st.rerun()


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

def navigation():

    st.sidebar.title(
        "🦵 OA Screening"
    )

    st.sidebar.write(
        f"**User:** {st.session_state.phone}"
    )

    pages = [

        "Home",
        "Profile",
        "MediaPipe Pose",
        "Sensor Data",
        "Knee Analysis",
        "Ask a Question",
        "Results"
    ]

    labels = {

        "Home":
            T("home"),

        "Profile":
            T("profile"),

        "MediaPipe Pose":
            T("pose"),

        "Sensor Data":
            T("sensor"),

        "Knee Analysis":
            T("knee"),

        "Ask a Question":
            T("question"),

        "Results":
            T("results_page")
    }

    selected = st.sidebar.radio(
        U("navigation"),
        pages,
        index=pages.index(
            st.session_state.page
        ),
        format_func=lambda x: labels[x]
    )

    if selected != st.session_state.page:

        st.session_state.page = selected

        st.rerun()

    st.sidebar.divider()

    if st.sidebar.button(
        "🚪 " + T("logout"),
        use_container_width=True
    ):

        logout()


# =========================================================
# MAIN
# =========================================================

if not st.session_state.logged_in:

    login_page()

else:

    navigation()

    page = st.session_state.page

    if page == "Home":

        home_page()

    elif page == "Profile":

        profile_page()

    elif page == "MediaPipe Pose":

        mediapipe_pose_page()

    elif page == "Sensor Data":

        sensor_page()

    elif page == "Knee Analysis":

        knee_analysis_page()

    elif page == "Ask a Question":

        ask_question_page()

    elif page == "Results":

        results_page()

    elif page == "Language":

        language_page()