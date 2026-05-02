
import os
from deepface import DeepFace
try:
    DeepFace.build_model("SFace")
    print("SFace Model loaded and ready.")
except Exception as e:
    print(f"Model pre-load warning: {e}")
    
from flask import Flask, request, jsonify, Response
from flask_cors import CORS, cross_origin
import cv2
import json
import pymysql
import os
import threading
import time
import signal
import sys
import io
import csv
import math
import re
import numpy as np
from datetime import datetime
from deepface import DeepFace
from scipy.spatial import distance as dst
from werkzeug.security import generate_password_hash, check_password_hash  # ← BOTH needed
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)


_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:5501")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

CORS(app,
     resources={r"/*": {"origins": ALLOWED_ORIGINS}},
     supports_credentials=True,
     methods=["GET", "POST", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"])

@app.after_request
def after_request(response):
    # Echo back the request origin if it is in our allowed list
    origin = request.headers.get("Origin", "")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"]  = origin
    response.headers["Access-Control-Allow-Headers"]     = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"]     = "GET,PUT,POST,DELETE,OPTIONS"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

DB_CONFIG = {
    "host":     os.getenv("DB_HOST"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}

FACE_PHOTOS_DIR = os.path.join(os.path.dirname(__file__), 'student_faces')
os.makedirs(FACE_PHOTOS_DIR, exist_ok=True)

ALLOWED_FACULTIES = {
    "Physical Sciences": [
        "Computer Science",
        "Mathematics",
        "Physics"
    ],
    "Life Sciences": [
        "Biochemistry",
        "Microbiology",
        "Plant Biology"
    ],
    "Communication and Information Sciences": [
        "Information Technology",
        "Computer Science",
        "Mass Communication"
    ]
}

# ── Global camera state
frame_lock      = threading.Lock()
camera          = cv2.VideoCapture(0)
latest_frame    = None
face_is_aligned = False

#  HELPERS

def preprocess_image(file_path):
    """Brighten + sharpen a saved JPEG so DeepFace has an easier time."""
    img = cv2.imread(file_path)
    if img is None:
        return file_path
    img = cv2.convertScaleAbs(img, alpha=1.3, beta=20)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    h, w = img.shape[:2]
    if w < 300:
        img = cv2.resize(img, (300, int(h * 300 / w)))
    cv2.imwrite(file_path, img)
    return file_path


def haversine(lat1, lon1, lat2, lon2):
    """Returns distance in metres between two GPS coordinates."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def validate_faculty_dept(faculty, department):
    """Returns (True, None) if valid, (False, error_msg) otherwise."""
    if faculty not in ALLOWED_FACULTIES:
        return False, f"Invalid faculty: '{faculty}'. Not a recognised faculty."
    if department not in ALLOWED_FACULTIES[faculty]:
        return False, f"Invalid department: '{department}' does not belong to '{faculty}'."
    return True, None

#  CAMERA FEED ROUTES

def gen_frames():
    global latest_frame, camera, face_is_aligned
    center_x, center_y, radius = 320, 240, 170
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    while True:
        if not camera or not camera.isOpened():
            time.sleep(0.1)
            continue
        success, frame = camera.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        with frame_lock:
            latest_frame = frame.copy()
        display = frame.copy()
        small   = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        gray    = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces   = face_cascade.detectMultiScale(gray, 1.1, 5)
        color, aligned = (0, 0, 255), False
        if len(faces) > 0:
            x, y, w, h = [v * 4 for v in faces[0]]
            if math.sqrt((x + w // 2 - center_x) ** 2 + (y + h // 2 - center_y) ** 2) < 60:
                color, aligned = (0, 255, 0), True
        face_is_aligned = aligned
        cv2.circle(display, (center_x, center_y), radius, color, 3)
        ret, buf = cv2.imencode('.jpg', display)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.03)


@app.route('/video_feed')
def video_feed():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/check_alignment')
def check_alignment():
    return jsonify({"aligned": face_is_aligned})


@app.route('/shutdown_camera', methods=['POST'])
def shutdown_camera():
    global camera
    if camera and camera.isOpened():
        camera.release()
    return jsonify({"status": "camera_off"})

#  SECURITY — serve faculty/dept list to frontend

@app.route('/get_faculties', methods=['GET'])
def get_faculties():
    return jsonify({"status": "success", "faculties": ALLOWED_FACULTIES})

#  REGISTRATION — text only (Lecturers)

@app.route('/register', methods=['POST', 'OPTIONS'])
def register_user():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
    db = None
    try:
        data     = request.json
        role     = data.get('role', '').strip()
        user_id  = data.get('userId', '').strip().upper()
        f_name   = data.get('firstName', '').strip()
        l_name   = data.get('lastName',  '').strip()
        password = data.get('password', '').strip()
        faculty  = data.get('faculty')
        dept     = data.get('department')

        if not all([role, user_id, f_name, l_name, password]):
            return jsonify({"status": "error", "message": "All fields are required."})
        if len(password) < 4:
            return jsonify({"status": "error", "message": "Password must be at least 4 characters."})

        hashed_pw = generate_password_hash(password)

        if role == "Student":
            if not user_id.startswith("STU-"):
                return jsonify({"status": "error", "message": "Student ID must start with 'STU-'."})
            valid, err = validate_faculty_dept(faculty, dept)
            if not valid:
                return jsonify({"status": "error", "message": err})
        elif role == "Lecturer":
            if not user_id.startswith("LEC-"):
                return jsonify({"status": "error", "message": "Lecturer ID must start with 'LEC-'."})
            faculty = None
            dept    = None
        else:
            return jsonify({"status": "error", "message": "Invalid role."})

        table, id_col = ("lecturers", "lecturer_id") if role == "Lecturer" else ("students", "student_id")

        db = pymysql.connect(**DB_CONFIG)
        with db.cursor() as cursor:
            cursor.execute(f"SELECT {id_col} FROM {table} WHERE {id_col} = %s", (user_id,))
            if cursor.fetchone():
                return jsonify({"status": "error", "message": "ID already registered."})

            if role == "Student":
                cursor.execute(
                    f"INSERT INTO {table} ({id_col}, first_name, last_name, password, faculty, department, face_encoding, photo_path) "
                    f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (user_id, f_name, l_name, hashed_pw, faculty, dept, "[]", "no_photo_yet.jpg")
                )
            else:
                cursor.execute(
                    f"INSERT INTO {table} ({id_col}, first_name, last_name, password, face_encoding, photo_path) "
                    f"VALUES (%s, %s, %s, %s, %s, %s)",
                    (user_id, f_name, l_name, hashed_pw, "[]", "no_photo_yet.jpg")
                )
            db.commit()
            return jsonify({"status": "success", "message": "Registration Successful!"})

    except Exception as e:
        print(f"[REGISTER ERROR] {e}")
        return jsonify({"status": "error", "message": f"Backend Error: {str(e)}"})
    finally:
        if db:
            db.close()

#  REGISTRATION WITH FACE CAPTURE (Students)

@app.route('/register_with_face', methods=['POST', 'OPTIONS'])
def register_with_face():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    db         = None
    photo_path = None
    try:
        user_id    = request.form.get('userId',     '').strip().upper()
        f_name     = request.form.get('firstName',  '').strip()
        l_name     = request.form.get('lastName',   '').strip()
        password   = request.form.get('password',   '').strip()
        faculty    = request.form.get('faculty',    '').strip()
        department = request.form.get('department', '').strip()

        if not all([user_id, f_name, l_name, password]):
            return jsonify({"status": "error", "message": "All fields are required."})
        if len(password) < 4:
            return jsonify({"status": "error", "message": "Password must be at least 4 characters."})
        if not user_id.startswith("STU-"):
            return jsonify({"status": "error", "message": "Student ID must start with 'STU-'."})

        # Server-side faculty/dept validation
        valid, err = validate_faculty_dept(faculty, department)
        if not valid:
            return jsonify({"status": "error", "message": err})

        hashed_pw = generate_password_hash(password)

        face_file = request.files.get('face')
        if not face_file:
            return jsonify({"status": "error", "message": "Face image is required for registration."})

        file_bytes = np.frombuffer(face_file.read(), np.uint8)
        img        = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"status": "error", "message": "Could not decode face image. Please try again."})

        # Duplicate check before touching disk
        db = pymysql.connect(**DB_CONFIG)
        with db.cursor() as cursor:
            cursor.execute("SELECT student_id FROM students WHERE student_id = %s", (user_id,))
            if cursor.fetchone():
                return jsonify({"status": "error", "message": "This Student ID is already registered."})

        # Save + preprocess face photo
        safe_id    = re.sub(r'[^a-zA-Z0-9_\-]', '_', user_id)
        filename   = f"{safe_id}.jpg"
        photo_path = os.path.join(FACE_PHOTOS_DIR, filename)
        cv2.imwrite(photo_path, img)
        print(f"[REG_FACE] Saved: {photo_path}")
        preprocess_image(photo_path)

        # Confirm DeepFace can detect a face
        detected = False
        for backend in ["opencv", "ssd"]:
            try:
                DeepFace.represent(
                    img_path=photo_path,
                    model_name="SFace",
                    enforce_detection=True,
                    detector_backend=backend
                )
                detected = True
                break
            except ValueError:
                continue

        if not detected:
            if os.path.exists(photo_path):
                os.remove(photo_path)
            return jsonify({
                "status":  "error",
                "message": "No face detected in your photo. Ensure good lighting and face the camera directly."
            })

        # Insert with hashed password
        with db.cursor() as cursor:
            cursor.execute(
                "INSERT INTO students (student_id, first_name, last_name, password, faculty, department, face_encoding, photo_path) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (user_id, f_name, l_name, hashed_pw, faculty, department, "[]", photo_path)
            )
            db.commit()

        print(f"[REG_FACE] ✅ {user_id} registered with face.")
        return jsonify({"status": "success", "message": f"Registration complete! Welcome, {f_name}."})

    except Exception as e:
        if photo_path and os.path.exists(photo_path):
            os.remove(photo_path)
        print(f"[REG_FACE ERROR] {e}")
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"})
    finally:
        if db:
            db.close()

#  LOGIN — FIXED

@app.route('/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    data     = request.json
    user_id  = data.get('userId',   '').strip().upper()
    password = data.get('password', '').strip()   # plain-text from form

    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:

            # ── Try students table (fetch password hash too) ───────
            c.execute(
                "SELECT student_id, first_name, last_name, department, password, 'Student' AS role "
                "FROM students WHERE student_id = %s",
                (user_id,)
            )
            user = c.fetchone()

            if not user:
                # ── Try lecturers table ───────────────────────────
                c.execute(
                    "SELECT lecturer_id, first_name, last_name, password, 'Lecturer' AS role "
                    "FROM lecturers WHERE lecturer_id = %s",
                    (user_id,)
                )
                user = c.fetchone()

            if not user:
                return jsonify({"status": "error", "message": "Invalid credentials"})

            # ── check_password_hash verifies plain vs stored hash ─
            # This is the correct way — never compare two generate calls
            stored_hash = user.get('password', '')
            if not check_password_hash(stored_hash, password):
                return jsonify({"status": "error", "message": "Invalid credentials"})

            return jsonify({
                "status": "success",
                "user": {
                    "userId":     user.get('student_id') or user.get('lecturer_id'),
                    "firstName":  user['first_name'],
                    "lastName":   user['last_name'],
                    "role":       user['role'],
                    "department": user.get('department')
                }
            })

    except Exception as e:
        print(f"[LOGIN ERROR] {e}")
        return jsonify({"status": "error", "message": "Server error during login"}), 500
    finally:
        db.close()

#  ATTENDANCE — STEP 1: Distance check only (no DB write)

@app.route('/check_distance', methods=['POST', 'OPTIONS'])
def check_distance():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
    data = request.json
    uid  = data.get('userId')
    code = data.get('courseCode')
    try:
        student_lat = float(data.get('lat', 0))
        student_lng = float(data.get('lng', 0))
    except (ValueError, TypeError):
        return jsonify({"status": "error", "message": "Invalid coordinates."}), 400

    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute(
                "SELECT id,lat,lng FROM attendance_sessions WHERE course_code=%s AND status='Active'",
                (code,)
            )
            session = c.fetchone()
            if not session:
                return jsonify({"status": "error", "message": "No active session for this course."})
            if session['lat'] is None or session['lng'] is None:
                return jsonify({"status": "error", "message": "Session location not set by lecturer."})

            dist = haversine(student_lat, student_lng, float(session['lat']), float(session['lng']))
            if dist > 150:
                return jsonify({
                    "status":  "error",
                    "message": f"You are {int(dist)}m away — must be within 150m of the lecturer."
                })
            return jsonify({"status": "in_range", "distance": int(dist)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()

#  ATTENDANCE — STEP 2: Face verification + record insert

@app.route('/verify_face_attendance', methods=['POST', 'OPTIONS'])
def verify_face_attendance():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    uid  = request.form.get('userId',     '').strip()
    code = request.form.get('courseCode', '').strip()
    try:
        student_lat = float(request.form.get('lat', 0))
        student_lng = float(request.form.get('lng', 0))
    except (ValueError, TypeError):
        return jsonify({"status": "error", "message": "Invalid coordinates."}), 400

    face_file = request.files.get('face')
    if not face_file:
        return jsonify({"status": "error", "message": "No face image received."}), 400

    file_bytes = np.frombuffer(face_file.read(), np.uint8)
    live_img   = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if live_img is None:
        return jsonify({"status": "error", "message": "Could not decode face image."}), 400

    temp_path = os.path.join(FACE_PHOTOS_DIR, f"temp_attend_{uid}.jpg")
    cv2.imwrite(temp_path, live_img)
    preprocess_image(temp_path)

    db = None
    try:
        db = pymysql.connect(**DB_CONFIG)
        with db.cursor(pymysql.cursors.DictCursor) as c:

            # 1. Re-confirm session + distance
            c.execute(
                "SELECT id,lat,lng FROM attendance_sessions WHERE course_code=%s AND status='Active'",
                (code,)
            )
            session = c.fetchone()
            if not session:
                return jsonify({"status": "error", "message": "Session no longer active."})

            if session['lat'] and session['lng']:
                dist = haversine(student_lat, student_lng, float(session['lat']), float(session['lng']))
                if dist > 150:
                    return jsonify({"status": "error", "message": f"Location check failed — {int(dist)}m away."})

            # 2. Duplicate check
            c.execute(
                "SELECT id FROM attendance_records WHERE student_id=%s AND session_id=%s",
                (uid, session['id'])
            )
            if c.fetchone():
                return jsonify({"status": "error", "message": "Attendance already marked for this session."})

            # 3. Load registered face path
            c.execute("SELECT photo_path FROM students WHERE student_id=%s", (uid,))
            row = c.fetchone()
            if not row or not row['photo_path'] or row['photo_path'] == 'no_photo_yet.jpg':
                return jsonify({
                    "status":  "error",
                    "message": "No face registered for this student. Please re-register."
                })

            registered_path = row['photo_path']
            if not os.path.exists(registered_path):
                return jsonify({
                    "status":  "error",
                    "message": "Stored face photo not found on server. Please re-register."
                })

            # 4. DeepFace verification
            try:
                result = DeepFace.verify(
                    img1_path        = registered_path,
                    img2_path        = temp_path,
                    model_name       = "SFace",
                    detector_backend = "opencv",
                    enforce_detection= False
                )
                verified  = result.get("verified",  False)
                distance  = result.get("distance",  1.0)
                threshold = result.get("threshold", 0.40)

                print(f"[FACE] {uid} | verified={verified} | dist={distance:.3f} | threshold={threshold:.3f}")

                if not verified:
                    return jsonify({
                        "status":  "face_mismatch",
                        "message": f"Face does not match (distance={distance:.2f}). Try again in better lighting."
                    })

            except Exception as face_err:
                print(f"[DEEPFACE ERROR] {face_err}")
                return jsonify({
                    "status":  "error",
                    "message": "Face verification error. Look directly at the camera and try again."
                })

            # 5. Record attendance
            c.execute(
                "INSERT INTO attendance_records (student_id, session_id, course_code, timestamp) VALUES (%s,%s,%s,NOW())",
                (uid, session['id'], code)
            )
            db.commit()

            return jsonify({
                "status":  "success",
                "message": f"Identity confirmed. Attendance marked for {code}!"
            })

    except Exception as e:
        print(f"[VERIFY_FACE_ATTENDANCE ERROR] {e}")
        return jsonify({"status": "error", "message": "Internal server error."}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if db:
            db.close()

#  SESSION ROUTES

@app.route('/check_active_session', methods=['GET'])
def check_active_session():
    code = request.args.get('courseCode')
    db   = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute(
                "SELECT id FROM attendance_sessions WHERE course_code=%s AND status='Active'",
                (code,)
            )
            s = c.fetchone()
            if s:
                return jsonify({"status": "active", "sessionId": s['id']})
            return jsonify({"status": "inactive", "message": "No session currently held."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    finally:
        db.close()


@app.route('/start_session', methods=['POST'])
def start_session():
    data        = request.json
    course_code = data.get('courseCode')
    lecturer_id = data.get('lecturerId') or data.get('userId')
    lat, lng    = data.get('lat'), data.get('lng')
    duration    = int(data.get('duration', 60))
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor() as c:
            c.execute(
                "UPDATE attendance_sessions SET status='Ended', ended_at=NOW() WHERE course_code=%s AND status='Active'",
                (course_code,)
            )
            c.execute(
                "INSERT INTO attendance_sessions (course_code, lecturer_id, lat, lng, duration, status, created_at) "
                "VALUES (%s,%s,%s,%s,%s,'Active',NOW())",
                (course_code, lecturer_id, lat, lng, duration)
            )
            db.commit()
            return jsonify({"status": "success", "message": f"Session for {course_code} is LIVE for {duration} minutes."})
    except Exception as e:
        db.rollback()
        return jsonify({"status": "error", "message": str(e)})
    finally:
        db.close()


@app.route('/end_session_by_course/<course_code>', methods=['POST'])
def end_session_by_course(course_code):
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor() as c:
            c.execute(
                "UPDATE attendance_sessions SET status='Completed', ended_at=NOW() "
                "WHERE course_code=%s AND status='Active' ORDER BY created_at DESC LIMIT 1",
                (course_code,)
            )
            db.commit()
            return jsonify({"status": "success", "message": "Session closed"})
    finally:
        db.close()

#  COURSE ROUTES
@app.route('/get_all_available_courses', methods=['GET'])
def get_all_available_courses():
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("SELECT course_name, course_code FROM courses")
            return jsonify({"status": "success", "courses": c.fetchall()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    finally:
        db.close()


@app.route('/get_courses/<lecturer_id>', methods=['GET'])
def get_courses(lecturer_id):
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("""
                SELECT c.course_name, c.course_code, s.status AS session_status, s.created_at, s.duration
                FROM courses c
                LEFT JOIN attendance_sessions s ON c.course_code = s.course_code AND s.status = 'Active'
                WHERE c.lecturer_id = %s
            """, (lecturer_id,))
            result = c.fetchall()
            for row in result:
                if row.get('created_at'):
                    row['created_at'] = row['created_at'].isoformat()
                if row.get('duration') is None:
                    row['duration'] = 60
            return jsonify(result or [])
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route('/create_course', methods=['POST'])
def create_course():
    data        = request.json
    course_name = data.get('courseName')
    course_code = data.get('courseCode', '').strip().upper()
    lecturer_id = data.get('lecturerId')
    if not all([course_code, course_name, lecturer_id]):
        return jsonify({"status": "error", "message": "Missing course details"}), 400
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor() as c:
            c.execute("SELECT course_code FROM courses WHERE course_code=%s", (course_code,))
            if c.fetchone():
                return jsonify({"status": "error", "message": f"Course code {course_code} already exists!"}), 400
            c.execute(
                "INSERT INTO courses (course_name, course_code, lecturer_id) VALUES (%s,%s,%s)",
                (course_name, course_code, lecturer_id)
            )
            db.commit()
            return jsonify({"status": "success", "message": "Course created successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route('/add_lecturer_to_course', methods=['POST'])
def add_lecturer_to_course():
    data = request.json
    lid  = data.get('lecturerId') or data.get('userId') or data.get('id')
    code = data.get('courseCode')
    if not lid:
        return jsonify({"status": "error", "message": "Lecturer ID is missing."})
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor() as c:
            c.execute("SELECT lecturer_id FROM courses WHERE course_code=%s", (code,))
            res = c.fetchone()
            if res:
                cur = res[0]
                if cur == lid:
                    return jsonify({"status": "error", "message": "Course already in your dashboard!"})
                if cur is not None and str(cur).strip() != "":
                    return jsonify({"status": "error", "message": "Course assigned to another lecturer."})
            c.execute("UPDATE courses SET lecturer_id=%s WHERE course_code=%s", (lid, code))
            if c.rowcount == 0:
                return jsonify({"status": "error", "message": "Course code not found."})
            db.commit()
            return jsonify({"status": "success", "message": "Course added to dashboard!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    finally:
        db.close()


@app.route('/delete_course/<course_code>', methods=['DELETE'])
def delete_course(course_code):
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor() as c:
            c.execute("UPDATE courses SET lecturer_id=NULL WHERE course_code=%s", (course_code,))
            db.commit()
            return jsonify({"status": "success", "message": "Course removed from dashboard"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()

# STUDENT ROUTES

@app.route('/enroll_student', methods=['POST'])
def enroll_student():
    data = request.json
    uid  = data.get('studentId') or data.get('userId')
    code = data.get('courseCode')
    if not uid or not code:
        return jsonify({"status": "error", "message": "Missing Student ID or Course Code."})
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor() as c:
            c.execute("SELECT student_id FROM students WHERE student_id=%s", (uid,))
            if not c.fetchone():
                return jsonify({"status": "error", "message": f"Student ID {uid} not found."})
            c.execute(
                "SELECT * FROM student_enrollments WHERE student_id=%s AND course_code=%s",
                (uid, code)
            )
            if c.fetchone():
                return jsonify({"status": "error", "message": "Already enrolled in this course."})
            c.execute(
                "INSERT INTO student_enrollments (student_id, course_code) VALUES (%s,%s)",
                (uid, code)
            )
            db.commit()
            return jsonify({"status": "success", "message": "Enrolled successfully!"})
    except Exception as e:
        print(f"[ENROLL ERROR] {e}")
        return jsonify({"status": "error", "message": "Database error occurred."})
    finally:
        db.close()


@app.route('/get_student_courses', methods=['GET'])
def get_student_courses():
    uid = request.args.get('userId')
    db  = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute(
                "SELECT c.course_name, c.course_code FROM courses c "
                "JOIN student_enrollments e ON c.course_code = e.course_code WHERE e.student_id=%s",
                (uid,)
            )
            return jsonify({"status": "success", "courses": c.fetchall()})
    finally:
        db.close()


@app.route('/get_student_attendance_log', methods=['GET'])
def get_student_attendance_log():
    uid = request.args.get('userId')
    db  = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("""
                SELECT s.course_code, c.course_name, s.created_at, s.status, s.ended_at
                FROM attendance_sessions s
                JOIN courses c ON s.course_code = c.course_code
                JOIN student_enrollments e ON s.course_code = e.course_code
                WHERE e.student_id = %s
                ORDER BY s.created_at DESC LIMIT 10
            """, (uid,))
            sessions = c.fetchall()
            log_data = []
            for s in sessions:
                dur = "---"
                if s['ended_at'] and s['created_at']:
                    dur = f"{int((s['ended_at'] - s['created_at']).total_seconds() // 60)} mins"
                log_data.append({
                    "date":     s['created_at'].strftime('%Y-%m-%d'),
                    "course":   f"{s['course_code']}: {s['course_name']}",
                    "status":   "Ongoing" if s['status'] == 'Active' else "Ended",
                    "duration": dur
                })
            return jsonify({"status": "success", "log": log_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    finally:
        db.close()


@app.route('/get_course_detail_stats', methods=['GET'])
def get_course_detail_stats():
    uid  = request.args.get('userId')
    code = request.args.get('courseCode')
    if not uid or not code:
        return jsonify({"status": "error", "message": "Missing userId or courseCode"})
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute(
                "SELECT id, created_at FROM attendance_sessions WHERE course_code=%s ORDER BY created_at DESC",
                (code,)
            )
            sessions = c.fetchall()
            c.execute(
                "SELECT session_id, timestamp FROM attendance_records WHERE student_id=%s AND course_code=%s",
                (uid, code)
            )
            records  = {r['session_id']: r['timestamp'] for r in c.fetchall()}
            log, attended = [], 0
            for s in sessions:
                present = s['id'] in records
                if present:
                    attended += 1
                log.append({
                    "date":   s['created_at'].strftime('%Y-%m-%d') if s['created_at'] else "N/A",
                    "status": "Present" if present else "Absent",
                    "time":   records[s['id']].strftime('%I:%M %p') if present and records.get(s['id']) else "--:--"
                })
            total   = len(sessions)
            percent = round((attended / total * 100), 1) if total > 0 else 0
            return jsonify({
                "status": "success", "course_code": code,
                "total_sessions": total, "attended_sessions": attended,
                "percentage": percent, "sessions": log
            })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    finally:
        db.close()


@app.route('/get_student_stats', methods=['GET'])
def get_student_stats():
    uid = request.args.get('userId')
    if not uid:
        return jsonify({"status": "error", "message": "No userId provided"})
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("SELECT COUNT(*) as c FROM student_enrollments WHERE student_id=%s", (uid,))
            c_count  = c.fetchone()['c']
            c.execute("SELECT COUNT(*) as a FROM attendance_records WHERE student_id=%s", (uid,))
            attended = c.fetchone()['a']
            c.execute(
                "SELECT COUNT(*) as t FROM attendance_sessions WHERE course_code IN "
                "(SELECT course_code FROM student_enrollments WHERE student_id=%s)",
                (uid,)
            )
            total  = c.fetchone()['t']
            rating = round((attended / total * 100), 1) if total > 0 else 0
            return jsonify({
                "status": "success", "rating": rating,
                "attended": attended, "total": total, "courses_count": c_count
            })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    finally:
        db.close()


@app.route('/drop_course', methods=['POST'])
def drop_course():
    data = request.json
    uid, code = data.get('userId'), data.get('courseCode')
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor() as c:
            c.execute(
                "DELETE FROM student_enrollments WHERE student_id=%s AND course_code=%s",
                (uid, code)
            )
            db.commit()
            return jsonify({"status": "success", "message": "Course dropped successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    finally:
        db.close()

#  LECTURER ANALYTICS ROUTES
@app.route('/get_students_by_course/<course_code>', methods=['GET'])
def get_students_by_course(course_code):
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("""
                SELECT
                    CONCAT(s.first_name,' ',s.last_name) AS full_name,
                    s.student_id, s.faculty, s.department,
                    (SELECT IFNULL(
                        (COUNT(a.id)*100.0 / NULLIF((SELECT COUNT(id) FROM attendance_sessions WHERE course_code=%s),0)),
                        0)
                     FROM attendance_records a
                     JOIN attendance_sessions sess ON a.session_id=sess.id
                     WHERE a.student_id=s.student_id AND sess.course_code=%s
                    ) AS attendance_rating
                FROM students s
                JOIN student_enrollments e ON s.student_id=e.student_id
                WHERE e.course_code=%s
            """, (course_code, course_code, course_code))
            return jsonify(c.fetchall() or [])
    finally:
        db.close()


@app.route('/get_session_history/<course_code>', methods=['GET'])
def get_session_history(course_code):
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("""
                SELECT s.id, s.created_at, s.ended_at, s.status, s.course_code,
                    (SELECT COUNT(*) FROM attendance_records r WHERE r.session_id=s.id) as attendee_count,
                    (SELECT COUNT(*) FROM student_enrollments e WHERE e.course_code=s.course_code) as total_enrolled
                FROM attendance_sessions s
                WHERE s.course_code=%s ORDER BY s.created_at DESC
            """, (course_code,))
            sessions = c.fetchall()
            for s in sessions:
                s['date_time'] = s['created_at'].strftime('%Y-%m-%d %H:%M')
                if s['ended_at']:
                    s['duration'] = f"{int((s['ended_at']-s['created_at']).total_seconds()//60)} mins"
                else:
                    s['duration'] = "Active" if s['status'] == 'Active' else "---"
                s['attendance_rate'] = (
                    f"{round((s['attendee_count']/s['total_enrolled'])*100,1)}%"
                    if s['total_enrolled'] > 0 else "0%"
                )
            return jsonify(sessions or [])
    finally:
        db.close()


@app.route('/get_session_attendance_detail/<session_id>', methods=['GET'])
def get_session_attendance_detail(session_id):
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("""
                SELECT CONCAT(s.first_name,' ',s.last_name) AS name,
                       s.student_id, s.faculty, s.department, 'Present' as status
                FROM students s
                JOIN attendance_records r ON s.student_id=r.student_id
                WHERE r.session_id=%s
            """, (session_id,))
            return jsonify(c.fetchall() or [])
    except Exception as e:
        print(f"[DETAIL ERROR] {e}")
        return jsonify([]), 500
    finally:
        db.close()


@app.route('/download_attendance_csv/<int:session_id>')
def download_attendance_csv(session_id):
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("""
                SELECT CONCAT(s.first_name,' ',s.last_name) AS full_name,
                       a.student_id, a.timestamp AS marked_at
                FROM attendance_records a
                JOIN students s ON a.student_id=s.student_id
                WHERE a.session_id=%s ORDER BY s.first_name ASC
            """, (session_id,))
            records = c.fetchall()
            if not records:
                return "No records found", 404
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['Student Name', 'Matric/ID No', 'Status', 'Time Marked'])
            for r in records:
                writer.writerow([
                    r['full_name'], r['student_id'], 'Present',
                    r['marked_at'].strftime('%H:%M:%S') if r['marked_at'] else "N/A"
                ])
            output.seek(0)
            return Response(
                output.getvalue(), mimetype="text/csv",
                headers={"Content-disposition": f"attachment; filename=LocusID_Attendance_Session_{session_id}.csv"}
            )
    except Exception as e:
        return str(e), 500
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════
#  ANNOUNCEMENTS / NOTIFICATIONS
# ═══════════════════════════════════════════════════════════════
@app.route('/post_announcement', methods=['POST'])
def post_announcement():
    data        = request.json
    course_code = data.get('course_code')
    message     = data.get('message')
    if not course_code or not message:
        return jsonify({"status": "error", "message": "Missing data"}), 400
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor() as c:
            c.execute(
                "INSERT INTO announcements (course_code, message) VALUES (%s,%s)",
                (course_code, message)
            )
            db.commit()
            return jsonify({"status": "success", "message": "Announcement posted!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route('/get_announcements/<course_code>', methods=['GET'])
def get_announcements(course_code):
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute(
                "SELECT id, message, created_at FROM announcements WHERE course_code=%s ORDER BY created_at DESC",
                (course_code,)
            )
            result = c.fetchall()
            for row in result:
                row['timestamp'] = row['created_at'].strftime('%Y-%m-%d %H:%M')
                del row['created_at']
            return jsonify({"status": "success", "announcements": result or []})
    finally:
        db.close()


@app.route('/get_notifications', methods=['GET'])
def get_notifications():
    uid = request.args.get('userId')
    db  = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("""
                SELECT id, course_code, message, created_at FROM announcements
                WHERE course_code IN (SELECT course_code FROM student_enrollments WHERE student_id=%s)
                ORDER BY created_at DESC
            """, (uid,))
            notifs = c.fetchall()
            for n in notifs:
                n['created_at'] = n['created_at'].strftime('%Y-%m-%d %H:%M')
            return jsonify({"status": "success", "notifications": notifs})
    finally:
        db.close()


@app.route('/delete_notification/<int:notif_id>', methods=['DELETE', 'OPTIONS'])
@cross_origin()
def delete_notification(notif_id):
    if request.method == 'OPTIONS':
        return jsonify({"status": "success"}), 200
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor() as c:
            c.execute("DELETE FROM announcements WHERE id=%s", (notif_id,))
            db.commit()
            return jsonify({"status": "success", "message": "Notification deleted"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════
#  ERROR HANDLER + SHUTDOWN
# ═══════════════════════════════════════════════════════════════
@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(error=str(e)), 404


def signal_handler(sig, frame):
    global camera
    if camera and camera.isOpened():
        camera.release()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)