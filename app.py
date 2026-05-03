import os
import io
import csv
import math
import re
import numpy as np
import cv2
import pymysql
import threading


from flask import Flask, request, jsonify, Response
from flask_cors import CORS, cross_origin
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:5501")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

CORS(app,
     resources={r"/*": {"origins": ALLOWED_ORIGINS}},
     supports_credentials=True,
     methods=["GET", "POST", "DELETE", "PUT", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"])

@app.after_request
def after_request(response):
    origin = request.headers.get("Origin", "")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"]      = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"]  = "GET,PUT,POST,DELETE,OPTIONS"
    return response

DB_CONFIG = {
    "host":     os.getenv("DB_HOST"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port":     int(os.getenv("DB_PORT", 16047)),
}

FACE_PHOTOS_DIR = os.path.join(os.path.dirname(__file__), "student_faces")
os.makedirs(FACE_PHOTOS_DIR, exist_ok=True)

# Max image dimension — keeps stored photos small to save RAM at compare time
MAX_IMG_SIZE = 400

ALLOWED_FACULTIES = {
    "Physical Sciences": ["Computer Science", "Mathematics", "Physics"],
    "Life Sciences":     ["Biochemistry", "Microbiology", "Plant Biology"],
    "Communication and Information Sciences": [
        "Information Technology", "Computer Science", "Mass Communication"
    ],
}

_mp_lock     = threading.Lock()
_mp_detector = None

def get_detector():
    global _mp_detector
    if _mp_detector is None:
        with _mp_lock:
            if _mp_detector is None:
                _mp_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,          # 0 = short-range (selfie), faster
                    min_detection_confidence=0.5
                )
    return _mp_detector

def resize_for_storage(img_bgr):

    h, w = img_bgr.shape[:2]
    if max(h, w) <= MAX_IMG_SIZE:
        return img_bgr
    scale = MAX_IMG_SIZE / max(h, w)
    return cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def count_faces(img_bgr):

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with _mp_lock:
        result = get_detector().process(rgb)
    return len(result.detections) if result.detections else 0


def face_histogram(img_bgr):
    """
    Crop to face bounding box, compute HSV histogram.
    Returns None if no face found.
    Uses the image already in memory — no disk re-read.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with _mp_lock:
        result = get_detector().process(rgb)
    if not result.detections:
        return None
    h, w = img_bgr.shape[:2]
    bb   = result.detections[0].location_data.relative_bounding_box
    x1   = max(0, int(bb.xmin * w))
    y1   = max(0, int(bb.ymin * h))
    x2   = min(w, int((bb.xmin + bb.width)  * w))
    y2   = min(h, int((bb.ymin + bb.height) * h))
    face = img_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face = cv2.resize(face, (64, 64))           # tiny — fast + low RAM
    hsv  = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def compare_faces(stored_path, live_img_bgr):
    """
    Compare stored JPEG (path) vs live image (numpy array).
    Returns (matched: bool, score_pct: float).
    Threshold: 70% histogram correlation.
    """
    # Load stored image — already small (MAX_IMG_SIZE) so low RAM
    stored_img = cv2.imread(stored_path)
    if stored_img is None:
        return False, 0.0

    h1 = face_histogram(stored_img)
    h2 = face_histogram(live_img_bgr)

    # Free stored image immediately
    del stored_img

    if h1 is None or h2 is None:
        return False, 0.0

    score     = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    score_pct = max(0.0, score) * 100
    print(f"[FACE] Histogram score: {score_pct:.1f}%")
    return score_pct >= 70, score_pct


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def validate_faculty_dept(faculty, department):
    if faculty not in ALLOWED_FACULTIES:
        return False, f"Invalid faculty: '{faculty}'."
    if department not in ALLOWED_FACULTIES[faculty]:
        return False, f"'{department}' does not belong to '{faculty}'."
    return True, None


@app.route("/get_faculties", methods=["GET"])
def get_faculties():
    return jsonify({"status": "success", "faculties": ALLOWED_FACULTIES})


@app.route("/register", methods=["POST", "OPTIONS"])
def register_user():
    if request.method == "OPTIONS": return jsonify({"status": "ok"}), 200
    db = None
    try:
        data     = request.json
        role     = data.get("role", "").strip()
        user_id  = data.get("userId", "").strip().upper()
        f_name   = data.get("firstName", "").strip()
        l_name   = data.get("lastName",  "").strip()
        password = data.get("password",  "").strip()
        faculty  = data.get("faculty")
        dept     = data.get("department")

        if not all([role, user_id, f_name, l_name, password]):
            return jsonify({"status": "error", "message": "All fields are required."})
        if len(password) < 4:
            return jsonify({"status": "error", "message": "Password must be at least 4 characters."})

        hashed = generate_password_hash(password)

        if role == "Student":
            if not user_id.startswith("STU-"):
                return jsonify({"status": "error", "message": "Student ID must start with 'STU-'."})
            valid, err = validate_faculty_dept(faculty, dept)
            if not valid: return jsonify({"status": "error", "message": err})
        elif role == "Lecturer":
            if not user_id.startswith("LEC-"):
                return jsonify({"status": "error", "message": "Lecturer ID must start with 'LEC-'."})
            faculty = dept = None
        else:
            return jsonify({"status": "error", "message": "Invalid role."})

        table, id_col = ("lecturers","lecturer_id") if role=="Lecturer" else ("students","student_id")
        db = pymysql.connect(**DB_CONFIG)
        with db.cursor() as c:
            c.execute(f"SELECT {id_col} FROM {table} WHERE {id_col}=%s", (user_id,))
            if c.fetchone():
                return jsonify({"status": "error", "message": "ID already registered."})
            if role == "Student":
                c.execute(
                    f"INSERT INTO {table} ({id_col},first_name,last_name,password,faculty,department,face_encoding,photo_path) "
                    f"VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",
                    (user_id, f_name, l_name, hashed, faculty, dept, "[]", "no_photo_yet.jpg"))
            else:
                c.execute(
                    f"INSERT INTO {table} ({id_col},first_name,last_name,password,face_encoding,photo_path) "
                    f"VALUES(%s,%s,%s,%s,%s,%s)",
                    (user_id, f_name, l_name, hashed, "[]", "no_photo_yet.jpg"))
            db.commit()
            return jsonify({"status": "success", "message": "Registration Successful!"})
    except Exception as e:
        print(f"[REGISTER ERROR] {e}")
        return jsonify({"status": "error", "message": f"Backend Error: {str(e)}"})
    finally:
        if db: db.close()
@app.route("/register_with_face", methods=["POST", "OPTIONS"])
def register_with_face():
    if request.method == "OPTIONS": return jsonify({"status": "ok"}), 200
    db = None
    photo_path = None
    try:
        user_id    = request.form.get("userId",     "").strip().upper()
        f_name     = request.form.get("firstName",  "").strip()
        l_name     = request.form.get("lastName",   "").strip()
        password   = request.form.get("password",   "").strip()
        faculty    = request.form.get("faculty",    "").strip()
        department = request.form.get("department", "").strip()

        if not all([user_id, f_name, l_name, password]):
            return jsonify({"status": "error", "message": "All fields are required."})
        if len(password) < 4:
            return jsonify({"status": "error", "message": "Password must be at least 4 characters."})
        if not user_id.startswith("STU-"):
            return jsonify({"status": "error", "message": "Student ID must start with 'STU-'."})

        valid, err = validate_faculty_dept(faculty, department)
        if not valid: return jsonify({"status": "error", "message": err})

        hashed = generate_password_hash(password)

        face_file = request.files.get("face")
        if not face_file:
            return jsonify({"status": "error", "message": "Face image required."})

        file_bytes = np.frombuffer(face_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"status": "error", "message": "Could not decode face image."})

        # Duplicate check
        db = pymysql.connect(**DB_CONFIG)
        with db.cursor() as c:
            c.execute("SELECT student_id FROM students WHERE student_id=%s", (user_id,))
            if c.fetchone():
                return jsonify({"status": "error", "message": "Student ID already registered."})

        # Resize to 400px max to save disk space
        h, w = img.shape[:2]
        if max(h, w) > 400:
            scale = 400 / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))

        # Save photo — no face detection, no AI
        safe_id    = re.sub(r"[^a-zA-Z0-9_\-]", "_", user_id)
        photo_path = os.path.join(FACE_PHOTOS_DIR, f"{safe_id}.jpg")
        cv2.imwrite(photo_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        del img

        with db.cursor() as c:
            c.execute(
                "INSERT INTO students (student_id,first_name,last_name,password,faculty,department,face_encoding,photo_path) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",
                (user_id, f_name, l_name, hashed, faculty, department, "[]", photo_path)
            )
            db.commit()

        return jsonify({"status": "success", "message": f"Registration complete! Welcome, {f_name}."})

    except Exception as e:
        if photo_path and os.path.exists(photo_path): os.remove(photo_path)
        print(f"[REG_FACE ERROR] {e}")
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"})
    finally:
        if db: db.close()


@app.route("/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS": return jsonify({"status": "ok"}), 200
    data     = request.json
    user_id  = data.get("userId",   "").strip().upper()
    password = data.get("password", "").strip()
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("SELECT student_id,first_name,last_name,department,password,'Student' AS role FROM students WHERE student_id=%s", (user_id,))
            user = c.fetchone()
            if not user:
                c.execute("SELECT lecturer_id,first_name,last_name,password,'Lecturer' AS role FROM lecturers WHERE lecturer_id=%s", (user_id,))
                user = c.fetchone()
            if not user:
                return jsonify({"status": "error", "message": "Invalid credentials"})
            if not check_password_hash(user["password"], password):
                return jsonify({"status": "error", "message": "Invalid credentials"})
            return jsonify({"status": "success", "user": {
                "userId":     user.get("student_id") or user.get("lecturer_id"),
                "firstName":  user["first_name"],
                "lastName":   user["last_name"],
                "role":       user["role"],
                "department": user.get("department"),
            }})
    except Exception as e:
        print(f"[LOGIN ERROR] {e}")
        return jsonify({"status": "error", "message": "Server error during login"}), 500
    finally:
        db.close()

@app.route("/check_distance", methods=["POST", "OPTIONS"])
def check_distance():
    if request.method == "OPTIONS": return jsonify({"status": "ok"}), 200
    data = request.json
    code = data.get("courseCode")
    try:
        lat = float(data.get("lat", 0))
        lng = float(data.get("lng", 0))
    except (ValueError, TypeError):
        return jsonify({"status": "error", "message": "Invalid coordinates."}), 400
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("SELECT id,lat,lng FROM attendance_sessions WHERE course_code=%s AND status='Active'", (code,))
            session = c.fetchone()
            if not session:
                return jsonify({"status": "error", "message": "No active session for this course."})
            if session["lat"] is None or session["lng"] is None:
                return jsonify({"status": "error", "message": "Session location not set by lecturer."})
            dist = haversine(lat, lng, float(session["lat"]), float(session["lng"]))
            if dist > 150:
                return jsonify({"status": "error", "message": f"You are {int(dist)}m away — must be within 150m."})
            return jsonify({"status": "in_range", "distance": int(dist)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()

@app.route("/verify_face_attendance", methods=["POST", "OPTIONS"])
def verify_face_attendance():
    if request.method == "OPTIONS": return jsonify({"status": "ok"}), 200

    uid  = request.form.get("userId",     "").strip().upper()
    code = request.form.get("courseCode", "").strip()
    try:
        lat = float(request.form.get("lat", 0))
        lng = float(request.form.get("lng", 0))
    except (ValueError, TypeError):
        return jsonify({"status": "error", "message": "Invalid coordinates."}), 400

    db = None
    try:
        db = pymysql.connect(**DB_CONFIG)
        with db.cursor(pymysql.cursors.DictCursor) as c:

            # Session + distance check
            c.execute("SELECT id,lat,lng FROM attendance_sessions WHERE course_code=%s AND status='Active'", (code,))
            session = c.fetchone()
            if not session:
                return jsonify({"status": "error", "message": "Session no longer active."})
            if session["lat"] and session["lng"]:
                dist = haversine(lat, lng, float(session["lat"]), float(session["lng"]))
                if dist > 150:
                    return jsonify({"status": "error", "message": f"Location check failed — {int(dist)}m away."})

            # Duplicate check
            c.execute("SELECT id FROM attendance_records WHERE student_id=%s AND session_id=%s", (uid, session["id"]))
            if c.fetchone():
                return jsonify({"status": "error", "message": "Attendance already marked for this session."})

            # Record attendance — face check skipped (free tier RAM limit)
            c.execute(
                "INSERT INTO attendance_records (student_id,session_id,course_code,timestamp) VALUES(%s,%s,%s,NOW())",
                (uid, session["id"], code)
            )
            db.commit()
            return jsonify({"status": "success", "message": f"Attendance marked for {code}!"})

    except Exception as e:
        print(f"[VERIFY ERROR] {e}")
        return jsonify({"status": "error", "message": "Internal server error."}), 500
    finally:
        if db: db.close()
#  SESSION ROUTES

@app.route("/check_active_session", methods=["GET"])
def check_active_session():
    code = request.args.get("courseCode")
    db   = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("SELECT id FROM attendance_sessions WHERE course_code=%s AND status='Active'", (code,))
            s = c.fetchone()
            return jsonify({"status":"active","sessionId":s["id"]} if s else {"status":"inactive","message":"No active session."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    finally:
        db.close()


@app.route("/start_session", methods=["POST"])
def start_session():
    data        = request.json
    course_code = data.get("courseCode")
    lecturer_id = data.get("lecturerId") or data.get("userId")
    lat, lng    = data.get("lat"), data.get("lng")
    duration    = int(data.get("duration", 60))
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor() as c:
            c.execute("UPDATE attendance_sessions SET status='Ended',ended_at=NOW() WHERE course_code=%s AND status='Active'", (course_code,))
            c.execute("INSERT INTO attendance_sessions (course_code,lecturer_id,lat,lng,duration,status,created_at) VALUES(%s,%s,%s,%s,%s,'Active',NOW())",
                      (course_code, lecturer_id, lat, lng, duration))
            db.commit()
            return jsonify({"status":"success","message":f"Session for {course_code} is LIVE for {duration} minutes."})
    except Exception as e:
        db.rollback()
        return jsonify({"status":"error","message":str(e)})
    finally:
        db.close()


@app.route("/end_session_by_course/<course_code>", methods=["POST"])
def end_session_by_course(course_code):
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor() as c:
            c.execute("UPDATE attendance_sessions SET status='Completed',ended_at=NOW() WHERE course_code=%s AND status='Active' ORDER BY created_at DESC LIMIT 1", (course_code,))
            db.commit()
            return jsonify({"status":"success","message":"Session closed"})
    finally:
        db.close()

#  COURSE ROUTES
@app.route("/get_all_available_courses", methods=["GET"])
def get_all_available_courses():
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("SELECT course_name,course_code FROM courses")
            return jsonify({"status":"success","courses":c.fetchall()})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)})
    finally:
        db.close()


@app.route("/get_courses/<lecturer_id>", methods=["GET"])
def get_courses(lecturer_id):
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("""SELECT c.course_name,c.course_code,s.status AS session_status,s.created_at,s.duration
                FROM courses c LEFT JOIN attendance_sessions s ON c.course_code=s.course_code AND s.status='Active'
                WHERE c.lecturer_id=%s""", (lecturer_id,))
            result = c.fetchall()
            for row in result:
                if row.get("created_at"): row["created_at"] = row["created_at"].isoformat()
                if row.get("duration") is None: row["duration"] = 60
            return jsonify(result or [])
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500
    finally:
        db.close()


@app.route("/create_course", methods=["POST"])
def create_course():
    data = request.json
    course_name = data.get("courseName")
    course_code = data.get("courseCode","").strip().upper()
    lecturer_id = data.get("lecturerId")
    if not all([course_code, course_name, lecturer_id]):
        return jsonify({"status":"error","message":"Missing course details"}), 400
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor() as c:
            c.execute("SELECT course_code FROM courses WHERE course_code=%s", (course_code,))
            if c.fetchone():
                return jsonify({"status":"error","message":f"Course code {course_code} already exists!"}), 400
            c.execute("INSERT INTO courses (course_name,course_code,lecturer_id) VALUES(%s,%s,%s)", (course_name, course_code, lecturer_id))
            db.commit()
            return jsonify({"status":"success","message":"Course created successfully"})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500
    finally:
        db.close()


@app.route("/add_lecturer_to_course", methods=["POST"])
def add_lecturer_to_course():
    data = request.json
    lid  = data.get("lecturerId") or data.get("userId") or data.get("id")
    code = data.get("courseCode")
    if not lid: return jsonify({"status":"error","message":"Lecturer ID is missing."})
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor() as c:
            c.execute("SELECT lecturer_id FROM courses WHERE course_code=%s", (code,))
            res = c.fetchone()
            if res:
                cur = res[0]
                if cur == lid: return jsonify({"status":"error","message":"Course already in your dashboard!"})
                if cur is not None and str(cur).strip() != "": return jsonify({"status":"error","message":"Course assigned to another lecturer."})
            c.execute("UPDATE courses SET lecturer_id=%s WHERE course_code=%s", (lid, code))
            if c.rowcount == 0: return jsonify({"status":"error","message":"Course code not found."})
            db.commit()
            return jsonify({"status":"success","message":"Course added to dashboard!"})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)})
    finally:
        db.close()


@app.route("/delete_course/<course_code>", methods=["DELETE"])
def delete_course(course_code):
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor() as c:
            c.execute("UPDATE courses SET lecturer_id=NULL WHERE course_code=%s", (course_code,))
            db.commit()
            return jsonify({"status":"success","message":"Course removed from dashboard"})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500
    finally:
        db.close()

#  STUDENT ROUTES

@app.route("/enroll_student", methods=["POST"])
def enroll_student():
    data = request.json
    uid  = data.get("studentId") or data.get("userId")
    code = data.get("courseCode")
    if not uid or not code: return jsonify({"status":"error","message":"Missing Student ID or Course Code."})
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor() as c:
            c.execute("SELECT student_id FROM students WHERE student_id=%s", (uid,))
            if not c.fetchone(): return jsonify({"status":"error","message":f"Student ID {uid} not found."})
            c.execute("SELECT * FROM student_enrollments WHERE student_id=%s AND course_code=%s", (uid, code))
            if c.fetchone(): return jsonify({"status":"error","message":"Already enrolled in this course."})
            c.execute("INSERT INTO student_enrollments (student_id,course_code) VALUES(%s,%s)", (uid, code))
            db.commit()
            return jsonify({"status":"success","message":"Enrolled successfully!"})
    except Exception as e:
        return jsonify({"status":"error","message":"Database error occurred."})
    finally:
        db.close()


@app.route("/get_student_courses", methods=["GET"])
def get_student_courses():
    uid = request.args.get("userId")
    db  = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("SELECT c.course_name,c.course_code FROM courses c JOIN student_enrollments e ON c.course_code=e.course_code WHERE e.student_id=%s", (uid,))
            return jsonify({"status":"success","courses":c.fetchall()})
    finally:
        db.close()


@app.route("/get_student_attendance_log", methods=["GET"])
def get_student_attendance_log():
    uid = request.args.get("userId")
    db  = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("""SELECT s.course_code,c.course_name,s.created_at,s.status,s.ended_at
                FROM attendance_sessions s JOIN courses c ON s.course_code=c.course_code
                JOIN student_enrollments e ON s.course_code=e.course_code
                WHERE e.student_id=%s ORDER BY s.created_at DESC LIMIT 10""", (uid,))
            sessions = c.fetchall()
            log_data = []
            for s in sessions:
                dur = "---"
                if s["ended_at"] and s["created_at"]:
                    dur = f"{int((s['ended_at']-s['created_at']).total_seconds()//60)} mins"
                log_data.append({
                    "date":     s["created_at"].strftime("%Y-%m-%d"),
                    "course":   f"{s['course_code']}: {s['course_name']}",
                    "status":   "Ongoing" if s["status"] == "Active" else "Ended",
                    "duration": dur
                })
            return jsonify({"status":"success","log":log_data})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)})
    finally:
        db.close()


@app.route("/get_course_detail_stats", methods=["GET"])
def get_course_detail_stats():
    uid  = request.args.get("userId")
    code = request.args.get("courseCode")
    if not uid or not code: return jsonify({"status":"error","message":"Missing params"})
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("SELECT id,created_at FROM attendance_sessions WHERE course_code=%s ORDER BY created_at DESC", (code,))
            sessions = c.fetchall()
            c.execute("SELECT session_id,timestamp FROM attendance_records WHERE student_id=%s AND course_code=%s", (uid, code))
            records  = {r["session_id"]: r["timestamp"] for r in c.fetchall()}
            log, attended = [], 0
            for s in sessions:
                present = s["id"] in records
                if present: attended += 1
                log.append({
                    "date":   s["created_at"].strftime("%Y-%m-%d") if s["created_at"] else "N/A",
                    "status": "Present" if present else "Absent",
                    "time":   records[s["id"]].strftime("%I:%M %p") if present and records.get(s["id"]) else "--:--"
                })
            total   = len(sessions)
            percent = round((attended/total*100), 1) if total > 0 else 0
            return jsonify({"status":"success","course_code":code,"total_sessions":total,"attended_sessions":attended,"percentage":percent,"sessions":log})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)})
    finally:
        db.close()


@app.route("/get_student_stats", methods=["GET"])
def get_student_stats():
    uid = request.args.get("userId")
    if not uid: return jsonify({"status":"error","message":"No userId provided"})
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("SELECT COUNT(*) as c FROM student_enrollments WHERE student_id=%s", (uid,))
            c_count  = c.fetchone()["c"]
            c.execute("SELECT COUNT(*) as a FROM attendance_records WHERE student_id=%s", (uid,))
            attended = c.fetchone()["a"]
            c.execute("SELECT COUNT(*) as t FROM attendance_sessions WHERE course_code IN (SELECT course_code FROM student_enrollments WHERE student_id=%s)", (uid,))
            total  = c.fetchone()["t"]
            rating = round((attended/total*100), 1) if total > 0 else 0
            return jsonify({"status":"success","rating":rating,"attended":attended,"total":total,"courses_count":c_count})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)})
    finally:
        db.close()


@app.route("/drop_course", methods=["POST"])
def drop_course():
    data = request.json
    uid, code = data.get("userId"), data.get("courseCode")
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor() as c:
            c.execute("DELETE FROM student_enrollments WHERE student_id=%s AND course_code=%s", (uid, code))
            db.commit()
            return jsonify({"status":"success","message":"Course dropped successfully"})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)})
    finally:
        db.close()

#  LECTURER ANALYTICS
@app.route("/get_students_by_course/<course_code>", methods=["GET"])
def get_students_by_course(course_code):
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("""SELECT CONCAT(s.first_name,' ',s.last_name) AS full_name,s.student_id,s.faculty,s.department,
                (SELECT IFNULL((COUNT(a.id)*100.0/NULLIF((SELECT COUNT(id) FROM attendance_sessions WHERE course_code=%s),0)),0)
                 FROM attendance_records a JOIN attendance_sessions sess ON a.session_id=sess.id
                 WHERE a.student_id=s.student_id AND sess.course_code=%s) AS attendance_rating
                FROM students s JOIN student_enrollments e ON s.student_id=e.student_id WHERE e.course_code=%s""",
                (course_code, course_code, course_code))
            return jsonify(c.fetchall() or [])
    finally:
        db.close()


@app.route("/get_session_history/<course_code>", methods=["GET"])
def get_session_history(course_code):
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("""SELECT s.id,s.created_at,s.ended_at,s.status,s.course_code,
                (SELECT COUNT(*) FROM attendance_records r WHERE r.session_id=s.id) as attendee_count,
                (SELECT COUNT(*) FROM student_enrollments e WHERE e.course_code=s.course_code) as total_enrolled
                FROM attendance_sessions s WHERE s.course_code=%s ORDER BY s.created_at DESC""", (course_code,))
            sessions = c.fetchall()
            for s in sessions:
                s["date_time"] = s["created_at"].strftime("%Y-%m-%d %H:%M")
                s["duration"]  = f"{int((s['ended_at']-s['created_at']).total_seconds()//60)} mins" if s["ended_at"] else ("Active" if s["status"]=="Active" else "---")
                s["attendance_rate"] = f"{round((s['attendee_count']/s['total_enrolled'])*100,1)}%" if s["total_enrolled"] > 0 else "0%"
            return jsonify(sessions or [])
    finally:
        db.close()


@app.route("/get_session_attendance_detail/<session_id>", methods=["GET"])
def get_session_attendance_detail(session_id):
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("""SELECT CONCAT(s.first_name,' ',s.last_name) AS name,s.student_id,s.faculty,s.department,'Present' as status
                FROM students s JOIN attendance_records r ON s.student_id=r.student_id WHERE r.session_id=%s""", (session_id,))
            return jsonify(c.fetchall() or [])
    except Exception as e:
        return jsonify([]), 500
    finally:
        db.close()


@app.route("/download_attendance_csv/<int:session_id>")
def download_attendance_csv(session_id):
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("""SELECT CONCAT(s.first_name,' ',s.last_name) AS full_name,a.student_id,a.timestamp AS marked_at
                FROM attendance_records a JOIN students s ON a.student_id=s.student_id
                WHERE a.session_id=%s ORDER BY s.first_name ASC""", (session_id,))
            records = c.fetchall()
            if not records: return "No records found", 404
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Student Name","Matric/ID No","Status","Time Marked"])
            for r in records:
                writer.writerow([r["full_name"], r["student_id"], "Present",
                                 r["marked_at"].strftime("%H:%M:%S") if r["marked_at"] else "N/A"])
            output.seek(0)
            return Response(output.getvalue(), mimetype="text/csv",
                headers={"Content-disposition": f"attachment; filename=LocusID_Attendance_Session_{session_id}.csv"})
    except Exception as e:
        return str(e), 500
    finally:
        db.close()


#  ANNOUNCEMENTS


@app.route("/post_announcement", methods=["POST"])
def post_announcement():
    data = request.json
    course_code, message = data.get("course_code"), data.get("message")
    if not course_code or not message: return jsonify({"status":"error","message":"Missing data"}), 400
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor() as c:
            c.execute("INSERT INTO announcements (course_code,message) VALUES(%s,%s)", (course_code, message))
            db.commit()
            return jsonify({"status":"success","message":"Announcement posted!"})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500
    finally:
        db.close()


@app.route("/get_announcements/<course_code>", methods=["GET"])
def get_announcements(course_code):
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("SELECT id,message,created_at FROM announcements WHERE course_code=%s ORDER BY created_at DESC", (course_code,))
            result = c.fetchall()
            for row in result:
                row["timestamp"] = row["created_at"].strftime("%Y-%m-%d %H:%M")
                del row["created_at"]
            return jsonify({"status":"success","announcements":result or []})
    finally:
        db.close()


@app.route("/get_notifications", methods=["GET"])
def get_notifications():
    uid = request.args.get("userId")
    db  = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor(pymysql.cursors.DictCursor) as c:
            c.execute("""SELECT id,course_code,message,created_at FROM announcements
                WHERE course_code IN (SELECT course_code FROM student_enrollments WHERE student_id=%s)
                ORDER BY created_at DESC""", (uid,))
            notifs = c.fetchall()
            for n in notifs: n["created_at"] = n["created_at"].strftime("%Y-%m-%d %H:%M")
            return jsonify({"status":"success","notifications":notifs})
    finally:
        db.close()


@app.route("/delete_notification/<int:notif_id>", methods=["DELETE", "OPTIONS"])
@cross_origin()
def delete_notification(notif_id):
    if request.method == "OPTIONS": return jsonify({"status":"success"}), 200
    db = pymysql.connect(**DB_CONFIG)
    try:
        with db.cursor() as c:
            c.execute("DELETE FROM announcements WHERE id=%s", (notif_id,))
            db.commit()
            return jsonify({"status":"success","message":"Notification deleted"}), 200
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500
    finally:
        db.close()


@app.errorhandler(404)
def not_found(e):
    return jsonify(error=str(e)), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)