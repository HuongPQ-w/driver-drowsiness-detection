"""
DEMO v2 - He thong giam sat ngu gat (ban nang cap)
====================================================
File nay la BAN DEMO de thu nghiem cach khac phuc cac "Nhuoc diem" da neu trong
tai lieu danh gia (Danh_gia_he_thong_giam_sat_ngu_guc.docx), khong phai ban dong
goi .exe hoan chinh. Muc tieu: giu code don gian, de doc, de tiep tuc phat trien
hoac lam lai, va KHONG vuot qua ngan sach tai nguyen ~2-4GB GPU RAM neu sau nay
gan them mot model AI nang hon (vi du: mo hinh phan loai mat/khuon mat dua tren
ONNX Runtime GPU, YOLO-nano, v.v).

Anh xa tung nhuoc diem -> giai phap trong file nay:
------------------------------------------------------------------------------
1) "Do tre trong viec phat hien buon ngu" (chi chay CPU)
   -> Tach rieng phan nhan dien landmark thanh 1 "backend" (class LandmarkBackend).
      Ban demo van dung MediaPipe tren CPU (nhe, khong can GPU), nhung kien truc
      cho phep sau nay cam mot backend GPU (vi du ONNX Runtime + CUDAExecutionProvider)
      ma khong phai viet lai toan bo vong lap xu ly. Ham check_gpu_budget() kiem tra
      va gioi han bo nho GPU du dinh dung (mac dinh <= 4GB) truoc khi cho phep bat
      backend GPU, tranh vuot ngan sach phan cung.

2) "Hieu qua phong ngua phu thuoc y thuc nguoi dung" (tai xe tu tat bang ESC)
   -> Khong the ngan tuyet doi viec tat trong 1 ban demo desktop, nhung MOI lan
      bat/tat giam sat deu duoc ghi log (thoi diem, ly do "user_esc" hay "user_stop")
      de sau nay lam bang chung / thong ke muc do lam dung.

3) "Khong co co che phuc hoi tu dong khi app bi treo"
   -> Them Watchdog (chay tren 1 thread rieng) theo doi "nhip tim" (heartbeat) cua
      vong lap xu ly frame. Neu qua WATCHDOG_TIMEOUT_SEC giay khong co frame moi
      (treo do camera rot, exception, video driver loi...), watchdog tu dong:
        - ghi log FREEZE_DETECTED
        - giai phong VideoCapture + FaceMesh cu
        - khoi tao lai tu dau (auto-recovery), KHONG can vao Task Manager tat thu cong

4) "Kha nang tuong thich moi truong" (Python/MediaPipe phien ban cu the)
   -> Moi loi goi API cua MediaPipe duoc boc trong ham rieng (create_face_mesh(),
      extract_landmark_xy()) de neu API doi ten/tham so o phien ban moi, chi can
      sua 1 cho. Kem theo requirements.txt ghim phien ban ro rang de tranh loi
      "chay duoc may nay, khong chay duoc may kia".

5) "Khong luu log/lich su canh bao"
   -> class EventLogger ghi CSV: thoi diem (ISO), loai su kien, chi tiet, session_id.
      Cac su kien duoc ghi: SESSION_START, SESSION_STOP, DROWSY_ALERT_START,
      DROWSY_ALERT_END (kem thoi luong), YAWN_DETECTED, HEAD_DOWN_DETECTED,
      FREEZE_DETECTED, AUTO_RECOVERY, CAMERA_LOST, EXPRESSION_CHANGED (bieu
      hien khuon mat moi - xem muc "BIEU HIEN KHUON MAT" ben duoi). Co the mo
      file CSV nay bang Excel de thong ke sau chuyen di.

6) "Kich thuoc cai dat lon, khoi dong lan dau cham" (do --collect-all mediapipe)
   -> Khong xu ly duoc trong pham vi 1 file .py (day la van de cua buoc DONG GOI
      .exe), nhung co ghi chu cu the o cuoi file (xem ham `if __name__ == "__main__"`)
      ve cach dong goi lai cho gon hon, khong dung --collect-all toan bo.

Ngoai ra, de giam phu thuoc hoan toan vao "mat nham/mo" (vi du: cuoi hip mat bi
nham lan thanh nham mat, nhin xuong bi nham lan, buon ngu ma mat van mo...), ban
demo them 2 tin hieu phu:
   - PERCLOS: % thoi gian mat nham trong 1 cua so truot (vi du 30s gan nhat) thay
     vi chi dem "nham lien tuc X giay". On dinh hon voi rung/mo hinh nhe (motion
     blur) va bot bao dong gia khi chi chop mat/cuoi.
   - MAR (mouth aspect ratio): phat hien ngap -> bat duoc dau hieu met moi khi
     mat chua the hien ro.
   - Goc cui dau (head pitch, uoc luong bang solvePnP): phat hien "cui dau ngu
     gat" ke ca khi camera khong con thay ro vung mat.

Nhung truong hop KHONG the giai quyet trong ban demo nay (can them phan cung/
du lieu huan luyen), da ghi ro trong README_v2.md di kem:
   - Deo kinh ram (can camera hong ngoai IR hoac model rieng huan luyen tren kinh ram)
   - Lai xe ban dem/thieu sang, anh sang nguoc/choi (can camera IR + bu sang)

------------------------------------------------------------------------------
BUOC 1 (v3) - HEAD NODDING nhu 1 TIN HIEU THOI GIAN (temporal feature)
------------------------------------------------------------------------------
Ban v2 chi co 1 luat tinh cho dau: "pitch >= nguong LIEN TUC qua X giay" (xem
_evaluate_head_down). Do la tin hieu cho "cui dau va GIU YEN" (vi du ngu guc/
nhin dien thoai), nhung KHONG bat duoc kieu "GAT DAU" dac trung cua buon ngu:
dau giat xuong nhanh roi giat nguoc len (do phan xa giat minh khi sap ngu),
lap lai nhieu lan trong 1 khoang thoi gian ngan.

Buoc 1 them 1 tang tin hieu MOI, doc lap voi _evaluate_head_down:
   - Toc do goc dau (pitch velocity, do/giay): dao ham roi rac cua pitch theo
     thoi gian, dung de phan biet "giat dau nhanh" (nod that) voi "cui dau tu
     tu, chu dong" (vi du cui xem dong ho).
   - 1 state machine nho (IDLE -> DROPPING -> ket thuc) de dem 1 "chu ky
     gat dau" hoan chinh: dau roi xuong qua nguong RỒI ngoc lai gan vi tri
     binh thuong trong 1 khoang thoi gian ngan (khac voi cui dau roi GIU YEN
     lien tuc, truong hop do van do rieng boi _evaluate_head_down).
   - Tan so gat dau (nod/phut) trong 1 cua so truot (mac dinh 60s), tuong tu
     cach tinh blink rate o Buoc truoc - day la tin hieu THOI GIAN dau tien
     trong lo trinh "Temporal + Multi-cue + Personalized Baseline".

Cac buoc tiep theo (chua lam trong file nay): Blink/Eye-closure temporal
features chi tiet hon (Buoc 2), Yawn frequency (Buoc 3), Personal baseline
cho tung tin hieu (Buoc 4), va tong hop thanh 1 Drowsiness Score lien tuc
(Buoc 5) thay vi cac canh bao roi rac nhu hien tai.

------------------------------------------------------------------------------
BIEU HIEN KHUON MAT (chan may + khoe mieng) - mo rong ngoai "cuoi"
------------------------------------------------------------------------------
Ban truoc chi co 1 tin hieu bieu hien: do rong mieng tang (cuoi/nhech mieng),
dung de LOAI TRU bao dong gia khi cuoi lam EAR giam gia tao. Ban nay them 2
tin hieu hinh hoc MOI, van tu chinh cac landmark da co san (khong can them
thu vien/model gi):
   - Chan may: khoang cach (long may -> mi mat tren), chuan hoa theo khoang
     cach 2 khoe mat, so voi baseline ca nhan -> phat hien nhuong may (ngac
     nhien/chu y cao) hoac cau may (kho chiu/tap trung cao).
   - Khoe mieng: do lech chieu doc cua khoe mieng so voi tam mieng, so voi
     baseline ca nhan -> phat hien mieng dang chui xuong (ve buon/kho chiu/
     met moi) thay vi chi phat hien luc nhech len (cuoi) nhu truoc.
Ca 3 tin hieu (cuoi, chan may, khoe mieng) duoc ket hop thanh 1 nhan bieu
hien don gian (xem DrowsinessDemo._update_expression), hien tren HUD va ghi
log moi khi thay doi (EXPRESSION_CHANGED). Day la luat hinh hoc DON GIAN de
tham khao them cho giam sat, KHONG phai 1 bo phan loai cam xuc (FER) chinh
xac - neu can chinh xac hon sau nay, day la cho de thay bang 1 model FER nho
(vi du ONNX mini-Xception) ma khong phai sua lai vong lap _process_frame.
"""

import argparse
import csv
import datetime
import os
import sys
import threading
import time
import uuid
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

# =============================================================================
# CAU HINH (gom het vao 1 cho de de chinh, tranh "magic number" rai rac)
# =============================================================================
EAR_THRESHOLD = 0.22                 # nguong mat nham (giong ban goc, de so sanh)
EYES_CLOSED_SECONDS = 1.5            # canh bao ngay neu nham LIEN TUC qua X giay

BLINK_IGNORE_SEC = 0.35              # nham mat NGAN HON khoang nay duoc coi la
                                      # CHOP MAT BINH THUONG, khong tinh vao PERCLOS.
                                      # Chop mat that su thuong keo dai ~0.1-0.4s.
                                      # Tang so nay len (vi du 0.5) neu van con bi
                                      # bao dong gia do chop mat; giam xuong (vi du
                                      # 0.2) neu muon he thong nhay hon.

SMILE_WIDTH_RATIO_INCREASE = 1.18    # mieng rong hon 18% so voi luc binh thuong
                                      # cua nguoi do -> coi la dang cuoi/nhech mieng
SMILE_BASELINE_EMA_ALPHA = 0.02      # toc do cap nhat muc "mieng binh thuong"
                                      # (cham, de khong bi anh huong boi 1-2 frame le)

BLINK_RATE_WINDOW_SEC = 60.0         # cua so tinh so lan chop mat / phut
MAX_NORMAL_BLINK_DURATION_SEC = 0.5  # nham mat LAU HON khoang nay khong tinh la
                                      # "chop mat" nua (coi nhu nham keo dai/buon ngu),
                                      # nen KHONG dua vao bo dem blink rate

PERCLOS_WINDOW_SEC = 30.0            # cua so truot de tinh PERCLOS
PERCLOS_THRESHOLD = 0.35             # >35% thoi gian nham mat trong 30s gan nhat -> bao dong

MAR_THRESHOLD = 0.60                 # nguong ngap (mouth aspect ratio)
MAR_MIN_DURATION_SEC = 0.8           # phai giu ngap toi thieu tung nay giay moi tinh la ngap that

HEAD_DOWN_PITCH_DEG = 25.0           # goc cui dau (so voi tu the nhin thang) de coi la "cui gat"
HEAD_DOWN_MIN_DURATION_SEC = 1.2

# --- Buoc 1 (v3): Head Nodding - tin hieu THOI GIAN, tach rieng voi cui-dau-tinh o tren ---
NOD_BASELINE_EMA_ALPHA = 0.03        # toc do cap nhat "goc dau binh thuong" cua tung nguoi
                                      # (cham, giong baseline nu cuoi, tranh bi lech boi 1-2 frame)
NOD_DROP_DEG = 12.0                  # pitch tang qua bao nhieu do SO VOI baseline thi coi la
                                      # "bat dau cui/giat dau xuong" (nguong nho hon HEAD_DOWN_PITCH_DEG
                                      # vi day la tin hieu som, chua can cui sau nhu canh bao chinh)
NOD_MIN_VELOCITY_DEG_PER_SEC = 15.0  # toc do goc TOI THIEU luc dau dang "roi" de tinh la giat/gat
                                      # that su (loc bo cui dau tu tu, chu dong, vi du xem dong ho)
PITCH_VELOCITY_WINDOW_SEC = 0.3      # cua so ngan de tinh dao ham (toc do) cua pitch theo thoi gian
NOD_EVENT_MIN_DURATION_SEC = 0.15    # 1 chu ky gat qua nhanh (nhieu do rung/landmark jitter) thi bo qua
NOD_EVENT_MAX_DURATION_SEC = 2.5     # neu cui xuong qua lau ma khong ngoc len lai trong khoang nay,
                                      # KHONG tinh la 1 lan "gat dau" nua (chuyen thanh cui-dau-giu-yen,
                                      # da co _evaluate_head_down lo rieng)
NOD_FREQ_WINDOW_SEC = 60.0           # cua so truot de tinh tan so gat dau (nod/phut)
NOD_FREQ_WARNING_COUNT = 3           # >= 3 lan gat dau trong NOD_FREQ_WINDOW_SEC gan nhat -> canh bao

# --- Buoc moi: BIEU HIEN KHUON MAT (chan may + khoe mieng) -----------------
# Muc tieu: mo rong ngoai "cuoi" (da co san qua do rong mieng) sang 2 tin hieu
# hinh hoc nua tu CHINH cac landmark dang co san (khong can model/them thu vien
# gi), giu dung tinh than "nhe, CPU-only" cua toan bo file:
#   - Khoang cach long may - mat tren (chuan hoa theo interocular): long may
#     CANG XA mat = dang nhuong/nhau may (ngac nhien, chu y cao); long may
#     CANG GAN/SUP xuong mat = dang cau may (kho chiu, tap trung cao, hoac
#     day cung la dau hieu thuong gap luc buon ngu/mat moi).
#   - Do lech chieu doc cua khoe mieng so voi tam mieng (chuan hoa tuong tu):
#     khoe mieng CHUI XUONG duoi tam mieng = ve mat buon/kho chiu/met moi.
# Ca 2 deu dung baseline ca nhan (EMA, cham) giong het logic baseline "mieng
# binh thuong" da co o _update_smile_state, vi moi nguoi co hinh dang chan
# may/mieng luc nghi ngoi khac nhau -> khong the dung 1 nguong tuyet doi chung.
EYEBROW_BASELINE_EMA_ALPHA = 0.02       # toc do cap nhat "do cao long may binh thuong"
EYEBROW_RAISE_RATIO_INCREASE = 1.15     # cao hon >=15% so binh thuong -> nhuong/nhau may
EYEBROW_FURROW_RATIO_DECREASE = 0.85    # thap hon <=85% so binh thuong -> cau may

MOUTH_CORNER_BASELINE_EMA_ALPHA = 0.02  # toc do cap nhat "vi tri khoe mieng binh thuong"
MOUTH_CORNER_DROOP_DELTA = 0.015        # lech (da chuan hoa) so voi baseline de coi la "chui xuong"

EXPRESSION_MIN_DURATION_SEC = 0.5       # bieu hien phai giu on dinh tung nay giay moi
                                         # duoc chot/ghi log, tranh nhay lien tuc do jitter

WATCHDOG_TIMEOUT_SEC = 3.0           # qua 3s khong co frame moi -> coi la treo, tu phuc hoi
WATCHDOG_CHECK_INTERVAL_SEC = 0.5
MAX_AUTO_RECOVERY_RETRIES = 5        # tranh vong lap phuc hoi vo tan neu camera hong that

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "drowsiness_events.csv")

# Chi so landmark cua MediaPipe FaceMesh (refine_landmarks=True)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_TOP_INNER = 13
MOUTH_BOTTOM_INNER = 14
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291

# Chi so long may va mi tren, dung cho tin hieu "bieu hien khuon mat" moi
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]
LEFT_EYE_UPPER_LID = [385, 387]
RIGHT_EYE_UPPER_LID = [160, 158]

# Chi so dung cho uoc luong huong dau (6 diem chuan thuong dung voi solvePnP)
HEAD_POSE_LANDMARK_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "mouth_left": 61,
    "mouth_right": 291,
}
# Mo hinh khuon mat 3D chung (don vi tuong doi, khong phai kich thuoc that) -
# chi dung de uoc luong GOC tuong doi, khong dung de do khoang cach chinh xac.
HEAD_POSE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, -330.0, -65.0),     # chin
    (-225.0, 170.0, -135.0),  # left eye corner
    (225.0, 170.0, -135.0),   # right eye corner
    (-150.0, -150.0, -125.0), # mouth left corner
    (150.0, -150.0, -125.0),  # mouth right corner
], dtype=np.float64)


# =============================================================================
# TIEN ICH TINH TOAN
# =============================================================================
def euclidean_dist(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def _landmark_xy(landmarks, idx, img_w, img_h):
    """Boc rieng cach doc 1 landmark thanh toa do pixel. Neu phien ban MediaPipe
    sau nay doi cau truc du lieu tra ve, chi can sua trong ham nay."""
    lm = landmarks[idx]
    return (lm.x * img_w, lm.y * img_h)


def calculate_ear(landmarks, eye_indices, img_w, img_h):
    points = [_landmark_xy(landmarks, i, img_w, img_h) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = points
    vertical_1 = euclidean_dist(p2, p6)
    vertical_2 = euclidean_dist(p3, p5)
    horizontal = euclidean_dist(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_mar(landmarks, img_w, img_h):
    """Mouth Aspect Ratio - dung de phat hien ngap. MAR cao = mieng mo to."""
    top = _landmark_xy(landmarks, MOUTH_TOP_INNER, img_w, img_h)
    bottom = _landmark_xy(landmarks, MOUTH_BOTTOM_INNER, img_w, img_h)
    left = _landmark_xy(landmarks, MOUTH_LEFT_CORNER, img_w, img_h)
    right = _landmark_xy(landmarks, MOUTH_RIGHT_CORNER, img_w, img_h)
    horizontal = euclidean_dist(left, right)
    if horizontal == 0:
        return 0.0
    return euclidean_dist(top, bottom) / horizontal


def calculate_mouth_width_ratio(landmarks, img_w, img_h):
    """Ti le do rong mieng so voi khoang cach 2 khoe mat ngoai. Dung khoang cach
    2 khoe mat ngoai lam thuoc do vi no khong doi nhieu theo bieu cam khuon mat
    (khac voi mieng), giup ti le nay on dinh du camera gan/xa. Khi cuoi, mieng
    gian rong ra ro ret nen ti le nay se tang cao hon han luc binh thuong."""
    mouth_left = _landmark_xy(landmarks, MOUTH_LEFT_CORNER, img_w, img_h)
    mouth_right = _landmark_xy(landmarks, MOUTH_RIGHT_CORNER, img_w, img_h)
    eye_left = _landmark_xy(landmarks, HEAD_POSE_LANDMARK_IDS["left_eye_corner"], img_w, img_h)
    eye_right = _landmark_xy(landmarks, HEAD_POSE_LANDMARK_IDS["right_eye_corner"], img_w, img_h)
    interocular = euclidean_dist(eye_left, eye_right)
    if interocular == 0:
        return 0.0
    return euclidean_dist(mouth_left, mouth_right) / interocular


def _avg_point(landmarks, indices, img_w, img_h):
    pts = [_landmark_xy(landmarks, i, img_w, img_h) for i in indices]
    xs, ys = zip(*pts)
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def calculate_eyebrow_raise_ratio(landmarks, img_w, img_h):
    """Ti le khoang cach (long may -> mi tren), chuan hoa theo khoang cach 2
    khoe mat ngoai (interocular) de khong bi anh huong boi camera gan/xa.
    Ti le CANG LON = long may cang xa mat (dang nhuong/nhau may len tren);
    ti le CANG NHO = long may cang sup gan mat (dang cau may xuong)."""
    left_brow = _avg_point(landmarks, LEFT_EYEBROW, img_w, img_h)
    right_brow = _avg_point(landmarks, RIGHT_EYEBROW, img_w, img_h)
    left_lid = _avg_point(landmarks, LEFT_EYE_UPPER_LID, img_w, img_h)
    right_lid = _avg_point(landmarks, RIGHT_EYE_UPPER_LID, img_w, img_h)
    eye_left = _landmark_xy(landmarks, HEAD_POSE_LANDMARK_IDS["left_eye_corner"], img_w, img_h)
    eye_right = _landmark_xy(landmarks, HEAD_POSE_LANDMARK_IDS["right_eye_corner"], img_w, img_h)
    interocular = euclidean_dist(eye_left, eye_right)
    if interocular == 0:
        return 0.0
    # y cang nho = cang len tren anh, nen (mi_tren.y - long_may.y) duong khi
    # long may nam TREN mi mat (truong hop binh thuong).
    left_gap = left_lid[1] - left_brow[1]
    right_gap = right_lid[1] - right_brow[1]
    return ((left_gap + right_gap) / 2.0) / interocular


def calculate_mouth_corner_offset(landmarks, img_w, img_h):
    """Do lech chieu doc cua khoe mieng so voi tam mieng (trung diem mep tren/
    mep duoi), chuan hoa theo interocular. Am (khoe mieng CAO hon tam mieng,
    tuc gia tri y nho hon) = mieng dang nhech len (cuoi); duong (khoe mieng
    THAP hon tam mieng) = mieng dang chui xuong (ve buon/kho chiu/met moi)."""
    left_corner = _landmark_xy(landmarks, MOUTH_LEFT_CORNER, img_w, img_h)
    right_corner = _landmark_xy(landmarks, MOUTH_RIGHT_CORNER, img_w, img_h)
    top = _landmark_xy(landmarks, MOUTH_TOP_INNER, img_w, img_h)
    bottom = _landmark_xy(landmarks, MOUTH_BOTTOM_INNER, img_w, img_h)
    eye_left = _landmark_xy(landmarks, HEAD_POSE_LANDMARK_IDS["left_eye_corner"], img_w, img_h)
    eye_right = _landmark_xy(landmarks, HEAD_POSE_LANDMARK_IDS["right_eye_corner"], img_w, img_h)
    interocular = euclidean_dist(eye_left, eye_right)
    if interocular == 0:
        return 0.0
    mouth_center_y = (top[1] + bottom[1]) / 2.0
    corner_y = (left_corner[1] + right_corner[1]) / 2.0
    return (corner_y - mouth_center_y) / interocular


def estimate_head_pitch_deg(landmarks, img_w, img_h):
    """Uoc luong goc cui/ngua dau (pitch) bang solvePnP voi 6 diem chuan.
    Day la uoc luong TUONG DOI cho muc dich canh bao "cui dau ngu gat", KHONG
    phai phep do goc chinh xac tuyet doi (can hieu chuan camera that de chinh xac
    hon). Tra ve None neu khong tinh duoc (vi du solvePnP that bai)."""
    image_points = np.array([
        _landmark_xy(landmarks, HEAD_POSE_LANDMARK_IDS["nose_tip"], img_w, img_h),
        _landmark_xy(landmarks, HEAD_POSE_LANDMARK_IDS["chin"], img_w, img_h),
        _landmark_xy(landmarks, HEAD_POSE_LANDMARK_IDS["left_eye_corner"], img_w, img_h),
        _landmark_xy(landmarks, HEAD_POSE_LANDMARK_IDS["right_eye_corner"], img_w, img_h),
        _landmark_xy(landmarks, HEAD_POSE_LANDMARK_IDS["mouth_left"], img_w, img_h),
        _landmark_xy(landmarks, HEAD_POSE_LANDMARK_IDS["mouth_right"], img_w, img_h),
    ], dtype=np.float64)

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))  # gia dinh khong co bien dang ong kinh

    success, rotation_vec, _translation_vec = cv2.solvePnP(
        HEAD_POSE_3D_MODEL, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    proj_matrix = np.hstack((rotation_mat, np.zeros((3, 1))))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch = float(euler_angles[0][0])
    # Chuan hoa ve khoang de doc: gia tri duong lon dan the hien dau cui xuong nhieu hon
    if pitch < -90:
        pitch += 180
    elif pitch > 90:
        pitch -= 180
    return abs(pitch)


# =============================================================================
# GHI LOG SU KIEN (khac phuc nhuoc diem #5: khong luu log/lich su)
# =============================================================================
class EventLogger:
    HEADER = ["timestamp_iso", "session_id", "event_type", "detail"]

    def __init__(self, log_file=LOG_FILE, session_id=None):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log_file = log_file
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self._lock = threading.Lock()
        is_new_file = not os.path.exists(log_file)
        if is_new_file:
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.HEADER)

    def log(self, event_type, detail=""):
        row = [datetime.datetime.now().isoformat(timespec="seconds"),
               self.session_id, event_type, detail]
        with self._lock:
            with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        print(f"[LOG] {row[0]} | {event_type} | {detail}")


# =============================================================================
# BACKEND NHAN DIEN LANDMARK - tach rieng de sau nay de cam GPU backend
# (khac phuc nhuoc diem #1 va tao duong mo rong trong pham vi 2-4GB GPU RAM)
# =============================================================================
def check_gpu_budget(max_gb=4.0):
    """Kiem tra so bo GPU co san va bao nhieu VRAM trong (neu co). Dung de
    QUYET DINH co nen bat backend GPU hay khong, tranh vuot ngan sach phan cung
    da de ra (2-4GB). Neu khong co GPU hoac khong doc duoc thong tin, tra ve
    (False, ly do) va he thong se tu dong dung CPU (an toan, khong crash)."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode != 0:
            return False, "Khong tim thay nvidia-smi (co the may khong co GPU NVIDIA)"
        free_mb = int(result.stdout.strip().split("\n")[0])
        free_gb = free_mb / 1024.0
        if free_gb < 1.0:
            return False, f"GPU chi con {free_gb:.1f}GB trong, qua thap de dung an toan"
        budget_gb = min(free_gb, max_gb)
        return True, f"GPU co {free_gb:.1f}GB trong, se gioi han su dung toi da {budget_gb:.1f}GB"
    except Exception as exc:
        return False, f"Khong kiem tra duoc GPU ({exc}), mac dinh dung CPU"


class LandmarkBackend:
    """Interface chung. Ban demo hien chi co CPUBackend (MediaPipe). Sau nay neu
    muon tang toc do (giam do tre - nhuoc diem #1), co the them GpuBackend ke
    thua class nay, dung ONNX Runtime voi CUDAExecutionProvider va gioi han
    'gpu_mem_limit' (vi du 3 * 1024**3 byte = 3GB) trong provider_options, ma
    KHONG can sua lai vong lap chinh trong DrowsinessDemo."""

    def process(self, rgb_frame):
        raise NotImplementedError

    def close(self):
        pass


TASKS_MODEL_PATH = os.path.join(LOG_DIR, "..", "models", "face_landmarker.task")
TASKS_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


class MediaPipeCpuBackend(LandmarkBackend):
    """Backend mac dinh: MediaPipe chay CPU, khong can GPU. Day la vi du CU THE
    cho nhuoc diem #4 (phu thuoc phien ban): tu ban 0.10.x tro len, Google da
    NGUNG cung cap API cu 'mp.solutions.face_mesh' o mot so ban dong goi (chuyen
    sang API moi 'mediapipe.tasks'). Neu chi viet cung 1 cach goi nhu ban goc,
    code se BAO LOI NGAY LAP TUC tren may co ban mediapipe moi hon (dung y het
    nhuoc diem da neu trong tai lieu).

    Backend nay tu kiem tra: neu may co API cu ('solutions') thi dung API cu
    (nhe, khong can tai model rieng); neu khong co (may moi) thi tu dong chuyen
    sang API moi ('tasks'), tai model 1 lan duy nhat va dung lai cho cac lan sau.
    Ca 2 duong deu tra ve cung 1 dinh dang landmark (list co .x/.y) nen phan
    con lai cua chuong trinh (tinh EAR/MAR/head-pose) KHONG can sua gi them."""

    def __init__(self):
        self._mode = "legacy" if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh") else "tasks"
        self._instance = self._create_instance()

    def _create_instance(self):
        # Boc rieng loi goi khoi tao de neu API doi tham so o ban MediaPipe moi,
        # chi can sua trong ham nay (khac phuc nhuoc diem #4 - phu thuoc phien ban).
        if self._mode == "legacy":
            return mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        return self._create_tasks_instance()

    def _create_tasks_instance(self):
        model_path = self._ensure_tasks_model()
        from mediapipe.tasks import python as mp_tasks_python
        from mediapipe.tasks.python import vision as mp_tasks_vision

        base_options = mp_tasks_python.BaseOptions(model_asset_path=model_path)
        options = mp_tasks_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_tasks_vision.RunningMode.VIDEO,
            num_faces=1,
        )
        return mp_tasks_vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _ensure_tasks_model():
        model_path = os.path.abspath(TASKS_MODEL_PATH)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if os.path.exists(model_path):
            return model_path
        try:
            import urllib.request
            print("[SETUP] Dang tai model face_landmarker.task (chi 1 lan)...")
            urllib.request.urlretrieve(TASKS_MODEL_URL, model_path)
            return model_path
        except Exception as exc:
            raise RuntimeError(
                "Khong the tu dong tai model 'face_landmarker.task' (co the do mang "
                "bi chan hoac khong co internet). Cach khac phuc:\n"
                f"  1) Tai thu cong tu: {TASKS_MODEL_URL}\n"
                f"     va dat vao: {model_path}\n"
                "  2) Hoac cai dat ban MediaPipe co san API cu, vi du:\n"
                "     pip install \"mediapipe==0.10.9\"\n"
                f"Chi tiet loi goc: {exc}"
            ) from exc

    def process(self, rgb_frame):
        if self._mode == "legacy":
            rgb_frame.flags.writeable = False
            results = self._instance.process(rgb_frame)
            rgb_frame.flags.writeable = True
            if results.multi_face_landmarks:
                return results.multi_face_landmarks[0].landmark
            return None

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        result = self._instance.detect_for_video(mp_image, timestamp_ms)
        if result.face_landmarks:
            return result.face_landmarks[0]
        return None

    def restart(self):
        """Dung khi watchdog phat hien treo: dong instance cu, tao lai instance moi."""
        self.close()
        self._instance = self._create_instance()

    def close(self):
        try:
            self._instance.close()
        except Exception:
            pass


# =============================================================================
# WATCHDOG - khac phuc nhuoc diem #3: khong co co che phuc hoi tu dong khi treo
# =============================================================================
class Watchdog(threading.Thread):
    def __init__(self, timeout_sec, on_timeout, check_interval=WATCHDOG_CHECK_INTERVAL_SEC):
        super().__init__(daemon=True)
        self.timeout_sec = timeout_sec
        self.on_timeout = on_timeout
        self.check_interval = check_interval
        self._last_heartbeat = time.time()
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()

    def beat(self):
        with self._lock:
            self._last_heartbeat = time.time()

    def stop(self):
        self._stop_flag.set()

    def run(self):
        while not self._stop_flag.is_set():
            time.sleep(self.check_interval)
            with self._lock:
                elapsed = time.time() - self._last_heartbeat
            if elapsed > self.timeout_sec:
                self.on_timeout(elapsed)
                self.beat()  # tranh bao lien tuc trong khi dang phuc hoi


# =============================================================================
# APP DEMO CHINH
# =============================================================================
class DrowsinessDemo:
    def __init__(self, source=0):
        self.source = source
        self.logger = EventLogger()
        self.backend = MediaPipeCpuBackend()
        self.cap = None
        self._open_capture()

        self.eyes_closed_start = None
        self.alert_active = False
        self.mar_high_start = None
        self.head_down_start = None
        self.perclos_window = deque()  # cac tuple (timestamp, is_closed)
        self.recovery_attempts = 0
        self._mouth_width_baseline = None      # baseline "mieng binh thuong" de loc cuoi
        self.blink_events = deque()            # timestamp cac lan chop mat hoan tat (60s gan nhat)
        self.total_blink_count = 0             # tong so lan chop mat tu luc bat dau (thong ke)

        # --- Buoc 1 (v3): trang thai cho Head Nodding (tin hieu thoi gian) ---
        self.pitch_baseline = None             # "goc dau binh thuong" cua nguoi dang ngoi (EMA)
        self.pitch_history = deque()           # (t, pitch) trong PITCH_VELOCITY_WINDOW_SEC gan nhat
        self.current_pitch_velocity = 0.0      # do/giay, duong = dang cui xuong, am = dang ngoc len
        self.nod_state = "IDLE"                # "IDLE" | "DROPPING" (dang trong 1 chu ky gat do)
        self.nod_drop_start_time = None
        self.nod_peak_velocity = 0.0
        self.nod_events = deque()              # timestamp cac lan "gat dau" hoan tat (60s gan nhat)
        self.total_nod_count = 0
        self._nod_freq_warning_active = False  # debounce, tranh log spam moi frame

        # --- Bieu hien khuon mat (chan may + khoe mieng) ---
        self._eyebrow_baseline = None          # "do cao long may binh thuong" cua nguoi dang ngoi (EMA)
        self._mouth_corner_baseline = None     # "vi tri khoe mieng binh thuong" (EMA)
        self.current_expression = "Binh thuong"
        self._pending_expression = None
        self._pending_expression_since = 0.0

        self.watchdog = Watchdog(WATCHDOG_TIMEOUT_SEC, self._on_freeze_detected)
        self.watchdog.start()

        gpu_ok, gpu_msg = check_gpu_budget(max_gb=4.0)
        self.logger.log("GPU_BUDGET_CHECK", f"gpu_available={gpu_ok} | {gpu_msg}")
        self.logger.log("SESSION_START", f"source={source}")

    def _open_capture(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Khong mo duoc nguon video: {self.source}")

    def _on_freeze_detected(self, elapsed_sec):
        """Callback cua watchdog khi phat hien treo (khong co frame moi qua lau)."""
        self.logger.log("FREEZE_DETECTED", f"khong co frame moi trong {elapsed_sec:.1f}s")
        self.recovery_attempts += 1
        if self.recovery_attempts > MAX_AUTO_RECOVERY_RETRIES:
            self.logger.log(
                "AUTO_RECOVERY_GAVE_UP",
                f"da thu {self.recovery_attempts} lan, dung lai de tranh vong lap vo tan",
            )
            return
        try:
            self.backend.restart()
            self._open_capture()
            self.logger.log("AUTO_RECOVERY", f"lan thu {self.recovery_attempts} - thanh cong")
        except Exception as exc:
            self.logger.log("AUTO_RECOVERY_FAILED", f"lan thu {self.recovery_attempts} - loi: {exc}")

    def _update_perclos(self, now, is_closed):
        self.perclos_window.append((now, is_closed))
        cutoff = now - PERCLOS_WINDOW_SEC
        while self.perclos_window and self.perclos_window[0][0] < cutoff:
            self.perclos_window.popleft()
        if len(self.perclos_window) < 2:
            return 0.0
        closed_count = sum(1 for _, closed in self.perclos_window if closed)
        return closed_count / len(self.perclos_window)

    def _update_smile_state(self, mouth_width_ratio, raw_is_closed):
        """Cap nhat muc 'mieng binh thuong' cua nguoi dang ngoi (chi cap nhat khi
        mat KHONG bi coi la nham, tranh lay nham luc dang nhech mieng/met moi).
        Tra ve True neu khung hinh hien tai duoc coi la dang cuoi/nhech mieng."""
        if not raw_is_closed:
            if self._mouth_width_baseline is None:
                self._mouth_width_baseline = mouth_width_ratio
            else:
                self._mouth_width_baseline = (
                    (1 - SMILE_BASELINE_EMA_ALPHA) * self._mouth_width_baseline
                    + SMILE_BASELINE_EMA_ALPHA * mouth_width_ratio
                )
        if not self._mouth_width_baseline:
            return False
        return mouth_width_ratio >= self._mouth_width_baseline * SMILE_WIDTH_RATIO_INCREASE

    # -------------------------------------------------------------------
    # BIEU HIEN KHUON MAT - ket hop chan may + khoe mieng + tin hieu cuoi
    # da co san, suy ra 1 nhan bieu hien don gian (khong dung model rieng,
    # chi dung hinh hoc landmark, giu dung tinh than nhe/CPU-only cua file).
    # -------------------------------------------------------------------
    def _update_expression(self, now, eyebrow_ratio, mouth_corner_offset, is_smiling, is_closed):
        # Chi cap nhat baseline long may luc mat KHONG nham (tranh nham lan
        # voi dong tac nham mat/chop mat lam long may hoi ha xuong theo).
        if not is_closed:
            if self._eyebrow_baseline is None:
                self._eyebrow_baseline = eyebrow_ratio
            else:
                self._eyebrow_baseline = (
                    (1 - EYEBROW_BASELINE_EMA_ALPHA) * self._eyebrow_baseline
                    + EYEBROW_BASELINE_EMA_ALPHA * eyebrow_ratio
                )

        # Chi cap nhat baseline khoe mieng luc KHONG cuoi (tuong tu logic
        # baseline do rong mieng o tren), tranh baseline bi keo lech theo
        # chinh luc dang cuoi.
        if self._mouth_corner_baseline is None:
            self._mouth_corner_baseline = mouth_corner_offset
        elif not is_smiling:
            self._mouth_corner_baseline = (
                (1 - MOUTH_CORNER_BASELINE_EMA_ALPHA) * self._mouth_corner_baseline
                + MOUTH_CORNER_BASELINE_EMA_ALPHA * mouth_corner_offset
            )

        eyebrow_raised = (
            self._eyebrow_baseline and self._eyebrow_baseline > 0
            and eyebrow_ratio >= self._eyebrow_baseline * EYEBROW_RAISE_RATIO_INCREASE
        )
        eyebrow_furrowed = (
            self._eyebrow_baseline and self._eyebrow_baseline > 0
            and eyebrow_ratio <= self._eyebrow_baseline * EYEBROW_FURROW_RATIO_DECREASE
        )
        mouth_drooped = (
            self._mouth_corner_baseline is not None
            and mouth_corner_offset - self._mouth_corner_baseline >= MOUTH_CORNER_DROOP_DELTA
        )

        # Thu tu uu tien: cuoi (tin hieu san co, dang tin cay nhat) -> ngac
        # nhien -> cau co ro ret (ca chan may lan mieng) -> chi rieng 1 trong
        # 2 tin hieu -> binh thuong. Day la luat don gian, KHONG phai phan
        # loai cam xuc chinh xac tuyet doi - chi de lam tin hieu THAM KHAO
        # them cho giam sat (vi du: cau co/nhau may keo dai cung co the la
        # dau hieu mat moi/kho chiu, tuong tu tinh than PERCLOS/MAR o tren).
        if is_smiling:
            label = "Cuoi / Vui"
        elif eyebrow_raised and not mouth_drooped:
            label = "Nhuong may / Ngac nhien"
        elif eyebrow_furrowed and mouth_drooped:
            label = "Cau co / Kho chiu"
        elif eyebrow_furrowed:
            label = "Nhau may / Tap trung cao"
        elif mouth_drooped:
            label = "Khoe mieng chui xuong / Ve buon-met"
        else:
            label = "Binh thuong"

        self._commit_expression_label(now, label)
        return self.current_expression

    def _commit_expression_label(self, now, label):
        """Chi CHOT va GHI LOG khi 1 bieu hien duoc giu on dinh lien tuc qua
        EXPRESSION_MIN_DURATION_SEC, tranh ghi log/nhay lien tuc do rung
        (jitter) landmark giua cac frame gan nhau."""
        if label != self._pending_expression:
            self._pending_expression = label
            self._pending_expression_since = now

        if now - self._pending_expression_since >= EXPRESSION_MIN_DURATION_SEC:
            if label != self.current_expression:
                self.logger.log("EXPRESSION_CHANGED", f"{self.current_expression} -> {label}")
                self.current_expression = label

    def _register_completed_blink(self, now, closure_duration):
        """Goi khi mat vua MO LAI sau 1 lan nham. Chi tinh la 'chop mat' binh
        thuong (dua vao bo dem blink rate) neu thoi gian nham <= nguong toi da;
        nham lau hon (co the la buon ngu) thi KHONG tinh vao blink rate, vi day
        la 2 hien tuong khac nhau ve y nghia (nghien cuu cho thay chop mat va
        nham mat keo dai can duoc phan biet ro)."""
        self.total_blink_count += 1
        if closure_duration <= MAX_NORMAL_BLINK_DURATION_SEC:
            self.blink_events.append(now)
        cutoff = now - BLINK_RATE_WINDOW_SEC
        while self.blink_events and self.blink_events[0] < cutoff:
            self.blink_events.popleft()

    def get_blink_rate_per_min(self):
        """So lan chop mat trong BLINK_RATE_WINDOW_SEC gan nhat, quy doi ve don
        vi 'lan/phut'. Luu y: theo nghien cuu, chi so nay KHONG tang don dieu
        theo muc do buon ngu (co the tang roi giam), nen chi dung de THAM KHAO/
        ghi log, khong dung rieng le de bat canh bao."""
        if BLINK_RATE_WINDOW_SEC <= 0:
            return 0.0
        return len(self.blink_events) * (60.0 / BLINK_RATE_WINDOW_SEC)

    def run(self, show_window=True):
        try:
            while True:
                ok, frame = self.cap.read()
                self.watchdog.beat()  # bao cho watchdog biet vong lap van song
                if not ok:
                    self.logger.log("CAMERA_LOST", "cap.read() tra ve False")
                    time.sleep(0.2)
                    continue

                try:
                    self._process_frame(frame, show_window)
                except Exception as exc:
                    # Bat loi ngay trong vong lap de 1 frame loi khong lam sap ca ung dung
                    # (gop phan khac phuc nhuoc diem #3 - khong co phuc hoi tu dong)
                    self.logger.log("FRAME_ERROR_RECOVERED", str(exc))

                if show_window and (cv2.waitKey(1) & 0xFF == 27):  # phim ESC
                    self.logger.log("SESSION_STOP", "reason=user_esc")
                    break
        finally:
            self.shutdown()

    def _process_frame(self, frame, show_window):
        now = time.time()
        frame = cv2.flip(frame, 1)
        img_h, img_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = self.backend.process(rgb_frame)

        status_text = "DANG TIM KHUON MAT..."
        status_color = (232, 209, 79)  # cyan (BGR)

        if landmarks is not None:
            left_ear = calculate_ear(landmarks, LEFT_EYE, img_w, img_h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE, img_w, img_h)
            avg_ear = (left_ear + right_ear) / 2.0
            raw_is_closed = avg_ear < EAR_THRESHOLD

            mouth_width_ratio = calculate_mouth_width_ratio(landmarks, img_w, img_h)
            is_smiling = self._update_smile_state(mouth_width_ratio, raw_is_closed)

            # Neu dang cuoi thi KHONG tinh la nham mat, du EAR co thap do hip mat.
            is_closed = raw_is_closed and not is_smiling

            # Uoc luong truoc thoi luong da nham lien tuc (dung chung voi ham
            # _evaluate_drowsiness ben duoi) de quyet dinh co tinh vao PERCLOS
            # hay khong: chi tinh la "nham" trong PERCLOS neu da nham qua
            # BLINK_IGNORE_SEC giay, tranh chop mat binh thuong lam PERCLOS
            # tang gia tao (nguyen nhan gay cac canh bao ngan 0.0-0.3s).
            if is_closed:
                provisional_start = self.eyes_closed_start if self.eyes_closed_start else now
                closed_so_far = now - provisional_start
            else:
                # Mat vua mo lai (hoac chua tung nham): neu truoc do dang nham,
                # day la luc "hoan tat" 1 lan nham mat -> ghi nhan vao bo dem
                # blink counter / blink rate (Buoc 2).
                if self.eyes_closed_start is not None:
                    closure_duration = now - self.eyes_closed_start
                    self._register_completed_blink(now, closure_duration)
                closed_so_far = 0.0
            counts_as_closed_for_perclos = is_closed and closed_so_far >= BLINK_IGNORE_SEC

            perclos = self._update_perclos(now, counts_as_closed_for_perclos)

            mar = calculate_mar(landmarks, img_w, img_h)
            pitch = estimate_head_pitch_deg(landmarks, img_w, img_h)

            eyebrow_ratio = calculate_eyebrow_raise_ratio(landmarks, img_w, img_h)
            mouth_corner_offset = calculate_mouth_corner_offset(landmarks, img_w, img_h)
            expression_label = self._update_expression(now, eyebrow_ratio, mouth_corner_offset,
                                                         is_smiling, is_closed)

            status_text, status_color = self._evaluate_drowsiness(now, is_closed, perclos, avg_ear)
            self._evaluate_yawn(now, mar)
            self._evaluate_head_down(now, pitch)
            self._update_head_nodding(now, pitch)

            if show_window:
                self._draw_overlay(frame, landmarks, img_w, img_h, avg_ear, perclos, mar, pitch,
                                    status_text, status_color, expression_label)
        elif show_window:
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        status_color, 2, cv2.LINE_AA)

        if show_window:
            cv2.imshow("Drowsiness Demo v2 (ESC de thoat)", frame)

    def _evaluate_drowsiness(self, now, is_closed, perclos, avg_ear):
        """Ket hop 2 tin hieu: nham LIEN TUC qua X giay (nhu ban goc) VA PERCLOS
        cao trong 30s gan nhat (on dinh hon, giam bao dong gia do chop/cuoi mat)."""
        if is_closed:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = now
            closed_duration = now - self.eyes_closed_start
        else:
            self.eyes_closed_start = None
            closed_duration = 0.0

        drowsy_by_duration = closed_duration >= EYES_CLOSED_SECONDS
        drowsy_by_perclos = perclos >= PERCLOS_THRESHOLD

        if drowsy_by_duration or drowsy_by_perclos:
            if not self.alert_active:
                reason = "nham_lien_tuc" if drowsy_by_duration else "perclos_cao"
                self.logger.log("DROWSY_ALERT_START", f"ly_do={reason} | EAR={avg_ear:.3f} | PERCLOS={perclos:.0%}")
                self.alert_active = True
                self._alert_start_time = now
            return "CANH BAO NGU GAT!", (61, 61, 232)  # do (BGR)

        if self.alert_active:
            duration = now - getattr(self, "_alert_start_time", now)
            self.logger.log("DROWSY_ALERT_END", f"thoi_luong={duration:.1f}s")
            self.alert_active = False

        if is_closed:
            return f"Mat dang nham... ({closed_duration:.1f}s)", (35, 166, 245)  # cam
        return "TINH TAO", (132, 220, 61)  # xanh la

    def _evaluate_yawn(self, now, mar):
        if mar >= MAR_THRESHOLD:
            if self.mar_high_start is None:
                self.mar_high_start = now
            elif now - self.mar_high_start >= MAR_MIN_DURATION_SEC and not getattr(self, "_yawn_logged", False):
                self.logger.log("YAWN_DETECTED", f"MAR={mar:.2f}")
                self._yawn_logged = True
        else:
            self.mar_high_start = None
            self._yawn_logged = False

    def _evaluate_head_down(self, now, pitch):
        if pitch is None:
            return
        if pitch >= HEAD_DOWN_PITCH_DEG:
            if self.head_down_start is None:
                self.head_down_start = now
            elif now - self.head_down_start >= HEAD_DOWN_MIN_DURATION_SEC and not getattr(self, "_head_down_logged", False):
                self.logger.log("HEAD_DOWN_DETECTED", f"goc_cui_dau={pitch:.1f} do")
                self._head_down_logged = True
        else:
            self.head_down_start = None
            self._head_down_logged = False

    # -------------------------------------------------------------------
    # Buoc 1 (v3): HEAD NODDING - tin hieu THOI GIAN (toc do + tan so gat dau)
    # -------------------------------------------------------------------
    def _update_pitch_baseline_and_velocity(self, now, pitch):
        """Cap nhat baseline 'goc dau binh thuong' (chi khi dau dang o gan vi
        tri binh thuong, tranh baseline bi keo theo luc dang cui/gat), va tinh
        toc do goc (do/giay) bang sai phan qua 1 cua so ngan de on dinh hon
        so voi lay sai phan giua 2 frame lien tiep (de bi nhieu do jitter)."""
        self.pitch_history.append((now, pitch))
        cutoff = now - PITCH_VELOCITY_WINDOW_SEC
        while len(self.pitch_history) > 1 and self.pitch_history[0][0] < cutoff:
            self.pitch_history.popleft()

        if len(self.pitch_history) >= 2:
            t0, p0 = self.pitch_history[0]
            dt = now - t0
            self.current_pitch_velocity = (pitch - p0) / dt if dt > 0 else 0.0
        else:
            self.current_pitch_velocity = 0.0

        # Chi cap nhat baseline khi dau dang gan vi tri binh thuong (chua bat
        # dau cui), giong logic baseline nu cuoi o tren.
        if self.pitch_baseline is None:
            self.pitch_baseline = pitch
        elif pitch < self.pitch_baseline + NOD_DROP_DEG:
            self.pitch_baseline = (
                (1 - NOD_BASELINE_EMA_ALPHA) * self.pitch_baseline
                + NOD_BASELINE_EMA_ALPHA * pitch
            )

    def _update_head_nodding(self, now, pitch):
        """State machine dem 1 'chu ky gat dau' hoan chinh: dau CUI XUONG qua
        nguong (so voi baseline) VOI TOC DO DU NHANH, roi NGOC LAI gan baseline
        trong khoang thoi gian ngan. Khac voi _evaluate_head_down (cui xuong
        va GIU YEN lien tuc) - 2 tin hieu nay bo sung cho nhau, khong thay the."""
        if pitch is None:
            return  # mat dau/solvePnP that bai o frame nay: giu nguyen trang thai, bo qua

        self._update_pitch_baseline_and_velocity(now, pitch)
        drop_threshold = self.pitch_baseline + NOD_DROP_DEG

        if self.nod_state == "IDLE":
            if pitch >= drop_threshold and self.current_pitch_velocity >= NOD_MIN_VELOCITY_DEG_PER_SEC:
                self.nod_state = "DROPPING"
                self.nod_drop_start_time = now
                self.nod_peak_velocity = self.current_pitch_velocity
        else:  # self.nod_state == "DROPPING"
            self.nod_peak_velocity = max(self.nod_peak_velocity, self.current_pitch_velocity)
            elapsed = now - self.nod_drop_start_time

            if pitch < drop_threshold:
                # Dau da ngoc lai gan baseline -> hoan tat 1 chu ky
                if NOD_EVENT_MIN_DURATION_SEC <= elapsed <= NOD_EVENT_MAX_DURATION_SEC:
                    self._register_completed_nod(now, elapsed)
                # Neu qua nhanh (< MIN) chi la nhieu/jitter, bo qua khong dem
                self.nod_state = "IDLE"
                self.nod_drop_start_time = None
                self.nod_peak_velocity = 0.0
            elif elapsed > NOD_EVENT_MAX_DURATION_SEC:
                # Cui qua lau khong ngoc len -> day la cui-dau-giu-yen, KHONG
                # phai gat dau; nhuong lai cho _evaluate_head_down xu ly, thoat
                # khoi state DROPPING de khong dem nham khi cuoi cung dau ngoc len.
                self.nod_state = "IDLE"
                self.nod_drop_start_time = None
                self.nod_peak_velocity = 0.0

        self._update_nod_frequency_warning(now)

    def _register_completed_nod(self, now, duration):
        self.total_nod_count += 1
        self.nod_events.append(now)
        self.logger.log(
            "NOD_DETECTED",
            f"thoi_luong_chu_ky={duration:.2f}s | toc_do_dinh={self.nod_peak_velocity:.1f} do/s",
        )

    def _update_nod_frequency_warning(self, now):
        cutoff = now - NOD_FREQ_WINDOW_SEC
        while self.nod_events and self.nod_events[0] < cutoff:
            self.nod_events.popleft()

        is_frequent = len(self.nod_events) >= NOD_FREQ_WARNING_COUNT
        if is_frequent and not self._nod_freq_warning_active:
            self.logger.log(
                "NODDING_FREQUENT_WARNING",
                f"{len(self.nod_events)} lan gat dau trong {NOD_FREQ_WINDOW_SEC:.0f}s gan nhat",
            )
            self._nod_freq_warning_active = True
        elif not is_frequent and self._nod_freq_warning_active:
            self._nod_freq_warning_active = False

    def get_nod_frequency_per_min(self):
        """Tan so gat dau (lan/phut) trong NOD_FREQ_WINDOW_SEC gan nhat - tin
        hieu THOI GIAN dau tien cua lo trinh (se duoc dua vao Drowsiness Score
        chung o Buoc 5, cung voi blink rate/PERCLOS/yawn frequency/baseline ca nhan)."""
        if NOD_FREQ_WINDOW_SEC <= 0:
            return 0.0
        return len(self.nod_events) * (60.0 / NOD_FREQ_WINDOW_SEC)

    def _draw_overlay(self, frame, landmarks, img_w, img_h, avg_ear, perclos, mar, pitch,
                       status_text, status_color, expression_label):
        for indices in (LEFT_EYE, RIGHT_EYE):
            pts = np.array([_landmark_xy(landmarks, i, img_w, img_h) for i in indices], dtype=np.int32)
            cv2.polylines(frame, [pts], True, status_color, 1, cv2.LINE_AA)

        # --- Nen ban trong suot phia sau khoi text: chu trang tren nen sang
        # (tuong, cua so...) rat kho doc va nhin "roi/chat cho"; ve 1 khung mo
        # tranh nhu 1 lop overlay HUD giup tach het chu ra khoi background that. ---
        # panel_h tang them 24px so voi truoc de co cho cho dong "Bieu hien"
        # (bieu hien khuon mat) ma khong de lan xuong che vung mat/mieng.
        panel_h = 142
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (img_w, panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        cv2.putText(frame, status_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)

        # Chia 2 COT thay vi xep 6 dong doc mot cot: giam chieu cao chiem dung
        # (tu ~230px con ~85px), do khong con lan xuong che vao vung mat/mieng.
        col1_x, col2_x = 20, max(300, img_w // 2)
        row_y = [58, 84, 110]

        col1_lines = [
            f"EAR {avg_ear:.2f} (nguong {EAR_THRESHOLD})",
            f"PERCLOS {perclos:.0%} (nguong {PERCLOS_THRESHOLD:.0%})",
            f"MAR {mar:.2f} (nguong {MAR_THRESHOLD})",
        ]
        col2_lines = [
            f"Cui dau: {pitch:.1f}do" if pitch is not None else "Cui dau: --",
            f"Chop mat: {self.get_blink_rate_per_min():.0f}/phut (tong {self.total_blink_count})",
            f"Gat dau: {self.get_nod_frequency_per_min():.0f}/phut (tong {self.total_nod_count})",
        ]
        for y, line in zip(row_y, col1_lines):
            cv2.putText(frame, line, (col1_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
        for y, line in zip(row_y, col2_lines):
            is_nod_row = line.startswith("Gat dau")
            color = (61, 61, 232) if (is_nod_row and self._nod_freq_warning_active) else (255, 255, 255)
            cv2.putText(frame, line, (col2_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        color, 1, cv2.LINE_AA)

        # Dong rieng, choan het chieu ngang, cho "Bieu hien khuon mat" (chan
        # may + khoe mieng) - dat duoi cung cua panel de khong lan vao 2 cot tren.
        cv2.putText(frame, f"Bieu hien: {expression_label}", (20, 133),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (79, 209, 232), 1, cv2.LINE_AA)

        if self.alert_active:
            cv2.rectangle(frame, (0, 0), (img_w, img_h), (61, 61, 232), 8)

    def shutdown(self):
        self.watchdog.stop()
        if self.cap is not None:
            self.cap.release()
        self.backend.close()
        cv2.destroyAllWindows()
        self.logger.log("SESSION_STOP", "reason=shutdown")


def parse_args():
    parser = argparse.ArgumentParser(description="Demo v2 - Giam sat ngu gat (co log + watchdog + PERCLOS/MAR/head-pose)")
    parser.add_argument("--source", default="0",
                         help="0 (webcam mac dinh) hoac duong dan file video de kiem tra")
    args = parser.parse_args()
    # Cho phep truyen "0" (webcam) hoac duong dan file video
    source = int(args.source) if args.source.isdigit() else args.source
    return source


if __name__ == "__main__":
    # ------------------------------------------------------------------------
    # GHI CHU VE NHUOC DIEM #6 (kich thuoc cai dat lon, khoi dong cham):
    # Van de nay nam o BUOC DONG GOI (PyInstaller), khong nam trong file .py.
    # Ban goc dung "--collect-all mediapipe" se dong goi TOAN BO cac model/
    # submodule cua mediapipe (ke ca cac model khong dung: hand, pose, holistic...),
    # lam file .exe rat nang va lan khoi dong dau tien phai giai nen het.
    # Khi dong goi ban nang cap nay, nen:
    #   1) Chi khai bao "--hidden-import mediapipe.python._framework_bindings"
    #      va cac submodule THAT SU dung (face_mesh), thay vi --collect-all.
    #   2) Dung "--onedir" thay vi "--onefile" de lan chay sau khong phai giai
    #      nen lai tu dau (chi cham o lan chay dau tien).
    #   3) Neu van con nang, can nhac thay MediaPipe FaceMesh bang mot model
    #      landmark nhe hon (vi du ONNX, ~5-10MB) chi xuat 6-10 diem can dung
    #      (mat, mieng, mui, cam) thay vi toan bo 478 diem khuon mat.
    # ------------------------------------------------------------------------
    video_source = parse_args()
    app = DrowsinessDemo(source=video_source)
    app.run(show_window=True)