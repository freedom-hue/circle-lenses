import sys
import os
import time
import json
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QPushButton, QVBoxLayout, QHBoxLayout, QSlider, 
                             QComboBox, QGroupBox, QFileDialog, QTabWidget, QCheckBox, 
                             QRadioButton, QLineEdit, QMessageBox, QScrollArea,
                             QDialog, QListWidget, QListWidgetItem, QInputDialog, QSplitter)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QIcon

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

try:
    mp_face_mesh = mp.solutions.face_mesh
except AttributeError:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
        path_in_internal = os.path.join(base_dir, '_internal', relative_path)
        if os.path.exists(path_in_internal):
            return path_in_internal
        path_in_exe = os.path.join(base_dir, relative_path)
        if os.path.exists(path_in_exe):
            return path_in_exe
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

def get_exe_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def cv2_imread_unicode(file_path, flags=cv2.IMREAD_UNCHANGED):
    if not os.path.exists(file_path):
        return None
    try:
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags)
    except Exception:
        return None

def generate_soft_radial_alpha(ch, cw, radius):
    """生成余弦径向边缘渐变 Alpha 遮罩，消除一切硬边塑料感"""
    cx, cy = cw // 2, ch // 2
    Y, X = np.ogrid[:ch, :cw]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    inner_r = radius * 0.70  
    outer_r = radius * 0.98  
    
    alpha = np.zeros((ch, cw), dtype=np.float32)
    alpha[dist <= inner_r] = 255.0
    
    fade_mask = (dist > inner_r) & (dist <= outer_r)
    if np.any(fade_mask):
        norm_dist = (dist[fade_mask] - inner_r) / (outer_r - inner_r)
        alpha[fade_mask] = 255.0 * 0.5 * (1.0 + np.cos(norm_dist * np.pi))
        
    alpha_uint8 = np.clip(alpha, 0, 255).astype(np.uint8)
    ksize = max(5, int(radius * 0.15) * 2 + 1)
    return cv2.GaussianBlur(alpha_uint8, (ksize, ksize), 0)

def load_lens_texture(img_path):
    raw_img = cv2_imread_unicode(img_path, cv2.IMREAD_UNCHANGED)
    if raw_img is None:
        return None

    if len(raw_img.shape) == 3 and raw_img.shape[-1] == 4:
        b, g, r, alpha_chan = cv2.split(raw_img)
        img_bgr = cv2.merge([b, g, r])
        if np.any(alpha_chan > 20):
            contours, _ = cv2.findContours((alpha_chan > 20).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_cnt = max(contours, key=cv2.contourArea)
                (cx, cy), radius = cv2.minEnclosingCircle(max_cnt)
                cx, cy, radius = int(cx), int(cy), int(radius)
            else:
                h, w = alpha_chan.shape
                cx, cy, radius = w // 2, h // 2, min(w, h) // 2
            
            radius = int(radius * 0.98)
            x1, x2 = max(0, cx - radius), min(img_bgr.shape[1], cx + radius)
            y1, y2 = max(0, cy - radius), min(img_bgr.shape[0], cy + radius)
            cropped_bgr = img_bgr[y1:y2, x1:x2]
            
            ch, cw = cropped_bgr.shape[:2]
            if ch <= 0 or cw <= 0: return None
            
            soft_alpha = generate_soft_radial_alpha(ch, cw, min(cw, ch) // 2)
            return cv2.merge([cropped_bgr[:,:,0], cropped_bgr[:,:,1], cropped_bgr[:,:,2], soft_alpha])

    if len(raw_img.shape) == 3 and raw_img.shape[-1] == 3:
        img = raw_img
    else:
        img = cv2_imread_unicode(img_path, cv2.IMREAD_COLOR)

    if img is None:
        return None

    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_cnt = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(max_cnt)
        cx, cy, radius = int(cx), int(cy), int(radius)
    else:
        cx, cy, radius = w // 2, h // 2, min(w, h) // 2

    crop_r = int(radius * 0.95)
    if crop_r <= 5: crop_r = min(w, h) // 2

    x1, x2 = max(0, cx - crop_r), min(w, cx + crop_r)
    y1, y2 = max(0, cy - crop_r), min(h, cy + crop_r)
    cropped = img[y1:y2, x1:x2]

    ch, cw = cropped.shape[:2]
    if ch <= 0 or cw <= 0: return None

    b, g, r_chan = cv2.split(cropped)
    soft_alpha = generate_soft_radial_alpha(ch, cw, min(cw, ch) // 2)
    return cv2.merge([b, g, r_chan, soft_alpha])


class CustomDialog(QDialog):
    def __init__(self, title, message, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedWidth(420)
        self.setStyleSheet("""
            QDialog { background-color: #1E293B; border: 2px solid #38BDF8; border-radius: 12px; }
            QLabel { color: #F8FAFC; font-size: 14px; }
            QPushButton { background-color: #6366F1; color: white; font-weight: bold; border-radius: 6px; padding: 8px 18px; border: none; }
            QPushButton:hover { background-color: #4F46E5; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        lbl = QLabel(message)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        btn = QPushButton("确定")
        btn.clicked.connect(self.accept)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)


class ArchiveManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔐 客户专属档案库 (Customer Profile)")
        self.resize(850, 550)
        self.setStyleSheet("""
            QDialog { background-color: #0F172A; }
            QLabel { color: #38BDF8; font-weight: bold; }
            QListWidget { background-color: #1E293B; color: #F8FAFC; border: 1px solid #334155; border-radius: 8px; }
            QPushButton { background-color: #6366F1; color: white; border-radius: 6px; padding: 6px 12px; }
            QPushButton:hover { background-color: #4F46E5; }
        """)
        
        main_layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(QLabel("👥 顾客名册:"))
        self.cust_list = QListWidget()
        self.cust_list.currentTextChanged.connect(self.on_customer_selected)
        left_layout.addWidget(self.cust_list)
        splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(QLabel("📜 试戴历史记录 & 试戴效果图:"))
        
        self.history_list = QListWidget()
        self.history_list.currentItemChanged.connect(self.on_history_item_selected)
        right_layout.addWidget(self.history_list)

        self.img_preview = QLabel("选择记录预览试戴效果图")
        self.img_preview.setAlignment(Qt.AlignCenter)
        self.img_preview.setStyleSheet("border: 1px dashed #475569; border-radius: 8px;")
        self.img_preview.setFixedHeight(220)
        right_layout.addWidget(self.img_preview)

        splitter.addWidget(right_widget)
        splitter.setSizes([250, 600])

        main_layout.addWidget(splitter)

        self.base_dir = os.path.join(get_exe_dir(), "Customer Profile")
        os.makedirs(self.base_dir, exist_ok=True)
        self.load_customers()

    def load_customers(self):
        self.cust_list.clear()
        if not os.path.exists(self.base_dir): return
        customers = [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]
        self.cust_list.addItems(customers)

    def on_customer_selected(self, cust_name):
        self.history_list.clear()
        self.img_preview.setText("选择记录预览试戴效果图")
        if not cust_name: return

        cust_dir = os.path.join(self.base_dir, cust_name)
        hist_file = os.path.join(cust_dir, "history.json")
        if os.path.exists(hist_file):
            try:
                with open(hist_file, 'r', encoding='utf-8') as f:
                    records = json.load(f)
                    for r in records:
                        item_text = f"⏰ {r.get('time')}  |  款式: {r.get('lens')}"
                        item = QListWidgetItem(item_text)
                        item.setData(Qt.UserRole, r.get("image"))
                        self.history_list.addItem(item)
            except Exception:
                pass

    def on_history_item_selected(self, current, previous):
        if not current: return
        img_filename = current.data(Qt.UserRole)
        cust_name = self.cust_list.currentItem().text()
        if img_filename and cust_name:
            img_path = os.path.join(self.base_dir, cust_name, img_filename)
            if os.path.exists(img_path):
                pix = QPixmap(img_path)
                self.img_preview.setPixmap(pix.scaled(self.img_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.img_preview.setText("未找到效果图照片")


class EyeFittingEngine:
    def __init__(self, high_perf=True):
        self.mp_face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6 if high_perf else 0.4,
            min_tracking_confidence=0.6 if high_perf else 0.4
        )
        self.prev_left = None
        self.prev_right = None
        self.prev_l_radius = None
        self.prev_r_radius = None
        self.high_perf = high_perf

    def reset_history(self):
        self.prev_left = None
        self.prev_right = None
        self.prev_l_radius = None
        self.prev_r_radius = None

    def extract_face_embedding(self, pts, w, h):
        key_indices = [1, 33, 263, 61, 291, 199, 10, 152]
        coords = []
        for idx in key_indices:
            coords.append([pts[idx].x * w, pts[idx].y * h])
        coords = np.array(coords)
        center = np.mean(coords, axis=0)
        coords = coords - center
        scale = np.linalg.norm(coords[1] - coords[2])
        if scale > 0:
            coords /= scale
        return coords.flatten()

    def process_frame(self, frame, config, texture_cache):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb_frame)

        out_frame = frame.copy()
        analysis_data = {
            "pose_valid": True,
            "pupil_distance_mm": 0,
            "eye_width_mm": 0,
            "skin_tone": "Warm",
            "face_vector": None
        }

        if not results.multi_face_landmarks:
            analysis_data["pose_valid"] = False
            return out_frame, analysis_data

        pts = results.multi_face_landmarks[0].landmark
        analysis_data["face_vector"] = self.extract_face_embedding(pts, w, h)

        nose = np.array([pts[1].x * w, pts[1].y * h])
        l_face = np.array([pts[234].x * w, pts[234].y * h])
        r_face = np.array([pts[454].x * w, pts[454].y * h])
        face_width = np.linalg.norm(l_face - r_face)
        
        dist_l = np.linalg.norm(nose - l_face)
        dist_r = np.linalg.norm(nose - r_face)
        if face_width > 0 and abs(dist_l - dist_r) / face_width > 0.28:
            analysis_data["pose_valid"] = False

        lx, ly = int(pts[468].x * w), int(pts[468].y * h)
        rx, ry = int(pts[473].x * w), int(pts[473].y * h)
        
        l_edge_x = int(pts[469].x * w)
        r_edge_x = int(pts[474].x * w)
        raw_l_r = max(6, int(np.hypot(lx - l_edge_x, ly - int(pts[469].y * h))))
        raw_r_r = max(6, int(np.hypot(rx - r_edge_x, ry - int(pts[474].y * h))))

        if self.prev_left is None or not self.high_perf:
            self.prev_left, self.prev_right = (lx, ly), (rx, ry)
            self.prev_l_radius, self.prev_r_radius = raw_l_r, raw_r_r
        else:
            lx = int(self.prev_left[0] * 0.4 + lx * 0.6)
            ly = int(self.prev_left[1] * 0.4 + ly * 0.6)
            rx = int(self.prev_right[0] * 0.4 + rx * 0.6)
            ry = int(self.prev_right[1] * 0.4 + ry * 0.6)
            self.prev_left, self.prev_right = (lx, ly), (rx, ry)
            self.prev_l_radius = int(self.prev_l_radius * 0.6 + raw_l_r * 0.4)
            self.prev_r_radius = int(self.prev_r_radius * 0.6 + raw_r_r * 0.4)

        outer_dist_px = np.hypot((pts[33].x - pts[263].x) * w, (pts[33].y - pts[263].y) * h)
        mm_per_px = 95.0 / max(outer_dist_px, 1.0)
        
        pd_px = np.hypot(lx - rx, ly - ry)
        analysis_data["pupil_distance_mm"] = round(pd_px * mm_per_px, 1)
        
        l_eye_w_px = np.hypot((pts[33].x - pts[133].x) * w, (pts[33].y - pts[133].y) * h)
        analysis_data["eye_width_mm"] = round(l_eye_w_px * mm_per_px, 1)

        left_eyelid_idx = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eyelid_idx = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        l_eyelid_pts = [(int(pts[i].x * w), int(pts[i].y * h)) for i in left_eyelid_idx]
        r_eyelid_pts = [(int(pts[i].x * w), int(pts[i].y * h)) for i in right_eyelid_idx]

        if config.get("mirror", True):
            out_frame = cv2.flip(out_frame, 1)
            lx, rx = w - lx, w - rx
            l_eyelid_pts = [(w - x, y) for x, y in l_eyelid_pts]
            r_eyelid_pts = [(w - x, y) for x, y in r_eyelid_pts]

        if not config.get("compare_mode", False):
            if config.get("enable_left", True):
                self._render_lens(out_frame, (lx, ly), self.prev_l_radius, l_eyelid_pts, config, texture_cache)
            if config.get("enable_right", True):
                self._render_lens(out_frame, (rx, ry), self.prev_r_radius, r_eyelid_pts, config, texture_cache)

        out_frame = self._apply_lighting_and_tone(out_frame, config)
        return out_frame, analysis_data

    def _render_lens(self, frame, center, iris_radius, eyelid_pts, config, texture_cache):
        cx, cy = center
        radius = int(iris_radius * config.get("scale", 1.00))
        if radius <= 0: return

        h, w, _ = frame.shape
        preset_name = config.get("preset_name", "")
        preset = config.get("preset_data", None)
        if not preset: return

        preset_type = preset.get("type", "color")
        if preset_type == "image":
            tex = texture_cache.get(preset_name)
            if tex is None: return
            size = radius * 2
            if size <= 0: return
            lens_img = cv2.resize(tex, (size, size), interpolation=cv2.INTER_AREA)
            lens_center = radius
        else:
            size = radius * 2 + 10
            lens_center = size // 2
            lens_img = np.zeros((size, size, 4), dtype=np.uint8)
            cv2.circle(lens_img, (lens_center, lens_center), radius, (*preset["outer"], 200), -1, cv2.LINE_AA)
            cv2.circle(lens_img, (lens_center, lens_center), int(radius * 0.82), (*preset["inner"], 180), -1, cv2.LINE_AA)
            cv2.circle(lens_img, (lens_center, lens_center), int(radius * 0.38), (0, 0, 0, 0), -1, cv2.LINE_AA)

            feather = max(3, config.get("feather", 5))
            if feather > 0 and lens_img.shape[0] > 0:
                ksize = feather * 2 + 1
                lens_img[:, :, 3] = cv2.GaussianBlur(lens_img[:, :, 3], (ksize, ksize), 0)

        y1, y2 = max(0, cy - lens_center), min(h, cy + lens_center)
        x1, x2 = max(0, cx - lens_center), min(w, cx + lens_center)
        ly1, ly2 = max(0, -(cy - lens_center)), size - max(0, (cy + lens_center) - h)
        lx1, lx2 = max(0, -(cx - lens_center)), size - max(0, (cx + lens_center) - w)

        if y1 >= y2 or x1 >= x2 or ly1 >= ly2 or lx1 >= lx2: return

        roi_frame = frame[y1:y2, x1:x2].copy()
        roi_lens = lens_img[ly1:ly2, lx1:lx2]

        eyelid_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(eyelid_mask, [np.array(eyelid_pts, dtype=np.int32)], 255)
        eyelid_mask_roi = eyelid_mask[y1:y2, x1:x2] / 255.0

        alpha = min(0.65, config.get("alpha", 0.55))
        alpha_lens = (roi_lens[:, :, 3] / 255.0) * alpha * eyelid_mask_roi
        alpha_lens = np.expand_dims(alpha_lens, axis=-1)

        blended = alpha_lens * roi_lens[:, :, :3] + (1.0 - alpha_lens) * roi_frame
        frame[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    def _apply_lighting_and_tone(self, img, config):
        scene = config.get("scene", "Natural")
        if scene == "Warm Light":
            img = cv2.addWeighted(img, 0.95, np.full_like(img, (10, 25, 45)), 0.05, 0)
        elif scene == "Cold Light":
            img = cv2.addWeighted(img, 0.95, np.full_like(img, (40, 20, 10)), 0.05, 0)
        elif scene == "Night":
            img = cv2.convertScaleAbs(img, alpha=0.8, beta=-15)

        temp = config.get("temperature", 0)
        bright = config.get("brightness", 0)
        
        if temp != 0 or bright != 0:
            img = img.astype(np.int16)
            if temp > 0: img[:, :, 2] += temp
            elif temp < 0: img[:, :, 0] += abs(temp)
            img += bright
            img = np.clip(img, 0, 255).astype(np.uint8)

        return img


class AdvancedLensStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("憨憨美瞳面诊体验馆 - AI 智能试戴系统")
        
        icon_path = resource_path(os.path.join("pictures", "hanhan.jpg"))
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.resize(1380, 780)

        self.NONE_LENS = "00. 原生眼睛 (未佩戴)"
        self.lens_presets = {
            self.NONE_LENS: None,
            "🔥 旗木卡卡西 · 写轮眼": {"type": "image", "path": resource_path(os.path.join("pictures", "kakaxi.png")), "category": "混血款", "dia": "14.5mm", "bc": "8.6mm", "water": "38%", "price": "¥168", "desc": "动漫风高显色神作"},
            "🔥 宇智波佐助 · 万花筒": {"type": "image", "path": resource_path(os.path.join("pictures", "zuozhu.png")), "category": "混血款", "dia": "14.5mm", "bc": "8.6mm", "water": "38%", "price": "¥168", "desc": "高冷写实纹理"},
            "🔥 宇智波鼬 · 万花筒": {"type": "image", "path": resource_path(os.path.join("pictures", "you.jpg")), "category": "混血款", "dia": "14.2mm", "bc": "8.5mm", "water": "42%", "price": "¥158", "desc": "复古暗红轮廓"},
            "01. 金晨锁边黑 (Soft Black)": {"type": "color", "inner": (50, 50, 50), "outer": (15, 15, 15), "category": "自然款", "dia": "14.0mm", "bc": "8.6mm", "water": "55%", "price": "¥99", "desc": "日常百搭 锁边锁眼"},
            "02. 奶油栗子棕 (Chestnut)": {"type": "color", "inner": (40, 80, 130), "outer": (20, 45, 80), "category": "浅瞳", "dia": "14.2mm", "bc": "8.6mm", "water": "42%", "price": "¥118", "desc": "温柔日系通透感"},
            "03. 清晨雾灰 (Misty Gray)": {"type": "color", "inner": (160, 160, 160), "outer": (60, 60, 60), "category": "深瞳", "dia": "14.2mm", "bc": "8.7mm", "water": "38%", "price": "¥128", "desc": "高级冷感水光纹"},
            "04. 微醺橄榄绿 (Olive Green)": {"type": "color", "inner": (70, 130, 90), "outer": (25, 60, 35), "category": "混血款", "dia": "14.5mm", "bc": "8.6mm", "water": "38%", "price": "¥138", "desc": "轻混血野猫气场"},
            "05. 蜜桃乌龙 (Peach Oolong)": {"type": "color", "inner": (120, 140, 220), "outer": (50, 70, 150), "category": "浅瞳", "dia": "14.2mm", "bc": "8.6mm", "water": "42%", "price": "¥118", "desc": "粉棕少女高光感"},
            "06. 极光冰蓝 (Aurora Blue)": {"type": "color", "inner": (200, 180, 100), "outer": (120, 80, 20), "category": "混血款", "dia": "14.5mm", "bc": "8.6mm", "water": "38%", "price": "¥148", "desc": "异域风情冰晶蓝"},
            "07. 焦糖玛奇朵 (Caramel)": {"type": "color", "inner": (30, 90, 160), "outer": (10, 40, 90), "category": "自然款", "dia": "14.0mm", "bc": "8.6mm", "water": "50%", "price": "¥108", "desc": "深瞳必入暖棕高光"},
            "08. 星空冷蓝 (Starry Blue)": {"type": "color", "inner": (180, 120, 50), "outer": (90, 50, 20), "category": "混血款", "dia": "14.5mm", "bc": "8.6mm", "water": "38%", "price": "¥138", "desc": "璀璨星空渐变"},
            "09. 薄荷摩卡 (Mint Mocha)": {"type": "color", "inner": (90, 120, 70), "outer": (40, 60, 30), "category": "自然款", "dia": "14.2mm", "bc": "8.6mm", "water": "45%", "price": "¥118", "desc": "清爽复古巧色"},
            "10. 仙气紫罗兰 (Violet)": {"type": "color", "inner": (180, 80, 140), "outer": (90, 30, 70), "category": "浅瞳", "dia": "14.2mm", "bc": "8.6mm", "water": "40%", "price": "¥128", "desc": "仙气十足梦幻紫"},
            "11. 琥珀蜜棕 (Amber Brown)": {"type": "color", "inner": (20, 100, 180), "outer": (10, 50, 100), "category": "自然款", "dia": "14.0mm", "bc": "8.6mm", "water": "50%", "price": "¥108", "desc": "琥珀般清亮眼神"}
        }

        self.texture_cache = {}
        self.load_textures()

        self.engine = EyeFittingEngine(high_perf=True)
        self.cap = None
        self.static_image = None
        self.is_live_mode = True
        self.current_recognized_user = None
        self.face_db = {}
        self.load_face_database()
        
        self.render_config = {
            "preset_name": self.NONE_LENS,
            "preset_data": None,
            "alpha": 0.55,
            "scale": 1.00,
            "feather": 6,
            "enable_left": True,
            "enable_right": True,
            "compare_mode": False,
            "mirror": True,
            "temperature": 0,
            "brightness": 0,
            "scene": "Natural"
        }

        self.init_ui()
        self.start_camera()

    def load_face_database(self):
        base_dir = os.path.join(get_exe_dir(), "Customer Profile")
        if not os.path.exists(base_dir): return
        for cust_name in os.listdir(base_dir):
            feature_file = os.path.join(base_dir, cust_name, "face_feature.npy")
            if os.path.exists(feature_file):
                try:
                    self.face_db[cust_name] = np.load(feature_file)
                except Exception:
                    pass

    def load_textures(self):
        for name, data in self.lens_presets.items():
            if data and data.get("type") == "image":
                tex = load_lens_texture(data["path"])
                if tex is not None:
                    self.texture_cache[name] = tex

    def init_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0F172A; }
            QLabel { color: #F8FAFC; font-family: 'Segoe UI', 'Microsoft YaHei'; }
            QGroupBox { color: #38BDF8; background-color: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px; margin-top: 18px; padding-top: 15px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 12px; padding: 0 6px; background-color: #0F172A; }
            QPushButton { background-color: #6366F1; color: white; font-weight: bold; border-radius: 8px; padding: 8px 12px; border: none; }
            QPushButton:hover { background-color: #4F46E5; }
            QComboBox, QLineEdit { background-color: #1E293B; color: white; border: 1px solid #475569; border-radius: 6px; padding: 5px; }
            QTabWidget::pane { border: 1px solid #334155; border-radius: 8px; }
            QTabBar::tab { background: #1E293B; color: #94A3B8; padding: 8px 16px; border-radius: 6px; }
            QTabBar::tab:selected { background: #6366F1; color: white; font-weight: bold; }
        """)

        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        
        self.main_view_label = QLabel("相机加载中...")
        self.main_view_label.setAlignment(Qt.AlignCenter)
        self.main_view_label.setStyleSheet("background: #020617; border-radius: 12px; border: 2px solid #334155;")
        self.main_view_label.setFixedSize(960, 540)

        self.overlay_info_bar = QLabel("👋 欢迎新用户！  |  瞳距: -- mm")
        self.overlay_info_bar.setStyleSheet("background: rgba(15, 23, 42, 0.85); color: #38BDF8; padding: 10px; border-radius: 6px; font-weight: bold; font-size: 15px;")
        self.overlay_info_bar.setFixedWidth(960)

        left_layout.addWidget(self.main_view_label)
        left_layout.addWidget(self.overlay_info_bar)
        left_layout.addStretch()

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFixedWidth(380)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_fitting_tab(), "🎨 试戴调参")
        self.tabs.addTab(self._build_compare_tab(), "🖼️ 多款对比/场景")
        self.tabs.addTab(self._build_business_tab(), "💼 商业/档案")

        right_layout.addWidget(self.tabs)
        
        quick_control = QHBoxLayout()
        self.btn_capture = QPushButton("📸 确认保存客户试戴截图")
        self.btn_capture.setStyleSheet("background-color: #10B981; font-size: 14px; padding: 10px;")
        self.btn_capture.clicked.connect(self.capture_and_save_workflow)
        
        self.btn_ai_recommend = QPushButton("✨ AI 推荐")
        self.btn_ai_recommend.setStyleSheet("background-color: #EC4899;")
        self.btn_ai_recommend.clicked.connect(self.run_ai_recommendation)

        quick_control.addWidget(self.btn_capture)
        quick_control.addWidget(self.btn_ai_recommend)

        right_layout.addLayout(quick_control)
        right_scroll.setWidget(right_widget)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_scroll)

        self.setCentralWidget(central_widget)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame_pipeline)

    def _build_fitting_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        source_group = QGroupBox("图像采集源")
        s_layout = QHBoxLayout()
        self.btn_src_cam = QRadioButton("实时摄像头")
        self.btn_src_cam.setChecked(True)
        self.btn_src_cam.toggled.connect(self.switch_source)
        self.btn_src_file = QRadioButton("本地自拍照")
        s_layout.addWidget(self.btn_src_cam)
        s_layout.addWidget(self.btn_src_file)
        self.btn_upload = QPushButton("上传图片")
        self.btn_upload.clicked.connect(self.upload_local_image)
        s_layout.addWidget(self.btn_upload)
        source_group.setLayout(s_layout)
        layout.addWidget(source_group)

        style_group = QGroupBox("款式挑选")
        st_layout = QVBoxLayout()
        self.cat_combo = QComboBox()
        self.cat_combo.addItems(["全部", "自然款", "混血款", "浅瞳", "深瞳"])
        self.cat_combo.currentTextChanged.connect(self.filter_lens_category)
        
        self.lens_combo = QComboBox()
        self.update_lens_combo_items()
        self.lens_combo.currentTextChanged.connect(self.on_lens_selected)

        st_layout.addWidget(QLabel("分类筛选:"))
        st_layout.addWidget(self.cat_combo)
        st_layout.addWidget(QLabel("选择款式:"))
        st_layout.addWidget(self.lens_combo)
        style_group.setLayout(st_layout)
        layout.addWidget(style_group)

        self.info_card = QGroupBox("产品参数卡")
        ic_layout = QVBoxLayout()
        self.lbl_product_info = QLabel("请选择款式查看属性...")
        self.lbl_product_info.setWordWrap(True)
        ic_layout.addWidget(self.lbl_product_info)
        self.info_card.setLayout(ic_layout)
        layout.addWidget(self.info_card)

        param_group = QGroupBox("试戴微调参数")
        p_layout = QVBoxLayout()

        p_layout.addWidget(QLabel("透明度 (Alpha):"))
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(10, 100)
        self.slider_alpha.setValue(55)
        self.slider_alpha.valueChanged.connect(lambda v: self.render_config.update({"alpha": v/100.0}))
        p_layout.addWidget(self.slider_alpha)

        p_layout.addWidget(QLabel("缩放比例 (Scale):"))
        self.slider_scale = QSlider(Qt.Horizontal)
        self.slider_scale.setRange(80, 150)
        self.slider_scale.setValue(100)
        self.slider_scale.valueChanged.connect(lambda v: self.render_config.update({"scale": v/100.0}))
        p_layout.addWidget(self.slider_scale)

        sw_layout = QHBoxLayout()
        self.chk_left = QCheckBox("左眼")
        self.chk_left.setChecked(True)
        self.chk_left.toggled.connect(lambda c: self.render_config.update({"enable_left": c}))
        self.chk_right = QCheckBox("右眼")
        self.chk_right.setChecked(True)
        self.chk_right.toggled.connect(lambda c: self.render_config.update({"enable_right": c}))
        self.chk_compare = QCheckBox("原图对比")
        self.chk_compare.toggled.connect(lambda c: self.render_config.update({"compare_mode": c}))

        sw_layout.addWidget(self.chk_left)
        sw_layout.addWidget(self.chk_right)
        sw_layout.addWidget(self.chk_compare)
        p_layout.addLayout(sw_layout)
        param_group.setLayout(p_layout)
        layout.addWidget(param_group)

        dog_group = QGroupBox("小店形象代言狗")
        dog_layout = QHBoxLayout()
        dog_avatar = QLabel()
        dog_avatar.setFixedSize(80, 80)
        dog_avatar.setStyleSheet("border-radius: 10px; border: 2px solid #38BDF8;")
        dog_avatar.setScaledContents(True)
        
        dog_pix = QPixmap(resource_path(os.path.join("pictures", "hanhan2.jpg")))
        if not dog_pix.isNull():
            dog_avatar.setPixmap(dog_pix)
        else:
            dog_avatar.setText("🐶 憨憨")

        dog_msg = QLabel("<b>欢迎光临~ 🐾🐾</b><br><font color='#F43F5E'>我是店长【憨憨】</font><br>祝您挑选到心仪的美瞳！ ✨")
        dog_msg.setWordWrap(True)

        dog_layout.addWidget(dog_avatar)
        dog_layout.addWidget(dog_msg)
        dog_group.setLayout(dog_layout)
        layout.addWidget(dog_group)

        layout.addStretch()
        return panel

    def _build_compare_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        env_group = QGroupBox("环境光与色温调节")
        e_layout = QVBoxLayout()
        
        e_layout.addWidget(QLabel("色温 (Temperature):"))
        self.slider_temp = QSlider(Qt.Horizontal)
        self.slider_temp.setRange(-40, 40)
        self.slider_temp.setValue(0)
        self.slider_temp.valueChanged.connect(lambda v: self.render_config.update({"temperature": v}))
        e_layout.addWidget(self.slider_temp)

        e_layout.addWidget(QLabel("亮度 (Brightness):"))
        self.slider_bright = QSlider(Qt.Horizontal)
        self.slider_bright.setRange(-40, 40)
        self.slider_bright.setValue(0)
        self.slider_bright.valueChanged.connect(lambda v: self.render_config.update({"brightness": v}))
        e_layout.addWidget(self.slider_bright)

        e_layout.addWidget(QLabel("场景模式:"))
        sc_combo = QComboBox()
        sc_combo.addItems(["Natural", "Warm Light", "Cold Light", "Night"])
        sc_combo.currentTextChanged.connect(lambda s: self.render_config.update({"scene": s}))
        e_layout.addWidget(sc_combo)

        env_group.setLayout(e_layout)
        layout.addWidget(env_group)
        layout.addStretch()
        return panel

    def _build_business_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        customer_group = QGroupBox("顾客试戴档案管理")
        c_layout = QVBoxLayout()

        btn_open_archive = QPushButton("🔐 查看顾客独立文件夹档案库")
        btn_open_archive.setStyleSheet("background-color: #0EA5E9; padding: 10px;")
        btn_open_archive.clicked.connect(self.open_archive_with_password)
        c_layout.addWidget(btn_open_archive)

        customer_group.setLayout(c_layout)
        layout.addWidget(customer_group)

        setting_group = QGroupBox("系统设置")
        st_layout = QVBoxLayout()
        
        self.chk_perf = QCheckBox("高帧率渲染模式")
        self.chk_perf.setChecked(True)
        self.chk_perf.toggled.connect(self.toggle_perf_mode)
        st_layout.addWidget(self.chk_perf)

        self.chk_mirror = QCheckBox("画面镜像显示")
        self.chk_mirror.setChecked(True)
        self.chk_mirror.toggled.connect(lambda m: self.render_config.update({"mirror": m}))
        st_layout.addWidget(self.chk_mirror)

        self.chk_watermark = QCheckBox("导出成品图带店铺水印")
        self.chk_watermark.setChecked(True)
        st_layout.addWidget(self.chk_watermark)

        btn_import = QPushButton("📂 批量导入素材文件夹")
        btn_import.clicked.connect(self.import_custom_folder)
        st_layout.addWidget(btn_import)

        setting_group.setLayout(st_layout)
        layout.addWidget(setting_group)

        layout.addStretch()
        return panel

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.timer.start(16)

    def switch_source(self):
        if self.btn_src_cam.isChecked():
            self.is_live_mode = True
            if not self.cap or not self.cap.isOpened(): self.start_camera()
        else:
            self.is_live_mode = False

    def upload_local_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择自拍照", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if file_path:
            img = cv2_imread_unicode(file_path, cv2.IMREAD_COLOR)
            if img is not None:
                self.static_image = img
                self.btn_src_file.setChecked(True)
                self.is_live_mode = False
                self.engine.reset_history()
                self.update_frame_pipeline()

    def filter_lens_category(self, cat):
        self.update_lens_combo_items(cat)

    def update_lens_combo_items(self, category="全部"):
        self.lens_combo.blockSignals(True)
        self.lens_combo.clear()
        for name, data in self.lens_presets.items():
            if category == "全部" or (data and data.get("category") == category):
                self.lens_combo.addItem(name)
        self.lens_combo.blockSignals(False)
        self.on_lens_selected(self.lens_combo.currentText())

    def on_lens_selected(self, text):
        if not hasattr(self, 'lbl_product_info') or not text: return
        self.render_config["preset_name"] = text
        data = self.lens_presets.get(text)
        self.render_config["preset_data"] = data

        if data:
            info = f"<b>【{text}】</b><br>" \
                   f"🏷️ 分类: {data.get('category','通用')}<br>" \
                   f"📐 直径: {data.get('dia','14.2mm')} | 基弧: {data.get('bc','8.6mm')}<br>" \
                   f"💧 含水量: {data.get('water','38%')}<br>" \
                   f"💰 价格: <font color='#F43F5E'><b>{data.get('price','¥128')}</b></font><br>" \
                   f"📝 描述: {data.get('desc','暂无')}"
            self.lbl_product_info.setText(info)
        else:
            self.lbl_product_info.setText("原生眼睛，无美瞳样式。")

    def update_frame_pipeline(self):
        if self.is_live_mode:
            if not self.cap: return
            ret, frame = self.cap.read()
            if not ret: return
        else:
            if self.static_image is None: return
            frame = self.static_image.copy()

        processed_frame, analysis = self.engine.process_frame(frame, self.render_config, self.texture_cache)

        self.current_face_vector = analysis["face_vector"]
        matched_user = None
        if self.current_face_vector is not None and len(self.face_db) > 0:
            min_dist = float('inf')
            for user_name, vec in self.face_db.items():
                dist = np.linalg.norm(self.current_face_vector - vec)
                if dist < min_dist:
                    min_dist = dist
                    matched_user = user_name
            if min_dist > 0.45:
                matched_user = None

        self.current_recognized_user = matched_user

        if matched_user:
            welcome_str = f"👋 欢迎老用户：<font color='#10B981'><b>【{matched_user}】</b></font>"
        else:
            welcome_str = "👋 欢迎新用户！"

        self.overlay_info_bar.setText(
            f"{welcome_str}  |  瞳距: {analysis['pupil_distance_mm']} mm  |  "
            f"眼裂: {analysis['eye_width_mm']} mm"
        )

        self._display_mat_to_label(processed_frame, self.main_view_label)

    def _display_mat_to_label(self, mat, label):
        h, w, c = mat.shape
        q_img = QImage(mat.data, w, h, w * c, QImage.Format_BGR888)
        pix = QPixmap.fromImage(q_img)
        scaled_pix = pix.scaled(960, 540, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pix)

    def capture_and_save_workflow(self):
        if not self.main_view_label.pixmap(): return

        default_name = self.current_recognized_user if self.current_recognized_user else ""
        cust_name, ok = QInputDialog.getText(
            self, "确认顾客信息", 
            "👤 请输入/确认顾客姓名（老用户自动归档，新用户自动建档）：", 
            QLineEdit.Normal, default_name
        )

        if not ok or not cust_name.strip():
            return

        cust_name = cust_name.strip()
        
        cust_dir = os.path.join(get_exe_dir(), "Customer Profile", cust_name)
        os.makedirs(cust_dir, exist_ok=True)

        if hasattr(self, 'current_face_vector') and self.current_face_vector is not None:
            np.save(os.path.join(cust_dir, "face_feature.npy"), self.current_face_vector)
            self.face_db[cust_name] = self.current_face_vector

        time_str = time.strftime("%Y%m%d_%H%M%S")
        img_filename = f"{time_str}_{self.render_config['preset_name'][:6]}.png"
        full_img_path = os.path.join(cust_dir, img_filename)

        pixmap = self.main_view_label.pixmap().copy()
        if self.chk_watermark.isChecked():
            painter = QPainter(pixmap)
            painter.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
            painter.setPen(QColor(255, 255, 255, 180))
            painter.drawText(20, pixmap.height() - 30, "PRODUCER: 憨憨美瞳面诊体验馆")
            painter.end()

        pixmap.save(full_img_path)

        hist_file = os.path.join(cust_dir, "history.json")
        history = []
        if os.path.exists(hist_file):
            try:
                with open(hist_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except Exception: pass

        history.append({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "lens": self.render_config["preset_name"],
            "image": img_filename
        })

        with open(hist_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=4)

        dlg = CustomDialog("保存成功", f"✅ 顾客【{cust_name}】的试戴照片与档案已成功保存入专属文件夹！\n📁 {cust_dir}", self)
        dlg.exec_()

    def run_ai_recommendation(self):
        msg = "✨ AI 智能诊断：推荐【03. 清晨雾灰】与【微醺橄榄绿】，极具通透水光感！"
        dlg = CustomDialog("✨ AI 智能推荐", msg, self)
        dlg.exec_()
        self.lens_combo.setCurrentText("03. 清晨雾灰 (Misty Gray)")

    def open_archive_with_password(self):
        pwd, ok = QInputDialog.getText(self, "身份验证", "🔑 请输入管理密码访问客户档案库:", QLineEdit.Password)
        if ok:
            if pwd == "123456":
                arch_dlg = ArchiveManagerDialog(self)
                arch_dlg.exec_()
            else:
                dlg = CustomDialog("验证失败", "❌ 密码错误，无法访问档案！", self)
                dlg.exec_()

    def toggle_perf_mode(self, enabled):
        self.engine = EyeFittingEngine(high_perf=enabled)

    def import_custom_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择自定义美瞳图片素材文件夹")
        if folder:
            count = 0
            # 扩展支持所有常见图片格式（包括大小写及 webp）
            valid_exts = ('.png', '.jpg', '.jpeg', '.webp', '.PNG', '.JPG', '.JPEG', '.WEBP')
            for file in os.listdir(folder):
                if file.endswith(valid_exts):
                    path = os.path.join(folder, file)
                    name = f"自定义_{os.path.splitext(file)[0]}"
                    tex = load_lens_texture(path)
                    if tex is not None:
                        self.texture_cache[name] = tex
                        self.lens_presets[name] = {
                            "type": "image", "path": path, "category": "自然款", 
                            "dia": "14.2mm", "bc": "8.6mm", "water": "38%", "price": "自定义", "desc": "外部导入素材"
                        }
                        count += 1
            self.update_lens_combo_items()
            dlg = CustomDialog("导入成功", f"🎉 成功导入 {count} 款美瞳素材！", self)
            dlg.exec_()

    def closeEvent(self, event):
        if self.cap and self.cap.isOpened(): self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyleSheet("""
        QDialog, QInputDialog {
            background-color: #1E293B !important;
            border: 2px solid #38BDF8 !important;
            border-radius: 12px !important;
        }
        QDialog QLabel, QInputDialog QLabel {
            color: #F8FAFC !important;
            font-size: 15px !important;
            font-weight: bold !important;
        }
        QInputDialog QLineEdit {
            background-color: #0F172A !important;
            color: #38BDF8 !important;
            border: 2px solid #6366F1 !important;
            border-radius: 8px !important;
            padding: 8px !important;
            font-size: 15px !important;
            font-weight: bold !important;
        }
        QInputDialog QPushButton {
            background-color: #6366F1 !important;
            color: #FFFFFF !important;
            font-weight: bold !important;
            border-radius: 6px !important;
            padding: 8px 18px !important;
            min-width: 80px !important;
        }
        QInputDialog QPushButton:hover {
            background-color: #4F46E5 !important;
        }
    """)

    window = AdvancedLensStudio()
    window.show()
    sys.exit(app.exec_())