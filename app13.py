import sys
import os
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QPushButton, QVBoxLayout, QHBoxLayout, QSlider, QComboBox, QGroupBox, QGraphicsDropShadowEffect, QSizePolicy)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPainterPath, QIcon

# 解决高分屏兼容性问题
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

# 兼容性导入 MediaPipe
try:
    mp_face_mesh = mp.solutions.face_mesh
except AttributeError:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh


# ==========================================
# 核心：防崩溃路径适配器（兼容各类 PyInstaller 打包环境）
# ==========================================
def resource_path(relative_path):
    """
    终极资源路径兼容函数：
    解决切换文件夹名称/移动目录后报 'The path does not exist' 的问题
    """
    # 1. PyInstaller --onefile 单文件解压模式
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    
    # 2. PyInstaller --onedir 文件夹打包模式 (PyInstaller 6+)
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
        # 优先从 _internal 文件夹读取
        path_in_internal = os.path.join(base_dir, '_internal', relative_path)
        if os.path.exists(path_in_internal):
            return path_in_internal
        # 找不到则从 exe 根目录读取
        path_in_exe = os.path.join(base_dir, relative_path)
        if os.path.exists(path_in_exe):
            return path_in_exe

    # 3. 未打包的源码运行环境
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


# ==========================================
# 安全读取图片 (防止包含中文路径时 OpenCV 报错)
# ==========================================
def cv2_imread_unicode(file_path, flags=cv2.IMREAD_UNCHANGED):
    if not os.path.exists(file_path):
        return None
    try:
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags)
    except Exception:
        return None


def load_lens_texture(img_path):
    """
    智能眼珠抠图加载器：
    自动识别有效眼珠区域，剔除四周的白边/方框
    """
    raw_img = cv2_imread_unicode(img_path, cv2.IMREAD_UNCHANGED)
    if raw_img is None:
        return None

    # 1. 如果图片带 transparent alpha 透明通道 (4通道 PNG)
    if len(raw_img.shape) == 3 and raw_img.shape[-1] == 4:
        b, g, r, alpha = cv2.split(raw_img)
        img_bgr = cv2.merge([b, g, r])
        if np.any(alpha > 20):
            contours, _ = cv2.findContours((alpha > 20).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_cnt = max(contours, key=cv2.contourArea)
                (cx, cy), radius = cv2.minEnclosingCircle(max_cnt)
                cx, cy, radius = int(cx), int(cy), int(radius)
            else:
                h, w = alpha.shape
                cx, cy, radius = w // 2, h // 2, min(w, h) // 2
            
            radius = int(radius * 0.98)
            x1, x2 = max(0, cx - radius), min(img_bgr.shape[1], cx + radius)
            y1, y2 = max(0, cy - radius), min(img_bgr.shape[0], cy + radius)
            cropped_bgr = img_bgr[y1:y2, x1:x2]
            
            ch, cw = cropped_bgr.shape[:2]
            if ch <= 0 or cw <= 0: return None
            
            alpha_mask = np.zeros((ch, cw), dtype=np.uint8)
            cv2.circle(alpha_mask, (cw // 2, ch // 2), min(cw, ch) // 2 - 1, 255, -1, cv2.LINE_AA)
            return cv2.merge([cropped_bgr[:,:,0], cropped_bgr[:,:,1], cropped_bgr[:,:,2], alpha_mask])

    # 2. 如果是普通 JPG 或 3通道 PNG
    img = cv2_imread_unicode(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 剔除纯白背景
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_cnt = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(max_cnt)
        cx, cy, radius = int(cx), int(cy), int(radius)
    else:
        cx, cy, radius = w // 2, h // 2, min(w, h) // 2

    # 半径内缩 5% 彻底滤去边缘杂色
    crop_r = int(radius * 0.95)
    if crop_r <= 5:
        crop_r = min(w, h) // 2

    x1, x2 = max(0, cx - crop_r), min(w, cx + crop_r)
    y1, y2 = max(0, cy - crop_r), min(h, cy + crop_r)
    cropped = img[y1:y2, x1:x2]

    ch, cw = cropped.shape[:2]
    if ch <= 0 or cw <= 0:
        return None

    b, g, r_chan = cv2.split(cropped)

    # 动态生成圆润的边缘
    alpha = np.zeros((ch, cw), dtype=np.uint8)
    cv2.circle(alpha, (cw // 2, ch // 2), min(cw, ch) // 2 - 1, 255, -1, cv2.LINE_AA)

    return cv2.merge([b, g, r_chan, alpha])


class BackgroundWidget(QWidget):
    """自定义背景画板"""
    def __init__(self, bg_path, parent=None):
        super().__init__(parent)
        self.bg_pixmap = None
        img = cv2_imread_unicode(bg_path, cv2.IMREAD_COLOR)
        if img is not None:
            h, w, _ = img.shape
            dark_overlay = np.zeros_like(img)
            img = cv2.addWeighted(img, 0.8, dark_overlay, 0.2, 0)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            q_img = QImage(img_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            self.bg_pixmap = QPixmap.fromImage(q_img)

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.bg_pixmap and not self.bg_pixmap.isNull():
            scaled = self.bg_pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.fillRect(self.rect(), QColor(15, 23, 42))


class HanHanContactLensApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("憨憨美瞳店 - AI 智能试戴系统")
        self.resize(1280, 800)
        self.setMinimumSize(1020, 680)

        # 动态转换图片路径（修改为 pictures/ 目录）
        self.bg_image_path = resource_path(os.path.join("pictures", "hanhan.jpg"))
        self.hanhan2_image_path = resource_path(os.path.join("pictures", "hanhan2.jpg"))
        
        if os.path.exists(self.bg_image_path):
            self.setWindowIcon(QIcon(self.bg_image_path))
        
        # 初始化 MediaPipe 人脸与虹膜检测
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.prev_left = None
        self.prev_right = None
        self.prev_l_radius = None
        self.prev_r_radius = None
        self.current_frame = None

        # 款式配置列表（修改图片路径为 pictures/ 目录）
        self.NONE_LENS = "00. 原生眼睛 (不佩戴)"
        self.lens_presets = {
            self.NONE_LENS: None,
            "🔥 旗木卡卡西 · 写轮眼": {"type": "image", "path": resource_path(os.path.join("pictures", "kakaxi.png"))},
            "🔥 宇智波佐助 · 万花筒": {"type": "image", "path": resource_path(os.path.join("pictures", "zuozhu.png"))},
            "🔥 宇智波鼬 · 万花筒": {"type": "image", "path": resource_path(os.path.join("pictures", "you.jpg"))},
            "🔥 六道仙人 · 六勾玉轮回眼": {"type": "image", "path": resource_path(os.path.join("pictures", "lunhui.jpg"))},
            "01. 金晨锁边黑 (Soft Black)": {"type": "color", "inner": (50, 50, 50), "outer": (15, 15, 15)},
            "02. 奶油栗子棕 (Chestnut Brown)": {"type": "color", "inner": (40, 80, 130), "outer": (20, 45, 80)},
            "03. 乌龙奶茶棕 (Milk Tea Brown)": {"type": "color", "inner": (70, 110, 160), "outer": (30, 60, 95)},
            "04. 落日琥珀棕 (Amber Glow)": {"type": "color", "inner": (30, 120, 190), "outer": (10, 60, 110)},
            "05. 清晨雾灰 (Misty Gray)": {"type": "color", "inner": (160, 160, 160), "outer": (60, 60, 60)},
            "06. 冰川冷灰 (Glacier Gray)": {"type": "color", "inner": (190, 185, 170), "outer": (90, 85, 70)},
            "07. 浅空洋甘菊 (Chamomiles Gray)": {"type": "color", "inner": (110, 170, 180), "outer": (50, 90, 100)},
            "08. 微醺橄榄绿 (Olive Green)": {"type": "color", "inner": (70, 130, 90), "outer": (25, 60, 35)},
            "09. 薄荷琉璃绿 (Mint Glass)": {"type": "color", "inner": (140, 190, 100), "outer": (60, 100, 40)},
            "10. 玫瑰粉棕 (Rose Brown)": {"type": "color", "inner": (110, 100, 180), "outer": (50, 40, 100)},
            "11. 仙气莓莓紫 (Berry Violet)": {"type": "color", "inner": (180, 110, 210), "outer": (90, 40, 120)},
            "12. 暗夜星空蓝 (Night Star Blue)": {"type": "color", "inner": (210, 130, 50), "outer": (110, 50, 15)},
            "13. 绝美冰海蓝 (Ocean Crystal)": {"type": "color", "inner": (230, 180, 90), "outer": (120, 80, 30)},
            "14. 微醺波尔多红 (Bordeaux Wine)": {"type": "color", "inner": (80, 60, 160), "outer": (30, 20, 80)},
            "15. 香槟落日金 (Champagne Gold)": {"type": "color", "inner": (80, 180, 220), "outer": (30, 90, 130)}
        }
        
        self.texture_cache = {}
        self.load_image_textures()

        self.current_preset = None
        self.current_preset_name = self.NONE_LENS
        self.alpha = 0.85
        self.size_scale = 1.15       

        self.init_ui()

    def load_image_textures(self):
        """预加载素材库中的图片"""
        for name, data in self.lens_presets.items():
            if data and data.get("type") == "image":
                path = data["path"]
                tex = load_lens_texture(path)
                if tex is not None:
                    self.texture_cache[name] = tex

    def get_rounded_pixmap(self, img_path, target_size, radius=12):
        if not os.path.exists(img_path):
            return QPixmap()
        
        src_img = cv2_imread_unicode(img_path, cv2.IMREAD_COLOR)
        if src_img is None:
            return QPixmap()
            
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        h, w, _ = src_img.shape
        q_src = QImage(src_img.data, w, h, 3 * w, QImage.Format_RGB888)
        src = QPixmap.fromImage(q_src)

        scaled_src = src.scaled(target_size, target_size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        
        x = (scaled_src.width() - target_size) // 2
        y = (scaled_src.height() - target_size) // 2
        cropped = scaled_src.copy(x, y, target_size, target_size)

        out_img = QPixmap(target_size, target_size)
        out_img.fill(Qt.transparent)

        painter = QPainter(out_img)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(0, 0, target_size, target_size, radius, radius)
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, cropped)
        painter.end()

        return out_img

    def init_ui(self):
        self.setStyleSheet("""
            QLabel { color: #FFFFFF; font-family: 'Microsoft YaHei', 'Segoe UI'; }
            QGroupBox { 
                color: #F8FAFC; 
                background-color: rgba(15, 23, 42, 0.75); 
                border: 1px solid rgba(255, 255, 255, 0.15); 
                border-radius: 12px; 
                margin-top: 14px; 
                font-weight: bold;
                font-size: 13px;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 12px; 
                padding: 3px 10px; 
                background-color: #6366F1;
                color: #FFFFFF;
                border-radius: 5px;
            }
            QComboBox {
                background-color: rgba(30, 41, 59, 0.9);
                color: #FFFFFF;
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 13px;
                font-weight: bold;
            }
            QComboBox QAbstractItemView {
                background-color: #1E293B;
                color: #FFFFFF;
                selection-background-color: #6366F1;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #818CF8;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QPushButton {
                background-color: #6366F1;
                color: #FFFFFF;
                font-weight: bold;
                font-size: 14px;
                border-radius: 10px;
                padding: 12px;
                border: 1px solid rgba(255, 255, 255, 0.15);
            }
            QPushButton:hover { background-color: #4F46E5; }
        """)

        bg_widget = BackgroundWidget(self.bg_image_path, self)
        self.setCentralWidget(bg_widget)
        
        main_layout = QHBoxLayout(bg_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        self.video_label = QLabel("正在初始化摄像头检测系统...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("""
            background-color: #0F172A; 
            border: 2px solid rgba(255, 255, 255, 0.25); 
            border-radius: 16px; 
            color: #94A3B8;
            font-size: 15px;
        """)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 160))
        shadow.setOffset(0, 6)
        self.video_label.setGraphicsEffect(shadow)

        self.info_card = QWidget()
        self.info_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.info_card.setStyleSheet("""
            QWidget {
                background-color: rgba(15, 23, 42, 0.85);
                border: 1px solid rgba(99, 102, 241, 0.5);
                border-radius: 14px;
            }
        """)
        info_layout = QHBoxLayout(self.info_card)
        info_layout.setContentsMargins(20, 10, 20, 10)

        tag_label = QLabel("CURRENT FITTING")
        tag_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        tag_label.setStyleSheet("color: #818CF8; background: transparent; letter-spacing: 1px;")

        self.lens_name_display = QLabel("✨ 原生眼睛 (未佩戴)")
        self.lens_name_display.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        self.lens_name_display.setStyleSheet("color: #38BDF8; background: transparent;")

        info_layout.addWidget(tag_label, 1, Qt.AlignLeft)
        info_layout.addWidget(self.lens_name_display, 0, Qt.AlignRight)

        left_layout.addWidget(self.video_label, 1)
        left_layout.addWidget(self.info_card, 0)

        main_layout.addWidget(left_container, 65)

        self.control_panel = QWidget()
        self.control_panel.setFixedWidth(460)
        self.control_panel.setStyleSheet("""
            QWidget {
                background-color: rgba(15, 23, 42, 0.85);
                border-radius: 18px;
                border: 1px solid rgba(255, 255, 255, 0.15);
            }
        """)
        
        panel_shadow = QGraphicsDropShadowEffect()
        panel_shadow.setBlurRadius(20)
        panel_shadow.setColor(QColor(0, 0, 0, 160))
        panel_shadow.setOffset(0, 6)
        self.control_panel.setGraphicsEffect(panel_shadow)

        control_layout = QVBoxLayout(self.control_panel)
        control_layout.setContentsMargins(22, 20, 22, 20)
        
        self.title_label = QLabel("🐶 憨憨美瞳店")
        self.title_label.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        self.title_label.setStyleSheet("color: #FFFFFF; background: transparent;")
        control_layout.addWidget(self.title_label)

        self.sub_title = QLabel("AI Real-Time Virtual Fitting System")
        self.sub_title.setFont(QFont("Segoe UI", 9))
        self.sub_title.setStyleSheet("color: #818CF8; background: transparent; margin-bottom: 4px;")
        control_layout.addWidget(self.sub_title)

        combo_group = QGroupBox("挑选试戴款式")
        combo_box_layout = QVBoxLayout()
        combo_box_layout.setContentsMargins(12, 16, 12, 10)
        self.color_combo = QComboBox()
        self.color_combo.addItems(list(self.lens_presets.keys()))
        self.color_combo.setCurrentText(self.NONE_LENS)
        self.color_combo.currentTextChanged.connect(self.change_color)
        combo_box_layout.addWidget(self.color_combo)
        combo_group.setLayout(combo_box_layout)
        control_layout.addWidget(combo_group)

        param_group = QGroupBox("镜头参数微调")
        param_layout = QVBoxLayout()
        param_layout.setContentsMargins(12, 16, 12, 10)

        self.label_alpha = QLabel(f"显色融合度 (Alpha): {self.alpha:.2f}")
        self.label_alpha.setStyleSheet("background: transparent;")
        param_layout.addWidget(self.label_alpha)
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(20, 100)
        self.slider_alpha.setValue(int(self.alpha * 100))
        self.slider_alpha.valueChanged.connect(self.change_alpha)
        param_layout.addWidget(self.slider_alpha)

        param_layout.addSpacing(10)

        self.label_scale = QLabel(f"美瞳相对大小 (Scale): {self.size_scale:.2f}x")
        self.label_scale.setStyleSheet("background: transparent;")
        param_layout.addWidget(self.label_scale)
        self.slider_scale = QSlider(Qt.Horizontal)
        self.slider_scale.setRange(100, 160)
        self.slider_scale.setValue(int(self.size_scale * 100))
        self.slider_scale.valueChanged.connect(self.change_scale)
        param_layout.addWidget(self.slider_scale)

        param_group.setLayout(param_layout)
        control_layout.addWidget(param_group)

        welcome_box = QGroupBox("小店形象代言狗")
        welcome_layout = QHBoxLayout()
        welcome_layout.setContentsMargins(14, 18, 14, 14)
        welcome_layout.setSpacing(16)

        dog_avatar_label = QLabel()
        dog_pixmap = self.get_rounded_pixmap(self.hanhan2_image_path, target_size=110, radius=14)
        if not dog_pixmap.isNull():
            dog_avatar_label.setPixmap(dog_pixmap)
            dog_avatar_label.setStyleSheet("border: 2px solid rgba(129, 140, 248, 0.6); border-radius: 16px; background: transparent;")
        else:
            dog_avatar_label.setText("🐶")
            dog_avatar_label.setStyleSheet("font-size: 30px; background: transparent;")

        welcome_text_container = QWidget()
        welcome_text_container.setStyleSheet("background: transparent;")
        welcome_text_layout = QVBoxLayout(welcome_text_container)
        welcome_text_layout.setContentsMargins(0, 0, 0, 0)
        welcome_text_layout.setSpacing(6)
        welcome_text_layout.setAlignment(Qt.AlignVCenter)

        greeting_title = QLabel("欢迎光临~ 🐾")
        greeting_title.setFont(QFont("Microsoft YaHei", 15, QFont.Bold))
        greeting_title.setStyleSheet("color: #F472B6; background: transparent;")

        greeting_sub = QLabel("我是店长【憨憨】\n祝您挑选到心仪的美瞳！✨")
        greeting_sub.setFont(QFont("Microsoft YaHei", 10))
        greeting_sub.setStyleSheet("color: #CBD5E1; background: transparent; line-height: 1.4;")

        welcome_text_layout.addWidget(greeting_title)
        welcome_text_layout.addWidget(greeting_sub)

        welcome_layout.addWidget(dog_avatar_label)
        welcome_layout.addWidget(welcome_text_container, 1)

        welcome_box.setLayout(welcome_layout)
        control_layout.addWidget(welcome_box)

        control_layout.addStretch()

        self.btn_toggle = QPushButton("暂停实时预览")
        self.btn_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_toggle.clicked.connect(self.toggle_camera)
        control_layout.addWidget(self.btn_toggle)

        main_layout.addWidget(self.control_panel, 35)
        self.timer.start(16)

    def change_color(self, text):
        if text in self.lens_presets:
            self.current_preset_name = text
            self.current_preset = self.lens_presets[text]
            if text == self.NONE_LENS:
                self.lens_name_display.setText("✨ 原生眼睛 (未佩戴)")
                self.lens_name_display.setStyleSheet("color: #94A3B8; background: transparent;")
            else:
                self.lens_name_display.setText(text)
                self.lens_name_display.setStyleSheet("color: #38BDF8; background: transparent;")

    def change_alpha(self, value):
        self.alpha = value / 100.0
        self.label_alpha.setText(f"显色融合度 (Alpha): {self.alpha:.2f}")

    def change_scale(self, value):
        self.size_scale = value / 100.0
        self.label_scale.setText(f"美瞳相对大小 (Scale): {self.size_scale:.2f}x")

    def toggle_camera(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_toggle.setText("开启实时预览")
            self.btn_toggle.setStyleSheet("background-color: rgba(34, 197, 94, 0.9);")
        else:
            self.timer.start(16)
            self.btn_toggle.setText("暂停实时预览")
            self.btn_toggle.setStyleSheet("background-color: rgba(99, 102, 241, 0.9);")

    def render_eye_lens(self, frame, center, iris_radius, eyelid_points):
        cx, cy = center
        radius = int(iris_radius * self.size_scale)
        if radius <= 0:
            return

        h, w, _ = frame.shape
        preset_type = self.current_preset.get("type", "color")

        if preset_type == "image":
            tex = self.texture_cache.get(self.current_preset_name)
            if tex is None:
                return
            size = radius * 2
            if size <= 0:
                return
            lens_img = cv2.resize(tex, (size, size), interpolation=cv2.INTER_AREA)
            lens_center = radius
        else:
            size = radius * 2 + 10
            lens_center = size // 2
            lens_img = np.zeros((size, size, 4), dtype=np.uint8)

            cv2.circle(lens_img, (lens_center, lens_center), radius, (*self.current_preset["outer"], 255), -1)
            cv2.circle(lens_img, (lens_center, lens_center), int(radius * 0.78), (*self.current_preset["inner"], 255), -1)
            cv2.circle(lens_img, (lens_center, lens_center), int(radius * 0.35), (0, 0, 0, 0), -1, cv2.LINE_AA)
            
            hl_x, hl_y = int(lens_center - radius * 0.3), int(lens_center - radius * 0.3)
            cv2.circle(lens_img, (hl_x, hl_y), int(radius * 0.14), (255, 255, 255, 180), -1, cv2.LINE_AA)
            lens_img = cv2.GaussianBlur(lens_img, (5, 5), 0)

        y1, y2 = max(0, cy - lens_center), min(h, cy + lens_center)
        x1, x2 = max(0, cx - lens_center), min(w, cx + lens_center)
        ly1, ly2 = max(0, -(cy - lens_center)), size - max(0, (cy + lens_center) - h)
        lx1, lx2 = max(0, -(cx - lens_center)), size - max(0, (cx + lens_center) - w)

        if y1 >= y2 or x1 >= x2 or ly1 >= ly2 or lx1 >= lx2:
            return

        roi_frame = frame[y1:y2, x1:x2]
        roi_lens = lens_img[ly1:ly2, lx1:lx2]

        eyelid_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(eyelid_mask, [np.array(eyelid_points, dtype=np.int32)], 255)
        eyelid_mask_roi = eyelid_mask[y1:y2, x1:x2] / 255.0

        alpha_lens = (roi_lens[:, :, 3] / 255.0) * self.alpha * eyelid_mask_roi
        alpha_lens = np.expand_dims(alpha_lens, axis=-1)

        blended = alpha_lens * roi_lens[:, :, :3] + (1.0 - alpha_lens) * roi_frame
        frame[y1:y2, x1:x2] = blended.astype(np.uint8)

    def display_frame(self, frame):
        self.current_frame = frame
        self.update_video_pixmap()

    def update_video_pixmap(self):
        if self.current_frame is None:
            return
        lbl_w = self.video_label.width()
        lbl_h = self.video_label.height()
        if lbl_w <= 0 or lbl_h <= 0:
            return

        h, w, _ = self.current_frame.shape
        bytes_per_line = 3 * w
        contiguous_img_data = cv2.resize(self.current_frame, (w, h), interpolation=cv2.INTER_LINEAR).copy()
        
        q_img = QImage(contiguous_img_data.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        self.update_video_pixmap()
        super().resizeEvent(event)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if self.current_preset is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    pts = face_landmarks.landmark

                    lx, ly = int(pts[468].x * w), int(pts[468].y * h)
                    rx, ry = int(pts[473].x * w), int(pts[473].y * h)
                    
                    l_edge_x = int(pts[469].x * w)
                    r_edge_x = int(pts[474].x * w)
                    raw_l_r = max(8, int(np.hypot(lx - l_edge_x, ly - int(pts[469].y * h))))
                    raw_r_r = max(8, int(np.hypot(rx - r_edge_x, ry - int(pts[474].y * h))))

                    if self.prev_left is None:
                        self.prev_left, self.prev_right = (lx, ly), (rx, ry)
                        self.prev_l_radius, self.prev_r_radius = raw_l_r, raw_r_r
                    else:
                        lx = int(self.prev_left[0] * 0.5 + lx * 0.5)
                        ly = int(self.prev_left[1] * 0.5 + ly * 0.5)
                        rx = int(self.prev_right[0] * 0.5 + rx * 0.5)
                        ry = int(self.prev_right[1] * 0.5 + ry * 0.5)
                        self.prev_left, self.prev_right = (lx, ly), (rx, ry)

                        l_radius = int(self.prev_l_radius * 0.7 + raw_l_r * 0.3)
                        r_radius = int(self.prev_r_radius * 0.7 + raw_r_r * 0.3)
                        self.prev_l_radius, self.prev_r_radius = l_radius, r_radius

                    left_eyelid_idx = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                    right_eyelid_idx = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                    
                    left_eyelid_pts = [(int(pts[i].x * w), int(pts[i].y * h)) for i in left_eyelid_idx]
                    right_eyelid_pts = [(int(pts[i].x * w), int(pts[i].y * h)) for i in right_eyelid_idx]

                    self.render_eye_lens(frame, (lx, ly), self.prev_l_radius, left_eyelid_pts)
                    self.render_eye_lens(frame, (rx, ry), self.prev_r_radius, right_eyelid_pts)

        self.display_frame(frame)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HanHanContactLensApp()
    window.show()
    sys.exit(app.exec_())