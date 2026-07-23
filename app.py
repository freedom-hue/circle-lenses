import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QPushButton, QVBoxLayout, QHBoxLayout, QSlider, QComboBox)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

# 解决 0.10+ 版本 solutions 被隐藏的终极兼容写法
try:
    mp_face_mesh = mp.solutions.face_mesh
except AttributeError:
    # 新版本手动触发内部模块加载
    import mediapipe.python.solutions.face_mesh as mp_face_mesh

class ContactLensApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("虚拟美瞳试戴系统 v1.0")
        self.setGeometry(100, 100, 900, 650)

        # 初始化 FaceMesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # 开启虹膜定位
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 视频捕获与定时器
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 默认美瞳参数
        self.lens_color = (255, 105, 180)  # BGR 格式：粉紫色
        self.alpha = 0.5                     # 融合透明度

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 视频显示区域
        self.video_label = QLabel("正在启动摄像头...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000; color: #fff; border-radius: 8px; font-size: 14px;")
        self.video_label.setFixedSize(640, 480)
        main_layout.addWidget(self.video_label)

        # 右侧控制面板
        control_layout = QVBoxLayout()
        
        # 控件：颜色选择
        control_layout.addWidget(QLabel("<b>选择美瞳颜色/样式:</b>"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(["魅惑粉紫", "森林墨绿", "深邃宝石蓝", "自然棕褐"])
        self.color_combo.currentIndexChanged.connect(self.change_color)
        control_layout.addWidget(self.color_combo)

        control_layout.addSpacing(15)

        # 控件：透明度调节
        control_layout.addWidget(QLabel("<b>美瞳融合透明度:</b>"))
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(1, 10)
        self.slider_alpha.setValue(5)
        self.slider_alpha.valueChanged.connect(self.change_alpha)
        control_layout.addWidget(self.slider_alpha)

        control_layout.addSpacing(20)

        # 开关按钮
        self.btn_toggle = QPushButton("暂停摄像头")
        self.btn_toggle.setStyleSheet("padding: 10px; background-color: #2563eb; color: white; font-weight: bold; border-radius: 5px;")
        self.btn_toggle.clicked.connect(self.toggle_camera)
        control_layout.addWidget(self.btn_toggle)

        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        # 启动定时器刷新画面 (30 FPS)
        self.timer.start(33)

    def change_color(self, index):
        colors = [
            (255, 105, 180),  # 粉紫
            (34, 139, 34),    # 墨绿
            (235, 120, 30),   # 宝石蓝
            (19, 69, 139)     # 棕褐
        ]
        self.lens_color = colors[index]

    def change_alpha(self, value):
        self.alpha = value / 10.0

    def toggle_camera(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_toggle.setText("开启摄像头")
        else:
            self.timer.start(33)
            self.btn_toggle.setText("暂停摄像头")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)  # 镜像翻转
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 468 与 473 分别为左、右虹膜中心点
                lx, ly = int(face_landmarks.landmark[468].x * w), int(face_landmarks.landmark[468].y * h)
                rx, ry = int(face_landmarks.landmark[473].x * w), int(face_landmarks.landmark[473].y * h)

                # 绘制融合美瞳效果
                overlay = frame.copy()
                cv2.circle(overlay, (lx, ly), 14, self.lens_color, -1)
                cv2.circle(overlay, (rx, ry), 14, self.lens_color, -1)
                cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)

        # 转换为 QImage 并在 PyQt5 界面上渲染
        bytes_per_line = 3 * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ContactLensApp()
    window.show()
    sys.exit(app.exec_())