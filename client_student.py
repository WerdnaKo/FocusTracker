import cv2
import socket
import json
import customtkinter as ctk
import mediapipe.python.solutions.face_mesh as mp_face_mesh
from PIL import Image

from focus_analyzer import FocusAnalyzer

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class StudentClient(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Ученик - Focus Client Pro")
        self.geometry("1100x680")

        self.analyzer = FocusAnalyzer()
        self.is_tracking = False
        self.cap = None
        self.client_socket = None
        self.student_name = ""

        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        self.build_ui()

    def build_ui(self):
        # === ЛЕВАЯ ЧАСТЬ (Чистое видео) ===
        self.video_frame = ctk.CTkFrame(self, width=800, height=600)
        self.video_frame.pack(side="left", padx=20, pady=20, fill="both", expand=True)

        self.video_label = ctk.CTkLabel(self.video_frame, text="Камера выключена\nВведите имя и нажмите «Подключиться»",
                                        font=("Arial", 16))
        self.video_label.pack(expand=True)

        # === ПРАВАЯ ЧАСТЬ (Панель) ===
        self.control_frame = ctk.CTkFrame(self, width=240)
        self.control_frame.pack(side="right", padx=20, pady=20, fill="y")

        ctk.CTkLabel(self.control_frame, text="🎓 Вход на урок", font=("Arial", 18, "bold")).pack(pady=15)

        self.name_entry = ctk.CTkEntry(self.control_frame, placeholder_text="Ваше Имя", width=200)
        self.name_entry.pack(pady=10)

        self.ip_entry = ctk.CTkEntry(self.control_frame, placeholder_text="IP Учителя", width=200)
        self.ip_entry.insert(0, "127.0.0.1")
        self.ip_entry.pack(pady=10)

        self.start_btn = ctk.CTkButton(self.control_frame, text="🚀 Подключиться",
                                       command=self.toggle_tracking, fg_color="#28a745", font=("Arial", 14, "bold"))
        self.start_btn.pack(pady=20, padx=10)

        self.calib_btn = ctk.CTkButton(self.control_frame, text="🎯 Сбросить калибровку",
                                       command=self.recalibrate, state="disabled", fg_color="#17a2b8")
        self.calib_btn.pack(pady=5)

        self.status_lbl = ctk.CTkLabel(self.control_frame, text="Статус: Ожидание...", font=("Arial", 13),
                                       text_color="gray")
        self.status_lbl.pack(pady=15)

        self.metrics_lbl = ctk.CTkLabel(self.control_frame, text="", font=("Arial", 12), justify="left")
        self.metrics_lbl.pack(pady=20)

    def recalibrate(self):
        if self.is_tracking:
            self.analyzer.start_recalibration()

    def connect_to_teacher(self, ip):
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.settimeout(2.0)
            self.client_socket.connect((ip, 5050))
            self.client_socket.settimeout(None)
            return True
        except Exception as e:
            print(f"Ошибка подключения: {e}")
            return False

    def toggle_tracking(self):
        if not self.is_tracking:
            self.student_name = self.name_entry.get()
            if not self.student_name:
                self.status_lbl.configure(text="Введите имя!", text_color="red")
                return

            ip = self.ip_entry.get()
            self.status_lbl.configure(text="Подключение...", text_color="yellow")
            self.update()

            if self.connect_to_teacher(ip):
                self.is_tracking = True
                self.analyzer.start_recalibration()
                self.cap = cv2.VideoCapture(0)
                self.start_btn.configure(text="⏹ Отключиться", fg_color="#dc3545")
                self.calib_btn.configure(state="normal")
                self.status_lbl.configure(text="В сети! Идет урок", text_color="green")
                self.name_entry.configure(state="disabled")
                self.ip_entry.configure(state="disabled")
                self.process_frame()
            else:
                self.status_lbl.configure(text="Ошибка связи с сервером", text_color="red")
        else:
            self.is_tracking = False
            self.start_btn.configure(text="🚀 Подключиться", fg_color="#28a745")
            self.calib_btn.configure(state="disabled")
            self.status_lbl.configure(text="Отключено", text_color="gray")
            self.metrics_lbl.configure(text="")
            self.name_entry.configure(state="normal")
            self.ip_entry.configure(state="normal")
            if self.cap:
                self.cap.release()
            if self.client_socket:
                self.client_socket.close()
            self.video_label.configure(image="", text="Камера выключена")

    def process_frame(self):
        if self.is_tracking and self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(frame_rgb)

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]

                    eye_analysis = self.analyzer.analyze_eyes(face_landmarks, w, h)
                    head_analysis = self.analyzer.analyze_head_pose(face_landmarks, w, h)

                    # Обновляем инфо в боковой панели (Оставляем только самое нужное)
                    calib_status = "Готово" if self.analyzer.is_calibrated else "В процессе..."
                    self.metrics_lbl.configure(text=
                                               f"Калибровка: {calib_status}\n\n"
                                               f"Ваш статус:\n{head_analysis['direction']}"
                                               )

                    # Отправляем на сервер
                    packet = {
                        "name": self.student_name,
                        "eye_status": eye_analysis['status'],
                        "head_direction": head_analysis['direction']
                    }
                    try:
                        json_data = json.dumps(packet) + "\n"
                        self.client_socket.sendall(json_data.encode('utf-8'))
                    except Exception as e:
                        print(f"Связь потеряна: {e}")
                        self.toggle_tracking()
                        return

                # Передаем ЧИСТЫЙ кадр в интерфейс (без всяких рисунков)
                img = Image.fromarray(frame_rgb)
                imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(800, 600))
                self.video_label.configure(image=imgtk, text="")
                self.video_label.image = imgtk

            self.after(25, self.process_frame)


if __name__ == "__main__":
    app = StudentClient()
    app.mainloop()