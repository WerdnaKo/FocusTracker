import cv2
import customtkinter as ctk
from PIL import Image
import mediapipe.python.solutions.face_mesh as mp_face_mesh

# Импортируем наши готовые "мозги"
from focus_analyzer import FocusAnalyzer
from session_logger import SessionLogger

# Настройка внешнего вида окна (Темная тема, синие акценты)
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class FocusTrackerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Focus Tracker Pro (HR-Tech Edition)")
        self.geometry("900x550")
        self.resizable(False, False)

        # Переменные состояния
        self.is_tracking = False
        self.cap = None
        self.analyzer = FocusAnalyzer()
        self.logger = None

        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.build_ui()

    def build_ui(self):
        # === ЛЕВАЯ ЧАСТЬ (Видео с камеры) ===
        self.video_frame = ctk.CTkFrame(self, width=640, height=480)
        self.video_frame.pack(side="left", padx=20, pady=20, fill="both", expand=True)

        self.video_label = ctk.CTkLabel(self.video_frame, text="Камера выключена\nНажмите «Запустить»",
                                        font=("Arial", 18))
        self.video_label.pack(expand=True)

        # === ПРАВАЯ ЧАСТЬ (Панель управления) ===
        self.control_frame = ctk.CTkFrame(self, width=220)
        self.control_frame.pack(side="right", padx=20, pady=20, fill="y")

        # Кнопка Старт/Стоп
        self.start_btn = ctk.CTkButton(self.control_frame, text="▶ Запустить трекер",
                                       command=self.toggle_tracking,
                                       font=("Arial", 14, "bold"), height=40,
                                       fg_color="#28a745", hover_color="#218838")
        self.start_btn.pack(pady=20, padx=20)

        # Главный статус
        self.status_label = ctk.CTkLabel(self.control_frame, text="Статус: Ожидание", font=("Arial", 18, "bold"))
        self.status_label.pack(pady=20)

        # Подробная аналитика
        self.ear_label = ctk.CTkLabel(self.control_frame, text="Глаза: --", font=("Arial", 14))
        self.ear_label.pack(pady=10)

        self.head_label = ctk.CTkLabel(self.control_frame, text="Голова: --", font=("Arial", 14))
        self.head_label.pack(pady=10)

    def toggle_tracking(self):
        """Включает или выключает камеру и логирование по кнопке"""
        if not self.is_tracking:
            # ЗАПУСК
            self.cap = cv2.VideoCapture(0)
            self.logger = SessionLogger()
            self.is_tracking = True
            self.start_btn.configure(text="⏹ Остановить", fg_color="#dc3545", hover_color="#c82333")
            self.update_frame()  # Запускаем цикл кадров
        else:
            # ОСТАНОВКА
            self.is_tracking = False
            self.start_btn.configure(text="▶ Запустить трекер", fg_color="#28a745", hover_color="#218838")
            if self.cap:
                self.cap.release()
            self.video_label.configure(image="", text="Камера выключена")
            self.status_label.configure(text="Статус: Остановлено", text_color="white")
            self.ear_label.configure(text="Глаза: --")
            self.head_label.configure(text="Голова: --")

            # Сохраняем отчет за сессию
            if self.logger:
                self.logger.generate_report()

    def update_frame(self):
        """Эта функция крутится по кругу и обновляет картинку в интерфейсе"""
        if self.is_tracking and self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                # Обработка нейросетью
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(frame_rgb)

                status_text = "Лицо не найдено"
                status_color = "gray"

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]

                    # Отправляем точки в наш мозг (FocusAnalyzer)
                    eye_analysis = self.analyzer.analyze_eyes(face_landmarks, w, h)
                    head_analysis = self.analyzer.analyze_head_pose(face_landmarks, w, h)

                    # Отправляем результаты в Логгер
                    self.logger.log_frame(eye_analysis['status'], head_analysis['direction'])

                    # Обновляем текст в боковом меню
                    self.ear_label.configure(text=f"Глаза: {eye_analysis['status']}")
                    self.head_label.configure(text=f"Голова: {head_analysis['direction']}")

                    # Логика для главного слова статуса
                    if "УСТАЛОСТЬ" in eye_analysis['status'] or "Моргание" in eye_analysis['status']:
                        status_text = "УСТАЛОСТЬ!"
                        status_color = "#dc3545"  # Красный
                    elif "Отвлекся" in head_analysis['direction'] or "Опустил" in head_analysis['direction']:
                        status_text = "ОТВЛЕКСЯ"
                        status_color = "#ffc107"  # Желтый
                    else:
                        status_text = "В ФОКУСЕ"
                        status_color = "#28a745"  # Зеленый

                    self.status_label.configure(text=f"Статус: {status_text}", text_color=status_color)

                    # Рисуем контуры глаз (чтобы было видно работу алгоритма)
                    for idx in self.analyzer.LEFT_EYE_INDICES + self.analyzer.RIGHT_EYE_INDICES:
                        pt = face_landmarks.landmark[idx]
                        cv2.circle(frame_rgb, (int(pt.x * w), int(pt.y * h)), 2, (0, 255, 255), -1)

                # Превращаем кадр OpenCV в формат, понятный интерфейсу CustomTkinter
                img = Image.fromarray(frame_rgb)
                imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))

                self.video_label.configure(image=imgtk, text="")
                self.video_label.image = imgtk

            # Запускаем эту же функцию снова через 15 миллисекунд
            self.after(15, self.update_frame)


if __name__ == "__main__":
    app = FocusTrackerApp()
    app.mainloop()