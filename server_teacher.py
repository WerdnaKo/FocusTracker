import socket
import threading
import json
import customtkinter as ctk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class TeacherDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Панель учителя (EdTech Focus Server)")
        self.geometry("800x600")

        # База данных текущих учеников: { "Имя": {"status": "В ФОКУСЕ", "color": "green"} }
        self.students = {}
        self.server_socket = None
        self.is_running = True

        self.build_ui()
        self.start_server()

    def build_ui(self):
        self.header = ctk.CTkLabel(self, text="👨‍🏫 Мониторинг класса", font=("Arial", 24, "bold"))
        self.header.pack(pady=20)

        # Контейнер для карточек учеников (Scrollable на случай большого класса)
        self.students_frame = ctk.CTkScrollableFrame(self, width=700, height=450)
        self.students_frame.pack(pady=10, fill="both", expand=True)

        # Виджеты карточек сохраняем сюда, чтобы обновлять их, а не перерисовывать заново
        self.student_widgets = {}

        # Запускаем цикл обновления интерфейса (2 раза в секунду)
        self.update_dashboard()

    def start_server(self):
        """Запуск TCP-сервера в отдельном потоке (чтобы не вешать интерфейс)"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(("0.0.0.0", 5050))  # Слушаем все подключения на порту 5050
        self.server_socket.listen(30)  # Максимум 30 учеников

        # Фоновый поток для приема новых учеников
        threading.Thread(target=self.accept_clients, daemon=True).start()
        print("[СЕРВЕР] Запущен. Ожидание подключений на порту 5050...")

    def accept_clients(self):
        while self.is_running:
            try:
                conn, addr = self.server_socket.accept()
                threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True).start()
            except:
                break

    def handle_client(self, conn, addr):
        """Обрабатывает входящие данные от конкретного ученика"""
        with conn:
            while self.is_running:
                try:
                    data = conn.recv(1024)
                    if not data:
                        break

                    # Расшифровываем JSON от ученика
                    packet = json.loads(data.decode('utf-8'))
                    name = packet.get("name", "Неизвестный")

                    # Обновляем базу данных сервера
                    self.students[name] = {
                        "status": packet.get("eye_status", "Нет данных"),
                        "direction": packet.get("head_direction", "Нет данных"),
                        "color": packet.get("color", "gray")
                    }
                except (ConnectionResetError, json.JSONDecodeError):
                    break

    def update_dashboard(self):
        """Обновляет интерфейс на основе данных из self.students"""
        for name, data in self.students.items():
            if name not in self.student_widgets:
                # Если ученик новый - создаем для него карточку
                card = ctk.CTkFrame(self.students_frame, corner_radius=10, border_width=2)
                card.pack(fill="x", pady=5, padx=10)

                name_lbl = ctk.CTkLabel(card, text=f"👤 {name}", font=("Arial", 16, "bold"))
                name_lbl.pack(side="left", padx=20, pady=10)

                status_lbl = ctk.CTkLabel(card, text="Подключение...", font=("Arial", 14))
                status_lbl.pack(side="right", padx=20, pady=10)

                self.student_widgets[name] = {"frame": card, "status_lbl": status_lbl}

            # Обновляем текст и цвет карточки
            widget = self.student_widgets[name]

            # Логика цветов (конвертируем из названий в hex)
            border_color = "#28a745"  # Зеленый по умолчанию
            text = f"{data['status']} | {data['direction']}"

            if "УСТАЛОСТЬ" in data['status']:
                border_color = "#dc3545"  # Красный
            elif "Отвлекся" in data['direction'] or "Опустил" in data['direction']:
                border_color = "#ffc107"  # Желтый

            widget["frame"].configure(border_color=border_color)
            widget["status_lbl"].configure(text=text, text_color=border_color)

        # Перезапуск функции через 500 мс (0.5 сек)
        self.after(500, self.update_dashboard)


if __name__ == "__main__":
    app = TeacherDashboard()
    app.mainloop()