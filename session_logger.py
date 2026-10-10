import time
import csv
from datetime import datetime


class SessionLogger:
    """Класс для сбора статистики и генерации отчетов по сессии"""

    def __init__(self):
        self.start_time = time.time()
        self.total_frames = 0

        # Словарик для хранения нашей статистики
        self.stats = {
            "В фокусе": 0,
            "Отвлекся": 0,
            "Усталость": 0
        }
        # Запоминаем время старта для названия файла
        self.session_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def log_frame(self, eye_status: str, head_status: str):
        """Вызывается каждый кадр и обновляет статистику"""
        self.total_frames += 1

        # Приоритет состояний: если засыпает - это важнее, чем если просто отвлекся
        if "УСТАЛОСТЬ" in eye_status or "Моргание" in eye_status:
            self.stats["Усталость"] += 1
        elif "Отвлекся" in head_status or "Опустил" in head_status or "вверх" in head_status:
            self.stats["Отвлекся"] += 1
        else:
            self.stats["В фокусе"] += 1

    def generate_report(self):
        """Вызывается при закрытии программы. Считает проценты и сохраняет лог."""
        duration = int(time.time() - self.start_time)

        if self.total_frames == 0:
            print("\n[!] Сессия была слишком короткой, нет данных для отчета.")
            return

        # Считаем проценты
        focus_pct = (self.stats["В фокусе"] / self.total_frames) * 100
        distract_pct = (self.stats["Отвлекся"] / self.total_frames) * 100
        fatigue_pct = (self.stats["Усталость"] / self.total_frames) * 100

        # 1. Печатаем красивый вывод в консоль
        print("\n" + "=" * 45)
        print(" 📊 ИТОГОВЫЙ ОТЧЕТ О ПРОДУКТИВНОСТИ")
        print("=" * 45)
        print(f" ⏱  Общее время сессии: {duration} секунд")
        print(f" 🎯 В фокусе:            {focus_pct:.1f}%")
        print(f" 📱 Отвлекался (телефон/стороны): {distract_pct:.1f}%")
        print(f" 😴 Признаки усталости:  {fatigue_pct:.1f}%")
        print("=" * 45)

        # 2. Сохраняем данные в файл .csv
        filename = f"session_{self.session_date}.csv"
        try:
            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Метрика", "Значение"])
                writer.writerow(["Длительность (сек)", duration])
                writer.writerow(["В фокусе (%)", round(focus_pct, 1)])
                writer.writerow(["Отвлекся (%)", round(distract_pct, 1)])
                writer.writerow(["Усталость (%)", round(fatigue_pct, 1)])
            print(f" 💾 Подробный лог сохранен в файл: {filename}\n")
        except Exception as e:
            print(f" [!] Ошибка при сохранении отчета: {e}")