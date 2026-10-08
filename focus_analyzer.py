import math


class FocusAnalyzer:
    """
    Класс для анализа состояния пользователя (усталость, фокус).
    """

    def __init__(self):
        # Порог EAR: если значение ниже, считаем глаз закрытым
        self.EAR_THRESHOLD = 0.20
        # Сколько кадров подряд глаза должны быть закрыты для фиксации усталости/сна
        self.FATIGUE_FRAMES_LIMIT = 20

        # Внутренний счетчик кадров с закрытыми глазами
        self.closed_frames_counter = 0

        # Индексы 6 точек для каждого глаза из MediaPipe Face Mesh
        # Порядок точек: левый край, верх-лево, верх-право, правый край, низ-право, низ-лево
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    def _euclidean_distance(self, p1: tuple, p2: tuple) -> float:
        """Считает расстояние между двумя 2D точками."""
        return math.dist(p1, p2)

    def _calculate_ear(self, eye_points: list) -> float:
        """
        Вычисляет Eye Aspect Ratio (соотношение сторон глаза).
        eye_points - список из 6 координат (x, y)
        """
        # Вертикальные расстояния
        v1 = self._euclidean_distance(eye_points[1], eye_points[5])
        v2 = self._euclidean_distance(eye_points[2], eye_points[4])

        # Горизонтальное расстояние
        h = self._euclidean_distance(eye_points[0], eye_points[3])

        # Защита от деления на ноль
        if h == 0:
            return 0.0

        # Формула EAR
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def analyze_eyes(self, landmarks, frame_width: int, frame_height: int) -> dict:
        """
        Главный метод. Принимает сырые точки от MediaPipe, возвращает статус.
        """

        # Функция для превращения нормализованных координат в пиксели
        def get_coords(indices):
            return [(int(landmarks.landmark[i].x * frame_width),
                     int(landmarks.landmark[i].y * frame_height)) for i in indices]

        left_eye_points = get_coords(self.LEFT_EYE_INDICES)
        right_eye_points = get_coords(self.RIGHT_EYE_INDICES)

        left_ear = self._calculate_ear(left_eye_points)
        right_ear = self._calculate_ear(right_eye_points)

        # Берем среднее значение обоих глаз
        avg_ear = (left_ear + right_ear) / 2.0

        # Бизнес-логика: определяем статус
        status = "В фокусе"
        color = (0, 255, 0)  # Зеленый

        if avg_ear < self.EAR_THRESHOLD:
            self.closed_frames_counter += 1
            status = "Моргание"
            color = (0, 255, 255)  # Желтый

            # Если глаза закрыты дольше лимита - бьем тревогу
            if self.closed_frames_counter >= self.FATIGUE_FRAMES_LIMIT:
                status = "УСТАЛОСТЬ / СОН!"
                color = (0, 0, 255)  # Красный
        else:
            # Глаза открыты - сбрасываем счетчик
            self.closed_frames_counter = 0

        # Возвращаем словарь с результатами
        return {
            "ear": avg_ear,
            "status": status,
            "color": color
        }