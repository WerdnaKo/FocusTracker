import math
import cv2
import numpy as np


class FocusAnalyzer:
    def __init__(self):
        # --- Настройки EAR (Усталость) ---
        self.EAR_THRESHOLD = 0.18  # Дефолт до калибровки
        self.FATIGUE_FRAMES_LIMIT = 15  # УМЕНЬШИЛИ! Теперь быстрее ловит засыпание
        self.closed_frames_counter = 0
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

        # --- Настройки Head Pose (Положение головы) ---
        self.HEAD_POSE_INDICES = [1, 152, 33, 263, 61, 291]
        self.face_3d = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 330.0, -65.0],
            [-225.0, -170.0, -135.0],
            [225.0, -170.0, -135.0],
            [-150.0, 150.0, -125.0],
            [150.0, 150.0, -125.0]
        ], dtype=np.float64)

        self.pitch_history = []
        self.yaw_history = []
        self.SMOOTH_FRAMES = 25

        self.is_calibrated = False
        self.calibration_frames = 0
        self.CALIBRATION_LIMIT = 30

        self.calib_pitch_sum = 0
        self.calib_yaw_sum = 0
        self.offset_pitch = 0.0
        self.offset_yaw = 0.0

        self.calib_ear_sum = 0
        self.natural_ear = 0.25

        # НОВОЕ: Храним последние углы головы для защиты глаз от искажений
        self.last_yaw = 0.0
        self.last_pitch = 0.0

    def start_recalibration(self):
        self.is_calibrated = False
        self.calibration_frames = 0
        self.calib_pitch_sum = 0
        self.calib_yaw_sum = 0
        self.calib_ear_sum = 0
        self.closed_frames_counter = 0
        print("[КАЛИБРОВКА] Начался сбор данных...")

    def _euclidean_distance(self, p1, p2):
        return math.dist(p1, p2)

    def _calculate_ear(self, eye_points):
        v1 = self._euclidean_distance(eye_points[1], eye_points[5])
        v2 = self._euclidean_distance(eye_points[2], eye_points[4])
        h = self._euclidean_distance(eye_points[0], eye_points[3])
        if h == 0: return 0.0
        return (v1 + v2) / (2.0 * h)

    def analyze_eyes(self, landmarks, frame_width, frame_height):
        def get_coords(indices):
            return [(int(landmarks.landmark[i].x * frame_width),
                     int(landmarks.landmark[i].y * frame_height)) for i in indices]

        left_eye = get_coords(self.LEFT_EYE_INDICES)
        right_eye = get_coords(self.RIGHT_EYE_INDICES)
        avg_ear = (self._calculate_ear(left_eye) + self._calculate_ear(right_eye)) / 2.0

        if not self.is_calibrated:
            self.calib_ear_sum += avg_ear
            return {"ear": avg_ear, "status": "Калибровка глаз...", "color": (255, 255, 0)}

        # ИСПРАВЛЕНИЕ 1: Блокируем ложные моргания при повороте головы!
        # Если голова повернута больше чем на 18 градусов, EAR искажается.
        if abs(self.last_yaw) > 18 or abs(self.last_pitch) > 18:
            self.closed_frames_counter = 0  # Сбрасываем счетчик усталости
            return {"ear": avg_ear, "status": "Не видно (Поворот)", "color": (128, 128, 128)}

        status, color = "В фокусе", (0, 255, 0)

        if avg_ear < self.EAR_THRESHOLD:
            self.closed_frames_counter += 1
            status, color = "Моргание", (0, 255, 255)
            if self.closed_frames_counter >= self.FATIGUE_FRAMES_LIMIT:
                status, color = "УСТАЛОСТЬ / СОН!", (0, 0, 255)
        else:
            self.closed_frames_counter = 0

        return {"ear": avg_ear, "status": status, "color": color}

    def analyze_head_pose(self, landmarks, frame_width, frame_height):
        face_2d = []
        for idx in self.HEAD_POSE_INDICES:
            pt = landmarks.landmark[idx]
            x, y = int(pt.x * frame_width), int(pt.y * frame_height)
            face_2d.append([x, y])

        face_2d = np.array(face_2d, dtype=np.float64)

        focal_length = 1.0 * frame_width
        cam_matrix = np.array([
            [focal_length, 0, frame_width / 2],
            [0, focal_length, frame_height / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(self.face_3d, face_2d, cam_matrix, dist_matrix)
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        raw_pitch = angles[0]
        raw_yaw = angles[1]

        if not self.is_calibrated:
            self.calibration_frames += 1
            self.calib_pitch_sum += raw_pitch
            self.calib_yaw_sum += raw_yaw

            if self.calibration_frames >= self.CALIBRATION_LIMIT:
                self.offset_pitch = self.calib_pitch_sum / self.CALIBRATION_LIMIT
                self.offset_yaw = self.calib_yaw_sum / self.CALIBRATION_LIMIT
                self.natural_ear = self.calib_ear_sum / self.CALIBRATION_LIMIT

                # ИСПРАВЛЕНИЕ 2: Сделали алгоритм более чувствительным к морганиям
                # Теперь порог = 80% от нормального открытого глаза (было 65%)
                # И поставили жесткий минимум 0.16, чтобы не проваливался.
                self.EAR_THRESHOLD = max(0.16, min(self.natural_ear * 0.80, 0.23))

                self.is_calibrated = True

            return {
                "pitch": 0.0, "yaw": 0.0,
                "direction": "Калибровка...", "color": (255, 255, 0),
                "nose_2d": face_2d[0], "rot_vec": rot_vec, "trans_vec": trans_vec, "cam_matrix": cam_matrix,
                "dist_matrix": dist_matrix
            }

        calibrated_pitch = raw_pitch - self.offset_pitch
        calibrated_yaw = raw_yaw - self.offset_yaw

        self.pitch_history.append(calibrated_pitch)
        self.yaw_history.append(calibrated_yaw)

        if len(self.pitch_history) > self.SMOOTH_FRAMES:
            self.pitch_history.pop(0)
        if len(self.yaw_history) > self.SMOOTH_FRAMES:
            self.yaw_history.pop(0)

        pitch = sum(self.pitch_history) / len(self.pitch_history)
        yaw = sum(self.yaw_history) / len(self.yaw_history)

        # НОВОЕ: Сохраняем углы для следующего кадра (чтобы функция глаз знала о повороте)
        self.last_pitch = pitch
        self.last_yaw = yaw

        direction = "В фокусе (Все четко!)"
        color = (0, 255, 0)

        YAW_THRESHOLD = 20
        PITCH_DOWN = -14
        PITCH_UP = 18

        if yaw < -YAW_THRESHOLD:
            direction = "Отвлекся влево"
            color = (0, 0, 255)
        elif yaw > YAW_THRESHOLD:
            direction = "Отвлекся вправо"
            color = (0, 0, 255)
        elif pitch < PITCH_DOWN:
            direction = "Опустил голову (Телефон?)"
            color = (0, 255, 255)
        elif pitch > PITCH_UP:
            direction = "Смотрит вверх"
            color = (0, 0, 255)

        return {
            "pitch": pitch,
            "yaw": yaw,
            "direction": direction,
            "color": color,
            "nose_2d": face_2d[0],
            "rot_vec": rot_vec,
            "trans_vec": trans_vec,
            "cam_matrix": cam_matrix,
            "dist_matrix": dist_matrix
        }