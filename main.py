import cv2
import mediapipe.python.solutions.face_mesh as mp_face_mesh

# Импортируем наш собственный класс из соседнего файла
from focus_analyzer import FocusAnalyzer


def main():
    # Инициализируем камеру и наш анализатор
    cap = cv2.VideoCapture(0)
    analyzer = FocusAnalyzer()

    # Запускаем MediaPipe без лишних модулей рисования
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Камера не найдена!")
                break

            # Зеркалим кадр для удобства и получаем его размеры
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Обрабатываем нейросетью
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                # Берем лицо (оно у нас одно, индекс 0)
                face_landmarks = results.multi_face_landmarks[0]

                # Отправляем точки в наш "мозг" для анализа
                analysis = analyzer.analyze_eyes(face_landmarks, w, h)

                # Достаем результаты
                ear_value = analysis["ear"]
                status_text = analysis["status"]
                color = analysis["color"]

                # Выводим данные на экран (HUD)
                cv2.putText(frame, f"EAR (Глаза): {ear_value:.2f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Статус: {status_text}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Для красоты: отрисуем контуры глаз (точки), чтобы видеть, что камера за ними следит
                for idx in analyzer.LEFT_EYE_INDICES + analyzer.RIGHT_EYE_INDICES:
                    pt = face_landmarks.landmark[idx]
                    x, y = int(pt.x * w), int(pt.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

            cv2.imshow("Focus Tracker MVP", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC для выхода
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()