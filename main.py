import cv2
import mediapipe.python.solutions.face_mesh as mp_face_mesh
from focus_analyzer import FocusAnalyzer


def main():
    cap = cv2.VideoCapture(0)
    analyzer = FocusAnalyzer()

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                # --- Анализ усталости (Глаза) ---
                eye_analysis = analyzer.analyze_eyes(face_landmarks, w, h)
                cv2.putText(frame, f"Статус: {eye_analysis['status']}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, eye_analysis['color'], 2)

                # --- Анализ фокуса (Поворот головы) ---
                head_analysis = analyzer.analyze_head_pose(face_landmarks, w, h)
                cv2.putText(frame, f"Фокус: {head_analysis['direction']}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, head_analysis['color'], 2)

                # Выводим точные градусы для отладки и демонстрации жюри
                cv2.putText(frame, f"Pitch (В/Н): {int(head_analysis['pitch'])}", (w - 200, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Yaw (Л/П): {int(head_analysis['yaw'])}", (w - 200, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Focus Tracker MVP", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()