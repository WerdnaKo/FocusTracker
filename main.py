import cv2
import mediapipe as mp

# Явно прописываем пути к модулям, чтобы обойти баг свежих версий MediaPipe
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

cap = cv2.VideoCapture(0)

# Запускаем трекер Face Mesh
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Включает детализированные точки зрачков и радужки
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Камера не найдена!")
            break

        # MediaPipe требует формат RGB, а OpenCV выдает BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        # Если лицо найдено, отрисовываем сетку
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Отрисовка базовой сетки лица
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                # Отрисовка контуров глаз и радужки
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

        cv2.imshow("Focus Tracker - Face Mesh", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Выход по клавише ESC
            break

cap.release()
cv2.destroyAllWindows()