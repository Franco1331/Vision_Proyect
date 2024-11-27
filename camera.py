import cv2
import mediapipe as mp
import torch
import numpy as np
from models import ANN
import json
import time

MODEL_PATH = "modely.pth"

# Cargar etiquetas desde el dataset.json
def load_labels(json_path="dataset.json"):
    with open(json_path) as file:
        data = json.load(file)
    labels = list(set([sample["word"] for sample in data]))
    return labels

labels = load_labels()

# Cargar el modelo
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = ANN(input_nodes=30 * 42 * 3, features=len(labels))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Inicializar Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Función para extraer puntos clave de una imagen
def extract_hand_keypoints(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z])
        return keypoints, results.multi_hand_landmarks
    return keypoints, None

# Preprocesar los puntos clave
def pad_keypoints(keypoints, target_length=42):
    padded_keypoints = np.zeros((target_length, 3))
    keypoints = np.array(keypoints)
    padded_keypoints[:min(len(keypoints), target_length), :] = keypoints[:target_length, :]
    return padded_keypoints

# Configurar dimensiones de la cámara
frame_width, frame_height = 640, 480

# Inicializar la cámara
cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)

frame_buffer = []
last_inference_word = None
collecting_frames = False  # Estado para determinar si se está recolectando
start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el video.")
        break

    # Invertir la imagen 
    frame = cv2.flip(frame, 1)

    # Botón para iniciar el proceso
    cv2.rectangle(frame, (50, 400), (200, 450), (255, 0, 0), -1)
    cv2.putText(frame, "START", (80, 435), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Verificar si se hace clic en el botón
    def is_button_clicked(x, y):
        return 50 <= x <= 200 and 400 <= y <= 450

    # Manejar clic del ratón
    def mouse_callback(event, x, y, flags, param):
        global collecting_frames, start_time
        if event == cv2.EVENT_LBUTTONDOWN:
            if is_button_clicked(x, y):
                collecting_frames = True
                start_time = time.time()  # Registrar el inicio del proceso

    cv2.setMouseCallback("Sign Language Detection", mouse_callback)

    if collecting_frames:
        elapsed_time = time.time() - start_time

        # Mostrar cuenta regresiva antes de iniciar la recolección
        if elapsed_time < 3:
            countdown = int(3 - elapsed_time)
            cv2.putText(frame, f"Starting in: {countdown}", (250, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Cambiar el indicador a recolectando cuadros
            indicator_color = (0, 0, 255)  # Rojo: recolectando cuadros

            # Extraer puntos clave de las manos
            keypoints, landmarks = extract_hand_keypoints(frame)

            if keypoints:
                # Dibujar los keypoints en la pantalla
                for hand_landmarks in landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                padded_keypoints = pad_keypoints(keypoints)
            else:
                # Rellenar con ceros si no hay datos previos
                padded_keypoints = np.zeros((42, 3))
            
            frame_buffer.append(padded_keypoints)

            # Dibujar progreso del buffer
            progress = int((len(frame_buffer) / 30) * 300)  # 300 px barra de progreso
            cv2.rectangle(frame, (50, 450), (50 + progress, 470), (0, 255, 0), -1)
            cv2.rectangle(frame, (50, 450), (350, 470), (255, 255, 255), 2)  # Marco de la barra

            # Mantener el tamaño máximo del buffer
            if len(frame_buffer) > 30:
                frame_buffer.pop(0)

            # Verificar si el buffer está lleno
            if len(frame_buffer) == 30:
                # Cambiar el color del indicador a verde (listo para inferencia)
                indicator_color = (0, 255, 0)

                # Convertir el buffer en tensor
                input_tensor = torch.FloatTensor(frame_buffer).to(device).flatten().unsqueeze(0)

                # Realizar predicción
                with torch.no_grad():
                    output = model(input_tensor)
                    predicted_label = torch.argmax(output, dim=1).item()
                    last_inference_word = labels[predicted_label]

                # Reiniciar el buffer para recolectar nuevos cuadros
                frame_buffer = []
                collecting_frames = False  # Reiniciar el proceso

            # Dibujar el indicador visual en la parte superior izquierda
            cv2.circle(frame, (30, 30), 20, indicator_color, -1)

    # Mostrar la última palabra detectada
    if last_inference_word:
        cv2.putText(frame, f"Detected: {last_inference_word}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar el video en tiempo real con indicador visual y botón
    cv2.imshow("Sign Language Detection", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
