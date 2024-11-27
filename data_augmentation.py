import cv2
import os

# Ruta de la carpeta principal
input_folder = "RENOMBRADOS"
output_folder_suffix = "_augmented"

# Función para cambiar brillo y contraste
def adjust_brightness_contrast(frame, alpha=1.2, beta=50):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

# Crear carpeta de salida con el sufijo "_augmented"
def create_output_folder_structure(input_folder, output_folder_suffix):
    for root, dirs, _ in os.walk(input_folder):
        # Filtrar carpetas con sufijo "_augmented"
        dirs[:] = [d for d in dirs if not d.endswith(output_folder_suffix)]
        for dir_name in dirs:
            new_dir = os.path.join(root, dir_name + output_folder_suffix)
            os.makedirs(new_dir, exist_ok=True)

# Aplicar transformaciones y guardar videos
def process_videos(input_folder, output_folder_suffix):
    for root, dirs, files in os.walk(input_folder):
        # Filtrar carpetas con sufijo "_augmented"
        dirs[:] = [d for d in dirs if not d.endswith(output_folder_suffix)]
        if not dirs:  # Solo procesar archivos dentro de carpetas con videos
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(root, file)
                    file_name, file_ext = os.path.splitext(file)
                    output_subfolder = root + output_folder_suffix

                    # Verificar si el video ya fue procesado
                    processed_files = [f for f in os.listdir(output_subfolder) if f.startswith(file_name)]
                    if processed_files:
                        print(f"El video ya fue procesado: {file}")
                        continue

                    # Ruta para el video con todas las transformaciones
                    combined_video = os.path.join(output_subfolder, f"{file_name}_combined{file_ext}")

                    # Crear VideoCapture
                    cap = cv2.VideoCapture(video_path)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                    # Configurar el escritor de video
                    out = cv2.VideoWriter(combined_video, fourcc, fps, (width, height))

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # 1. Reflejo horizontal
                        reflected_frame = cv2.flip(frame, 1)
                        
                        # 2. Cambio de brillo/contraste
                        bright_frame = adjust_brightness_contrast(reflected_frame, alpha=1.2, beta=50)

                        # 3. Escribir el cuadro modificado
                        out.write(bright_frame)

                    cap.release()
                    out.release()

                    print(f"Video combinado generado: {combined_video}")

# Crear estructura de carpetas de salida
create_output_folder_structure(input_folder, output_folder_suffix)

# Procesar videos y aplicar transformaciones
process_videos(input_folder, output_folder_suffix)

print("Data augmentation completado.")