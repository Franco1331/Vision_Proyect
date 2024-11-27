# Sign Language Recognition System 
This project implements a sign language recognition system that translates hand gestures from videos into text. It utilizes Mediapipe for hand keypoint detection and an Artificial Neural Network (ANN) for classification.

This project requires Python 3.8 or higher.

Ensure the training.json file contains the correct mapping between video IDs and corresponding glosses (words).

This script trains the model using the data in dataset.json and saves the trained model as model2.pth. It also displays training progress, including loss and evaluation metrics, and generates plots for visualization.

Inference: (Currently integrated with training) The main.py script currently performs inference on the training data after each training epoch.
# Setup

Para utilizar el repositorio es necesario abrir `Anaconda Prompt` e ir a una la direccion en la que se quiere instalar el repositorio, una vez ahí, se inicia con los siguientes pasos:

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/Franco1331/Vision_Proyect.git
   cd hand-vision
   ```

2. **Crear el entorno en Anaconda:**
   ```bash
   conda create --name hand-vision python=3.10
   conda activate hand-vision
   ```

3. **Instalar las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

   - Si hay problemas con `torch`, ejecutar:
     ```bash
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
     ```

# Training

1. **Preparar los datos:**
   - Crear la carpeta `training-videos` y cargar todos los videos ahí para el entrenamiento.
   - Ejecutar `frame_split.py` para extraer los cuadros de video a 25 fps:
     ```bash
     python preprocess/frame_split.py
     ```
   - Ejecutar `extract-points.py` para extraer los puntos clave (30 cuadros) que proporcionan la acción del gesto de la mano:
     ```bash
     python preprocess/extract-points.py
     ```
   - Recolectar las coordenadas de `mediapipe` de cada mano en un arreglo de puntos clave.
   - Compilar todos los puntos clave de todos los cuadros en un solo arreglo.
   - Seguir la estructura de `dataset.json` para la configuración.

3. **Crear el dataset:**
   - Ejecutar `dataset.py` para crear el dataset para entrenamiento:
     ```bash
     python dataset.py
     ```

3. **Entrenar el modelo:**
   - Ejecutar `main.py` para inferencia y entrenamiento:
     ```bash
     python main.py
     ```

# Running

Ejecutar camera.py para realizar las pruebas de funcionamiento en tiempo real de forma local
