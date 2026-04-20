### *dataScienceCrimePredictor*

> "Tu cara es tu destiny. Tus píxeles, tu sentencia."

Bienvenido a **dSCP (LombrosAI)**, una herramienta de análisis biométrico de vanguardia diseñada para identificar la "tendencia criminal" a través de rasgos óseos y faciales. Digitalizando la antropología criminal del siglo XIX con Deep Learning del siglo XXI, proporcionamos una "Puntuación de Probabilidad de Riesgo" para cualquier individuo con rostro.

---

## El Problema

El objetivo es diseñar una herramienta capaz de procesar fotografías faciales para clasificar a los individuos y devolver un porcentaje de **"probabilidad de tendencia criminal"**.

Utilizando **Biometría Forense** aplicada a la **Seguridad** y la **Psicología de la Percepción**, el sistema actúa como un recomendador que asocia rasgos biométricos específicos con etiquetas de conducta extraídas de bases de datos históricas.

## El Algoritmo

Implementamos una arquitectura de **Deep Learning** basada en **Redes Neuronales Convolucionales (CNN)**:

- **Framework:** TensorFlow / PyTorch (Python).
- **Optimización:** Búsqueda bayesiana de hiperparámetros mediante **Optuna** (ajustando learning rate, tamaño de filtros y dropout).
- **Justificación:** Las CNN son superiores en la extracción de características jerárquicas y patrones espaciales complejos en datos de imágenes no estructurados.

## Los Datasets (Los Buenos vs. Los Malos)

Para entrenar a nuestro juez electrónico, utilizamos dos fuentes equilibradas:

1. **Perfil Estándar:** Extraído de [Labeled Faces in the Wild (LFW)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset).
2. **Perfil de Riesgo:** Extraído del [Departamento Correccional de Illinois (Illinois DOC)](https://www.kaggle.com/datasets/davidjfisher/illinois-doc-labeled-faces-dataset).

## El Proceso (Pipeline de Ejecución)

El sistema automatiza todo el ciclo de vida del dato, desde la descarga hasta la inferencia explicable:

1.  **Verificación e Ingesta:** Comprobación de integridad de los datasets (Illinois y LFW) mediante MD5.
2.  **Análisis Exploratorio de Datos (EDA) Avanzado:**
    *   **Calidad de Imagen:** Distribución de brillo y contraste (espacio HSV).
    *   **Demografía y Etnia:** Análisis de raza (usando **DeepFace** para LFW) y edad/sexo (Illinois).
    *   **Análisis de Emociones:** Clasificación de expresiones predominantes en los rostros.
    *   **Cara Promedio:** Generación visual del "rostro promedio" de cada grupo.
3.  **Preprocesamiento con MTCNN:** Detección y extracción quirúrgica de rostros, eliminando el ruido del fondo.
4.  **Entrenamiento Comparativo:**
    *   Entrenamiento de **ResNet50** y **VGG-16** en paralelo.
    *   Balanceo de clases automático para evitar sesgos algorítmicos.
5.  **Auditoría y Explicabilidad (XAI):**
    *   Generación de **Matrices de Confusión** detalladas por modelo.
    *   **Grad-CAM (Heatmaps):** Visualización de las regiones faciales que más influyen en la predicción del modelo.
6.  **Validación Externa:** Clasificación automática de sujetos externos mediante URLs (test de estrés del modelo).

## Reportes y Resultados

Tras la ejecución, la carpeta `/reports` contendrá:
- `illinois_quality.png` / `lfw_quality.png`: Análisis técnico de las imágenes.
- `illinois_demographics.png` / `lfw_demographics.png`: Perfilado estadístico de los sujetos.
- `illinois_average_face.png` / `lfw_average_face.png`: Visualización del canon facial de cada dataset.
- `confusion_matrix_[modelo].png`: Rendimiento exacto del clasificador.
- `heatmap_[modelo]_[imagen].jpg`: Explicación visual de por qué el modelo tomó una decisión.
- `url_[nombre]_[modelo].png`: Resultados de la clasificación de sujetos conocidos externos.

## Uso

Dentro del repositorio en un terminal tipo bash ejecutar:

```bash
uv venv --python 3.13
uv sync
source .venv/bin/activate
pre-commit install
```

Para ejecutar el pipeline completo (Descarga, EDA, Preprocesamiento, Entrenamiento y Auditoría):

```bash
uv run app/main.py
```

---

### Autores

- **Oriol**
- **Guillem**
- **Pablo**

---

*Descargo de responsabilidad: Este proyecto tiene fines académicos en el dominio de la Biometría Forense. Ningún ser humano ha sido juzgado legalmente por este README.*
