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

## El Proceso (Pipeline)

1. **Selección e Integración:** Unificación de los datasets LFW (Control) e Illinois DOC (Riesgo).
2. **Preprocesamiento:**
   - Reescalado uniforme y normalización de píxeles (escala 0-1).
   - **Detección de Landmarks Faciales** para alinear rostros y eliminar sesgos por inclinación.
3. **Minería de Datos (Modelado):** Configuración de la CNN con optimización automatizada.
4. **Extracción y Evaluación:** Generación de predicciones y análisis de activación de capas para ver *exactamente* qué rasgos "delatan" al sujeto.

## Diseño Experimental

Para garantizar que nuestros sesgos sean "estadísticamente sólidos":

- **Estimador de Error:** 80% Entrenamiento / 20% Test (Holdout).
- **Validación Cruzada:** k-fold durante la fase de optimización con Optuna.
- **Métricas:**
  - **Matriz de Confusión:** Para cuantificar cuántos "inocentes" han sido señalados.
  - **F1-Score y AUC-ROC:** Para medir la verdadera capacidad discriminatoria del modelo.

---
## USO

Dentro del repositorio en un terminal tipo bash ejecutar:

```bash
uv venv --python 3.13
uv sync
source .venv/bin/activate
```

To download the data you will have to run the next command:
```bash
uv run main.py
```

This command downloads and unzips the content of
[davidjfisher's illinois-doc-labeled-faces-dataset](https://www.kaggle.com/datasets/davidjfisher/illinois-doc-labeled-faces-dataset)

---

### Autores

- **Oriol**
- **Guillem**
- **Pablo**

---

*Descargo de responsabilidad: Este proyecto tiene fines académicos en el dominio de la Biometría Forense. Ningún ser humano ha sido juzgado legalmente por este README.*
