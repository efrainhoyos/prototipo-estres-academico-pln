# Prototipo de Detección de Estrés Académico con PLN

Proyecto de grado para optar al título de Ingeniero de Sistemas  
**Fundación Universitaria de Popayán - FUP**  
**Programa de Ingeniería de Sistemas**

## Descripción
Prototipo tecnológico basado en Procesamiento de Lenguaje Natural (PLN) para la detección temprana de estrés académico en estudiantes de Ingeniería de Sistemas mediante análisis de sentimientos.

## Objetivos
- Implementar un modelo de análisis de sentimientos basado en transformers (BETO/RoBERTa-es)
- Construir un corpus lingüístico especializado en estrés académico
- Desarrollar una interfaz web funcional para la aplicación del modelo

## Tecnologías
- **Lenguaje:** Python 3.10+
- **PLN:** HuggingFace Transformers, spaCy, NLTK
- **ML:** PyTorch, Scikit-learn
- **Interfaz:** Streamlit
- **Visualización:** Matplotlib, Seaborn

## Estructura del Proyecto
estres-academico-pln/

data/ # Corpus y datos

│ ├── raw/ # Datos crudos (NO subir a GitHub)

│ ├── processed/ # Datos preprocesados

│ └── labeled/ # Datos etiquetados (train/val/test)

├──  docs/ # Documentación
│
├──  notebooks/ # Notebooks de experimentación
│ ├── 01_exploracion.ipynb
│ ├── 02_preprocesamiento.ipynb
│ ├── 03_entrenamiento.ipynb
│ └── 04_evaluacion.ipynb
│
├──  results/ # Resultados y gráficos
│ ├── figures
│ ├── metrics
├──  src/ # Código fuente
│ ├── app/ # Interfaz Streamlit
│ ├── evaluation/ # Métricas y evaluación
│ ├── models/ # Modelos entrenados
│ └── preprocessing/ # Pipeline de PLN
│

├──  tests/ # Pruebas unitarias
│
├── requirements.txt # Dependencias Python
├── .gitignore # Archivos ignorados
├── README.md # Este archivo
└── LICENSE # Licencia MIT

<img width="386" height="713" alt="imagen" src="https://github.com/user-attachments/assets/91554608-80ff-49fd-90b4-193a87c08337" />


## Autores
- Efraín Alfonso Hoyos Muñoz
- Yheison José Pino Acosta

## Director
- Mag. Luis Alfonso Vejarano

## Fecha
Septiembre 2026
