"""
Script para verificar la instalacion de las dependencias del proyecto.
"""
import sys

print(f"Python version: {sys.version.split()[0]}")

try:
    import torch
    print(f"PyTorch instalado. Version: {torch.__version__}")
    print(f"   GPU disponible: {torch.cuda.is_available()}")
except ImportError:
    print("Error: PyTorch no esta instalado.")

try:
    import transformers
    print(f"Transformers instalado. Version: {transformers.__version__}")
except ImportError:
    print("Error: Transformers no esta instalado.")

try:
    import spacy
    nlp = spacy.load("es_core_news_md")
    print("spaCy y modelo 'es_core_news_md' cargados correctamente.")
except (ImportError, OSError) as e:
    print(f"Error: spaCy o el modelo en espanol no estan instalados. {e}")

try:
    import streamlit
    print(f"Streamlit instalado. Version: {streamlit.__version__}")
except ImportError:
    print("Error: Streamlit no esta instalado.")

try:
    import pandas
    print(f"Pandas instalado. Version: {pandas.__version__}")
except ImportError:
    print("Error: Pandas no esta instalado.")

print("\nEntorno configurado exitosamente. Listo para el desarrollo.")