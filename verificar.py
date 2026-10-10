# verificar.py
import sys
print(f"✅ Python versión: {sys.version.split()[0]}")

try:
    import torch
    print(f"✅ PyTorch instalado. Versión: {torch.__version__}")
    print(f"   GPU disponible: {torch.cuda.is_available()}")
except ImportError:
    print("❌ Error: PyTorch no está instalado.")

try:
    import transformers
    print(f"✅ Transformers instalado. Versión: {transformers.__version__}")
except ImportError:
    print("❌ Error: Transformers no está instalado.")

try:
    import spacy
    nlp = spacy.load("es_core_news_md")
    print("✅ spaCy y modelo 'es_core_news_md' cargados correctamente.")
except (ImportError, OSError) as e:
    print(f"❌ Error: spaCy o el modelo en español no están instalados. {e}")

try:
    import streamlit
    print(f"✅ Streamlit instalado. Versión: {streamlit.__version__}")
except ImportError:
    print("❌ Error: Streamlit no está instalado.")

try:
    import pandas
    print(f"✅ Pandas instalado. Versión: {pandas.__version__}")
except ImportError:
    print("❌ Error: Pandas no está instalado.")

print("\n🎉 ¡Entorno configurado exitosamente! Listo para el Sprint 1.")