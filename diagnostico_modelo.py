"""
Script de diagnóstico para verificar el funcionamiento del modelo BETO fine-tuned
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import os

# Agrega src al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from preprocessing.pipeline import PipelinePLN

# Configuración
MODEL_PATH = "src/models/beto_finetuned"
LABELS = {0: "Positivo/Tranquilo", 1: "Neutro", 2: "Negativo/Estresado"}

# Textos de prueba (estos deberían dar resultados variados)
textos_prueba = [
    ("Estoy muy feliz y contento con mi progreso académico", 0),  # Positivo
    ("La clase estuvo bien, nada especial", 1),  # Neutro
    ("No puedo más con el estrés, estoy agotado y frustrado", 2),  # Negativo
    ("Me siento muy estresado con los parciales", 2),  # Negativo
    ("Todo va bien en la universidad", 0),  # Positivo
]

print("="*70)
print("DIAGNÓSTICO DEL MODELO BETO FINE-TUNED")
print("="*70)

# Carga el modelo y tokenizador
print("\nCargando modelo y tokenizador...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    print("Modelo cargado correctamente")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    sys.exit(1)

# Verifica la configuración del modelo
print(f"\nConfiguración del modelo:")
print(f"   - Número de etiquetas: {model.config.num_labels}")
print(f"   - Etiquetas configuradas: {model.config.id2label}")
print(f"   - Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# Inicializa el pipeline
print("\nInicializando pipeline de preprocesamiento...")
pipeline = PipelinePLN(idioma='es')

# Prueba de cada texto
print("\n" + "="*70)
print("PRUEBAS DE CLASIFICACIÓN")
print("="*70)

for i, (texto, etiqueta_esperada) in enumerate(textos_prueba, 1):
    print(f"\n Prueba {i}:")
    print(f"   Texto: {texto}")
    print(f"   Etiqueta esperada: {LABELS[etiqueta_esperada]}")
    
    # Preprocesar
    texto_procesado = pipeline.procesar_texto_completo(texto)['texto_procesado']
    print(f"   Texto procesado: {texto_procesado}")
    
    # Tokenizar
    inputs = tokenizer(texto_procesado, return_tensors="pt", truncation=True, max_length=128)
    
    # Predicción
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_label].item()
    
    # Muestra probabilidades de todas las clases
    print(f"   Probabilidades:")
    for label_id, label_name in LABELS.items():
        prob = probs[0][label_id].item()
        print(f"      - {label_name}: {prob:.4f} ({prob*100:.2f}%)")
    
    print(f"  Predicción: {LABELS[pred_label]} (confianza: {confidence*100:.2f}%)")
    
    # Verificar si es correcto
    if pred_label == etiqueta_esperada:
        print(f"   CORRECTO")
    else:
        print(f"INCORRECTO (esperado: {LABELS[etiqueta_esperada]})")

print("\n" + "="*70)
print("RESUMEN")
print("="*70)
print("\nSi todas las predicciones son 'Positivo/Tranquilo', el modelo no está")
print("aprendiendo correctamente. Posibles causas:")
print("  1. Dataset de entrenamiento desbalanceado o con errores")
print("  2. Muy pocas épocas de entrenamiento")
print("  3. Learning rate muy alto o muy bajo")
print("  4. Problema en el mapeo de etiquetas (id2label/label2id)")
print("\nRecomendación: Reentrenar el modelo con más datos y ajustar hiperparámetros")