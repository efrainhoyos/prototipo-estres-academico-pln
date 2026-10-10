"""
Pipeline de Preprocesamiento de Lenguaje Natural (PLN)
Proyecto: Detección de Estrés Académico mediante PLN
Autores: Efraín Alfonso Hoyos Muñoz, Yheison José Pino Acosta
"""

import re
import spacy
import nltk
from nltk.corpus import stopwords
from typing import List, Dict
import logging

# Configurar logging para registrar el proceso
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class PipelinePLN:
    """
    Pipeline completo de preprocesamiento de texto para análisis de sentimientos.
    """
    
    def __init__(self, idioma: str = 'es'):
        self.idioma = idioma
        
        try:
            if idioma == 'es':
                # Cargar modelo de spaCy para español (incluye lematización)
                self.nlp = spacy.load('es_core_news_md')
                # Cargar stopwords de NLTK en español
                self.stop_words = set(stopwords.words('spanish'))
                # Stopwords personalizadas para dominio académico (opcional)
                self.stop_words.update(['estudiante', 'universidad', 'facultad', 'clase', 'profesor'])
            else:
                raise ValueError(f"Idioma no soportado: {idioma}. Use 'es'.")
            
            logger.info(f"Pipeline inicializado exitosamente para idioma: {idioma}")
            
        except Exception as e:
            logger.error(f"Error al inicializar el pipeline: {e}")
            raise

    def normalizar_texto(self, texto: str) -> str:
        if not texto or not isinstance(texto, str):
            return ""
        
        # 1. Convertir a minúsculas
        texto = texto.lower()
        # 2. Eliminar URLs
        texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
        # 3. Eliminar menciones (@) y hashtags (#)
        texto = re.sub(r'@\w+', '', texto)
        texto = re.sub(r'#', '', texto)
        # 4. Eliminar caracteres especiales y números (mantener solo letras y espacios)
        texto = re.sub(r'[^a-záéíóúñü\s]', '', texto)
        # 5. Eliminar espacios múltiples y recortar extremos
        texto = re.sub(r'\s+', ' ', texto).strip()
        
        return texto

    def tokenizar(self, texto: str) -> List[str]:
        if not texto:
            return []
        doc = self.nlp(texto)
        return [token.text for token in doc if token.text.strip()]

    def remover_stopwords(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return []
        return [token for token in tokens if token.lower() not in self.stop_words and len(token) > 1]

    def lematizar(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return []
        # Reconstruir texto temporalmente para que spaCy lo procese como una unidad
        texto = ' '.join(tokens)
        doc = self.nlp(texto)
        return [token.lemma_ for token in doc if token.text.strip() and token.lemma_ != '-PRON-']

    def procesar_texto_completo(self, texto: str, return_steps: bool = False) -> Dict:
        resultado = {
            'texto_original': texto,
            'texto_procesado': '',
            'tokens': [],
            'lemas': []
        }
        
        if not texto or not isinstance(texto, str):
            return resultado
        
        try:
            texto_normalizado = self.normalizar_texto(texto)
            if not texto_normalizado:
                return resultado
            
            tokens = self.tokenizar(texto_normalizado)
            tokens_sin_stopwords = self.remover_stopwords(tokens)
            lemas = self.lematizar(tokens_sin_stopwords)
            
            resultado['texto_procesado'] = ' '.join(lemas)
            resultado['tokens'] = tokens_sin_stopwords
            resultado['lemas'] = lemas
            
            if return_steps:
                resultado['texto_normalizado'] = texto_normalizado
                resultado['tokens_originales'] = tokens
            
            return resultado
            
        except Exception as e:
            logger.error(f"Error al procesar el texto: '{texto}'. Detalle: {e}")
            return resultado

# Bloque de prueba rápida
if __name__ == "__main__":
    print("Iniciando prueba del Pipeline PLN...")
    pipeline = PipelinePLN(idioma='es')
    
    texto_ejemplo = "¡Estoy MUY estresado con los exámenes finales! No puedo más con tanta carga académica 😔. Visita https://ejemplo.com para más info @usuario"
    
    resultado = pipeline.procesar_texto_completo(texto_ejemplo, return_steps=True)
    
    print("\n" + "="*70)
    print("RESULTADOS DEL PIPELINE DE PREPROCESAMIENTO")
    print("="*70)
    print(f"\n📝 Texto original:\n{resultado['texto_original']}")
    print(f"\n🔧 Paso 1 - Normalización:\n{resultado.get('texto_normalizado', 'N/A')}")
    print(f"\n🔤 Paso 2 - Tokenización:\n{resultado.get('tokens_originales', [])}")
    print(f"\n🚫 Paso 3 - Sin stopwords:\n{resultado['tokens']}")
    print(f"\n📚 Paso 4 - Lematización:\n{resultado['lemas']}")
    print(f"\n✨ Texto final procesado:\n{resultado['texto_procesado']}")
    print("="*70 + "\n")