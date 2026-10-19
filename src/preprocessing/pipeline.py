"""
Pipeline de Preprocesamiento de Lenguaje Natural (PLN)
Proyecto: Deteccion de Estres Academico mediante PLN
Autores: Efrain Alfonso Hoyos Munoz, Yheison Jose Pino Acosta
"""

import re
import spacy
import nltk
from nltk.corpus import stopwords
from typing import List, Dict
import logging

# Configura logging para registrar el proceso
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class PipelinePLN:
    """
    Pipeline completo de preprocesamiento de texto para analisis de sentimientos.
    """

    def __init__(self, idioma: str = 'es'):
        self.idioma = idioma

        try:
            if idioma == 'es':
                # Carga modelo de spaCy para espanol (incluye lematizacion)
                self.nlp = spacy.load('es_core_news_md')
                # Carga stopwords de NLTK en espanol
                self.stop_words = set(stopwords.words('spanish'))
                # Agrega stopwords personalizadas para dominio academico
                self.stop_words.update([
                    'estudiante', 'universidad', 'facultad', 'clase', 'profesor'
                ])
            else:
                raise ValueError(f"Idioma no soportado: {idioma}. Use 'es'.")

            logger.info(f"Pipeline inicializado exitosamente para idioma: {idioma}")

        except Exception as e:
            logger.error(f"Error al inicializar el pipeline: {e}")
            raise

    def normalizar_texto(self, texto: str) -> str:
        """Convierte a minusculas y elimina URLs, menciones y caracteres especiales."""
        if not texto or not isinstance(texto, str):
            return ""

        # Convierte a minusculas
        texto = texto.lower()

        # Elimina URLs
        texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)

        # Elimina menciones (@) y hashtags (#)
        texto = re.sub(r'@\w+', '', texto)
        texto = re.sub(r'#', '', texto)

        # Elimina emojis y caracteres especiales (mantiene letras con tildes y n)
        texto = re.sub(r'[^\w\sáéíóúñüÁÉÍÓÚÑÜ]', ' ', texto)

        # Corrige palabras pegadas comunes en espanol
        texto = re.sub(r'depronto', 'de pronto', texto)
        texto = re.sub(r'tambien', 'tambien', texto)

        # Elimina espacios multiples y recorta extremos
        texto = re.sub(r'\s+', ' ', texto).strip()

        return texto

    def tokenizar(self, texto: str) -> List[str]:
        """Divide el texto en tokens usando spaCy."""
        if not texto:
            return []
        doc = self.nlp(texto)
        return [token.text for token in doc if token.text.strip()]

    def remover_stopwords(self, tokens: List[str]) -> List[str]:
        """Elimina palabras vacias del espanol."""
        if not tokens:
            return []
        return [
            token for token in tokens
            if token.lower() not in self.stop_words and len(token) > 1
        ]

    def lematizar(self, tokens: List[str]) -> List[str]:
        """Reduce palabras a su forma canonica (lema)."""
        if not tokens:
            return []
        # Reconstruye texto temporalmente para que spaCy lo procese como unidad
        texto = ' '.join(tokens)
        doc = self.nlp(texto)
        return [
            token.lemma_ for token in doc
            if token.text.strip() and token.lemma_ != '-PRON-'
        ]

    def procesar_texto_completo(self, texto: str, return_steps: bool = False) -> Dict:
        """Ejecuta el pipeline completo y retorna el texto procesado."""
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


# Bloque de prueba rapida
if __name__ == "__main__":
    print("Iniciando prueba del Pipeline PLN...")
    pipeline = PipelinePLN(idioma='es')

    texto_ejemplo = (
        "Estoy MUY estresado con los examenes finales! "
        "No puedo mas con tanta carga academica. "
        "Visita https://ejemplo.com para mas info @usuario"
    )

    resultado = pipeline.procesar_texto_completo(texto_ejemplo, return_steps=True)

    print("\n" + "=" * 70)
    print("RESULTADOS DEL PIPELINE DE PREPROCESAMIENTO")
    print("=" * 70)
    print(f"\nTexto original:\n{resultado['texto_original']}")
    print(f"\nPaso 1 - Normalizacion:\n{resultado.get('texto_normalizado', 'N/A')}")
    print(f"\nPaso 2 - Tokenizacion:\n{resultado.get('tokens_originales', [])}")
    print(f"\nPaso 3 - Sin stopwords:\n{resultado['tokens']}")
    print(f"\nPaso 4 - Lematizacion:\n{resultado['lemas']}")
    print(f"\nTexto final procesado:\n{resultado['texto_procesado']}")
    print("=" * 70 + "\n")