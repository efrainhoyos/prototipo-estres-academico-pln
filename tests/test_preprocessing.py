"""
Pruebas unitarias para el Pipeline de Preprocesamiento de PLN
Proyecto: Detección de Estrés Académico mediante PLN
"""

import unittest
import sys
import os

# Se Agrega la carpeta raíz al path para importar el pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.pipeline import PipelinePLN


class TestPipelinePLN(unittest.TestCase):
    """Suite de pruebas para el Pipeline de PLN"""
    
    def setUp(self):
        """Configuración inicial antes de cada prueba"""
        self.pipeline = PipelinePLN(idioma='es')
    
    def test_normalizacion_texto(self):
        """Prueba la normalización: minúsculas y eliminación de caracteres especiales"""
        texto = "¡ESTOY MUY ESTRESADO! 😔 #examen"
        resultado = self.pipeline.normalizar_texto(texto)
        
        self.assertEqual(resultado, "estoy muy estresado examen")
        self.assertTrue(resultado.islower())
        self.assertNotIn('#', resultado)
    
    def test_normalizacion_urls(self):
        """Prueba la eliminación de URLs"""
        texto = "Visita https://www.ejemplo.com para más información"
        resultado = self.pipeline.normalizar_texto(texto)
        
        self.assertNotIn('http', resultado)
        self.assertNotIn('www', resultado)
    
    def test_tokenizacion(self):
        """Prueba la división correcta en tokens"""
        texto = "estoy estresado con los examenes"
        tokens = self.pipeline.tokenizar(texto)
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        self.assertIn('estoy', tokens)
    
    def test_remover_stopwords(self):
        """Prueba la eliminación de palabras vacías"""
        tokens = ['estoy', 'muy', 'estresado', 'con', 'los', 'examenes']
        resultado = self.pipeline.remover_stopwords(tokens)
        
        # 'con' y 'los' son stopwords en español
        self.assertNotIn('con', resultado)
        self.assertNotIn('los', resultado)
        self.assertIn('estresado', resultado)
    
    def test_lematizacion(self):
        """Prueba la reducción a la forma base (lema)"""
        tokens = ['estresado', 'cansado', 'agotado']
        lemas = self.pipeline.lematizar(tokens)
        
        self.assertIsInstance(lemas, list)
        self.assertGreater(len(lemas), 0)
        # "estresado" debería lematizarse a "estresar"
        self.assertIn('estresar', lemas)
    
    def test_pipeline_completo(self):
        """Prueba el flujo completo de extremo a extremo"""
        texto = "¡Estoy MUY estresado con los exámenes finales!"
        resultado = self.pipeline.procesar_texto_completo(texto)
        
        self.assertIn('texto_original', resultado)
        self.assertIn('texto_procesado', resultado)
        self.assertEqual(resultado['texto_original'], texto)
        self.assertIsInstance(resultado['texto_procesado'], str)
        self.assertGreater(len(resultado['texto_procesado']), 0)
    
    def test_texto_vacio_o_none(self):
        """Prueba el manejo robusto de entradas inválidas"""
        resultado_vacio = self.pipeline.procesar_texto_completo("")
        resultado_none = self.pipeline.procesar_texto_completo(None)
        
        self.assertEqual(resultado_vacio['texto_procesado'], '')
        self.assertEqual(resultado_none['texto_procesado'], '')


if __name__ == '__main__':
    # Ejecutar las pruebas con salida detallada
    unittest.main(verbosity=2)