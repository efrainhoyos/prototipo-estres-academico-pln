"""
Generador de dataset sintético para entrenamiento del modelo BETO.
Simula 100 respuestas de estudiantes de Ingeniería de Sistemas de la FUP.
"""

import csv
import os

# Datos sintéticos realistas distribuidos en 3 clases
# 0 = Positivo/Tranquilo, 1 = Neutro, 2 = Negativo/Estresado

dataset = [
    # === CLASE 0: Positivo/Tranquilo (34 registros) ===
    (1, "Me siento muy bien con mi progreso en el semestre, las clases están interesantes", 0),
    (2, "Hoy entendí por fin los árboles binarios, me siento satisfecho con mi aprendizaje", 0),
    (3, "La clase de programación estuvo excelente, logré resolver todos los ejercicios", 0),
    (4, "Me gusta mucho la carrera, cada día aprendo cosas nuevas y motivadoras", 0),
    (5, "Terminé el proyecto a tiempo y quedó muy bien, estoy orgulloso del resultado", 0),
    (6, "Los profesores explican muy bien y me siento apoyado en mi proceso académico", 0),
    (7, "Tengo todo bajo control con mis tareas y me siento tranquilo con el semestre", 0),
    (8, "Me siento motivado con las clases de bases de datos, todo va muy bien", 0),
    (9, "Aprobé el parcial con buena nota, estoy contento con mi esfuerzo", 0),
    (10, "La universidad me brinda buenas oportunidades y me siento agradecido", 0),
    (11, "Disfruto aprender sobre desarrollo web, es mi área favorita", 0),
    (12, "Me siento seguro con mis conocimientos en programación orientada a objetos", 0),
    (13, "El ambiente en la universidad es muy bueno y me siento a gusto", 0),
    (14, "Logré organizar bien mi tiempo y todo marcha según lo planeado", 0),
    (15, "Estoy muy contento con mi grupo de trabajo, somos muy eficientes", 0),
    (16, "Me siento preparado para el examen de mañana, estudié bastante", 0),
    (17, "La carrera de sistemas me apasiona y no cambiaría lo que estoy haciendo", 0),
    (18, "Todo está saliendo bien este semestre, me siento en paz", 0),
    (19, "Me gusta el reto de resolver problemas con algoritmos, me divierte", 0),
    (20, "Estoy satisfecho con mi rendimiento académico este periodo", 0),
    (21, "Los compañeros son muy colaborativos y el ambiente es positivo", 0),
    (22, "Me siento confiado con las habilidades que estoy adquiriendo", 0),
    (23, "Aprendí mucho en el laboratorio de hoy, fue una gran experiencia", 0),
    (24, "Todo va en orden con mis entregas, no tengo preocupaciones", 0),
    (25, "Me siento realizado con el proyecto final, quedó mejor de lo esperado", 0),
    (26, "La FUP me está dando una excelente formación como ingeniero", 0),
    (27, "Estoy tranquilo con mi carga académica, puedo con todo", 0),
    (28, "Me encanta cuando el código funciona a la primera, gran satisfacción", 0),
    (29, "Siento que estoy en el camino correcto con mi carrera", 0),
    (30, "El semestre va muy bien, tengo buenas notas y buen ánimo", 0),
    (31, "Me siento orgulloso de ser estudiante de Ingeniería de Sistemas", 0),
    (32, "Todo fluye bien, tengo buen equilibrio entre estudio y descanso", 0),
    (33, "Agradezco el apoyo de mis profesores, son muy dedicados", 0),
    (34, "Me siento feliz con mi decisión de estudiar esta carrera", 0),

    # === CLASE 1: Neutro (33 registros) ===
    (35, "Hoy tuve clase normal, nada fuera de lo común", 1),
    (36, "El semestre va transcurriendo como siempre, sin novedades", 1),
    (37, "Tengo tareas pendientes pero nada que no pueda manejar", 1),
    (38, "La clase estuvo bien, ni muy buena ni muy mala", 1),
    (39, "Voy al día con mis entregas, todo en orden", 1),
    (40, "Este semestre es similar al anterior, misma rutina", 1),
    (41, "Tengo algunas materias más difíciles que otras, pero normal", 1),
    (42, "La universidad funciona bien, sin quejas ni elogios", 1),
    (43, "Mis notas están en el promedio, ni muy altas ni muy bajas", 1),
    (44, "El profesor explicó el tema, quedé más o menos claro", 1),
    (45, "Tengo un proyecto para entregar la próxima semana", 1),
    (46, "La carga académica es la esperada para este semestre", 1),
    (47, "Asisto a clases regularmente, cumplo con lo básico", 1),
    (48, "No tengo nada especial que reportar sobre mi situación académica", 1),
    (49, "Voy más o menos bien, sin grandes altibajos", 1),
    (50, "Las materias están en su nivel normal de dificultad", 1),
    (51, "Tengo compañeros nuevos este semestre, aún los conozco", 1),
    (52, "El horario está bien distribuido, ni muy cargado ni muy libre", 1),
    (53, "Estoy en el semestre que me corresponde, todo en orden", 1),
    (54, "La metodología de clase es la de siempre, nada nuevo", 1),
    (55, "Tengo pendientes algunas lecturas pero nada urgente", 1),
    (56, "El ambiente académico es el esperado para la universidad", 1),
    (57, "Voy cumpliendo con los requisitos del programa", 1),
    (58, "Las evaluaciones son como en semestres anteriores", 1),
    (59, "No tengo quejas particulares sobre mi formación", 1),
    (60, "El semestre va en su punto medio, todo normal", 1),
    (61, "Tengo actividades regulares como cualquier estudiante", 1),
    (62, "La carrera sigue su curso, sin sorpresas", 1),
    (63, "Mis responsabilidades académicas son las habituales", 1),
    (64, "No tengo nada especial que destacar esta semana", 1),
    (65, "Voy avanzando en el semestre de manera estándar", 1),
    (66, "La dinámica de clases es la acostumbrada", 1),
    (67, "Todo sigue igual que el semestre pasado", 1),

    # === CLASE 2: Negativo/Estresado (33 registros) ===
    (68, "No puedo más con la carga de tareas, estoy agotado mentalmente", 2),
    (69, "Tengo ansiedad constante por los parciales que se acercan", 2),
    (70, "No duermo bien por la presión de entregar el proyecto a tiempo", 2),
    (71, "Me siento frustrado porque no entiendo los algoritmos complejos", 2),
    (72, "La sobrecarga académica me está afectando la salud", 2),
    (73, "No sé cómo voy a terminar todos los trabajos de esta semana", 2),
    (74, "Estoy muy estresado con el semestre, no doy abasto", 2),
    (75, "Me siento abrumado por la cantidad de asignaturas simultáneas", 2),
    (76, "La presión por mantener el promedio me tiene muy angustiado", 2),
    (77, "No tengo tiempo ni para comer bien por la universidad", 2),
    (78, "Me siento incapaz de seguir el ritmo de la carrera", 2),
    (79, "Los exámenes me generan pánico, no puedo concentrarme", 2),
    (80, "Estoy al límite, necesito ayuda con tanto estrés académico", 2),
    (81, "No puedo con la presión de los profesores exigentes", 2),
    (82, "Me siento solo enfrentando todas estas dificultades académicas", 2),
    (83, "La ansiedad no me deja rendir bien en las evaluaciones", 2),
    (84, "Tengo miedo de perder el semestre por tanto estrés", 2),
    (85, "No aguanto más esta presión constante de la universidad", 2),
    (86, "Me siento fracasado cuando no entiendo la materia", 2),
    (87, "El estrés me está causando problemas físicos, dolores de cabeza", 2),
    (88, "No tengo vida social por culpa de la carga académica", 2),
    (89, "Me siento perdido en la carrera, no sé si podré terminar", 2),
    (90, "La competencia entre compañeros me genera mucha ansiedad", 2),
    (91, "No puedo dormir pensando en la entrega de mañana", 2),
    (92, "Estoy quemado, necesito un descanso urgente", 2),
    (93, "Me siento desmotivado y sin energía para seguir", 2),
    (94, "La presión familiar por buenas notas me tiene muy estresado", 2),
    (95, "No tengo tiempo para nada, solo estudio y más estudio", 2),
    (96, "Me siento overwhelmed con tantos proyectos simultáneos", 2),
    (97, "La incertidumbre laboral me genera ansiedad adicional", 2),
    (98, "No puedo concentrarme, mi mente está en blanco por el estrés", 2),
    (99, "Siento que no soy suficiente para esta carrera tan difícil", 2),
    (100, "Estoy al borde del colapso por la presión académica", 2),
]

# Ruta del archivo
ruta = os.path.join('data', 'labeled', 'dataset_ejemplo.csv')

# Crea el directorio si no existe
os.makedirs(os.path.dirname(ruta), exist_ok=True)

# Escribw el CSV
with open(ruta, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'texto', 'etiqueta'])
    for registro in dataset:
        writer.writerow(registro)

print(f"Dataset generado exitosamente con {len(dataset)} registros")
print(f"Ubicación: {os.path.abspath(ruta)}")

# Muestra distribución
from collections import Counter
distribucion = Counter([r[2] for r in dataset])
print("\nDistribución de clases:")
print(f"   Clase 0 (Positivo/Tranquilo): {distribucion[0]} registros")
print(f"   Clase 1 (Neutro):             {distribucion[1]} registros")
print(f"   Clase 2 (Negativo/Estresado): {distribucion[2]} registros")