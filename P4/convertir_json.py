import json
import os
from tqdm import tqdm # Para una barra de progreso

# --- 1. CONFIGURACIÓN ---
# ¡¡IMPORTANTE: ADAPTA ESTAS VARIABLES!!

# ---
# DEFINE TUS CARPETAS
# ( tendrás que ejecutar este script 3 veces, 
#   una para train, una para val y una para test )
# ---
# Empecemos con la carpeta 'test'
JSON_DIR = 'TGC_RBNW/test/labels/'  # Dónde están tus .json (y dónde se guardarán los .txt)
IMAGE_DIR = 'TGC_RBNW/test/images/' # Dónde están las imágenes (para verificar nombres)

# ---
# DEFINE TU MAPA DE CLASES
# ¡¡ASEGÚRATE DE QUE EL NOMBRE 'matricula' ES EXACTAMENTE
# EL MISMO QUE ESCRIBISTE EN LABELME!!
# ---
CLASS_MAP = {
    'matricula': 0
    # Si tuvieras más clases, las añadirías aquí:
    # 'coche': 1,
    # 'persona': 2,
}
# --- FIN DE LA CONFIGURACIÓN ---


def convert_labelme_to_yolo(json_dir, image_dir, class_map):
    """
    Convierte anotaciones .json de labelme a formato .txt de YOLO.
    Guarda los .txt en el mismo directorio que los .json.
    """
    
    # 1. Encontrar todos los archivos .json
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    if not json_files:
        print(f"No se encontraron archivos .json en {json_dir}")
        return

    print(f"Encontrados {len(json_files)} archivos JSON. Iniciando conversión...")
    
    # 2. Procesar cada archivo
    for json_file in tqdm(json_files):
        json_path = os.path.join(json_dir, json_file)
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error al leer {json_file}: {e}")
            continue
            
        # 3. Obtener dimensiones de la imagen (necesarias para normalizar)
        img_width = data['imageWidth']
        img_height = data['imageHeight']
        
        # 4. Verificar que la imagen exista
        img_filename = data['imagePath']
        if not os.path.exists(os.path.join(image_dir, img_filename)):
            print(f"\nAdvertencia: No se encuentra la imagen {img_filename} para {json_file}. Saltando.")
            continue
        
        yolo_annotations = []
        
        # 5. Iterar sobre cada "shape" (anotación/caja)
        for shape in data['shapes']:
            label = shape['label']
            
            # 6. Obtener el ID de la clase
            if label not in class_map:
                print(f"\nAdvertencia: Etiqueta '{label}' en {json_file} no está en CLASS_MAP. Saltando.")
                continue
            class_id = class_map[label]
            
            # 7. Obtener coordenadas (labelme guarda [x1, y1] y [x2, y2])
            if shape['shape_type'] != 'rectangle':
                print(f"\nAdvertencia: Shape no es 'rectangle' en {json_file}. Saltando.")
                continue
                
            points = shape['points']
            x_min = min(points[0][0], points[1][0])
            y_min = min(points[0][1], points[1][1])
            x_max = max(points[0][0], points[1][0])
            y_max = max(points[0][1], points[1][1])
            
            # 8. Convertir a formato YOLO (x_center, y_center, width, height)
            box_width = x_max - x_min
            box_height = y_max - y_min
            x_center = x_min + (box_width / 2)
            y_center = y_min + (box_height / 2)
            
            # 9. Normalizar (dividir por las dimensiones de la imagen)
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            width_norm = box_width / img_width
            height_norm = box_height / img_height
            
            # 10. Formatear la línea de YOLO
            yolo_line = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
            yolo_annotations.append(yolo_line)
        
        # 11. Escribir el archivo .txt
        # (ej. '27716760.json' -> '27716760.txt')
        txt_filename = os.path.splitext(img_filename)[0] + '.txt'
        txt_output_path = os.path.join(json_dir, txt_filename)
        
        with open(txt_output_path, 'w') as f:
            f.write("\n".join(yolo_annotations))

    print(f"\n¡Conversión completada para {json_dir}!")
    print(f"Archivos TXT guardados en: {json_dir}")

# --- 2. EJECUTAR EL SCRIPT ---
if __name__ == "__main__":
    
    # -----------------
    # PRIMERA EJECUCIÓN: TEST
    print("--- PROCESANDO CARPETA TEST ---")
    JSON_DIR_TEST = 'TGC_RBNW/test/labels/'
    IMAGE_DIR_TEST = 'TGC_RBNW/test/images/'
    convert_labelme_to_yolo(JSON_DIR_TEST, IMAGE_DIR_TEST, CLASS_MAP)
    
    # -----------------
    # SEGUNDA EJECUCIÓN: TRAIN
    print("\n--- PROCESANDO CARPETA TRAIN ---")
    JSON_DIR_TRAIN = 'TGC_RBNW/train/labels/'
    IMAGE_DIR_TRAIN = 'TGC_RBNW/train/images/'
    convert_labelme_to_yolo(JSON_DIR_TRAIN, IMAGE_DIR_TRAIN, CLASS_MAP)
    
    # -----------------
    # TERCERA EJECUCIÓN: VAL
    print("\n--- PROCESANDO CARPETA VAL ---")
    JSON_DIR_VAL = 'TGC_RBNW/val/labels/'
    IMAGE_DIR_VAL = 'TGC_RBNW/val/images/'
    convert_labelme_to_yolo(JSON_DIR_VAL, IMAGE_DIR_VAL, CLASS_MAP)