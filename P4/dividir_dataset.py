import os
import shutil
from pathlib import Path
import random
import math

# --- 1. CONFIGURACIÓN ---

# 1. Ruta a tu carpeta con TODAS las imágenes (ej: ./TGC_RBNW/imagenes_totales)
SOURCE_IMAGES_DIR = Path('./plates/images')

# 2. Ruta a tu carpeta con TODOS los .txt (ej: ./TGC_RBNW/labels_totales)
SOURCE_LABELS_DIR = Path('./plates/txt') 

# 3. Ruta de destino donde se creará el dataset dividido
OUTPUT_DIR = Path('./dataset_dividido') # <--- AJUSTA ESTO (se creará si no existe)

# 4. Proporciones (la suma no debe ser > 1.0)
#    Test se quedará con el porcentaje restante.
#
#    Para 70% train, 20% val, 10% test (Recomendado):
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2

# Extensiones de imagen que buscará
LABEL_EXTENSION = '.txt'

# --- FIN DE CONFIGURACIÓN ---


def split_dataset():
    """
    Encuentra todos los pares de imagen/etiqueta, los baraja
    y los copia a las carpetas train/val/test.
    """
    print("Iniciando el script de división...")
    print(f"Directorio de Imágenes: {SOURCE_IMAGES_DIR}")
    print(f"Directorio de Etiquetas: {SOURCE_LABELS_DIR}")
    print(f"Directorio de Salida:   {OUTPUT_DIR}")
    
    # 1. Crear directorios de destino
    try:
        for split in ['train', 'val', 'test']:
            os.makedirs(OUTPUT_DIR / split / 'images', exist_ok=True)
            os.makedirs(OUTPUT_DIR / split / 'labels', exist_ok=True)
    except Exception as e:
        print(f"Error creando directorios de destino: {e}")
        return

    # 2. Encontrar todos los archivos de imagen
    print(f"Buscando imágenes con extensiones: {IMAGE_EXTENSIONS}")
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(list(SOURCE_IMAGES_DIR.glob(f'*{ext}')))
    
    if not image_files:
        print(f"¡Error! No se encontraron imágenes en '{SOURCE_IMAGES_DIR}'")
        return

    print(f"Se encontraron {len(image_files)} imágenes.")
    
    # 3. Emparejar imágenes con etiquetas y barajar
    paired_files = []
    missing_labels = 0
    
    for img_path in image_files:
        # Nombre base sin extensión (ej: 'img_001')
        basename = img_path.stem
        
        # Ruta de la etiqueta correspondiente
        label_path = SOURCE_LABELS_DIR / f"{basename}{LABEL_EXTENSION}"

        # Comprobar si la etiqueta existe
        if label_path.exists():
            paired_files.append((img_path, label_path))
        else:
            print(f"  ¡Aviso! Falta la etiqueta para: {img_path.name}")
            missing_labels += 1
            
    if not paired_files:
        print("¡Error! No se encontró ningún par válido de imagen/etiqueta.")
        return
        
    print(f"Se encontraron {len(paired_files)} pares (imagen+etiqueta).")
    print(f"Se omitieron {missing_labels} imágenes por falta de etiqueta.")
    
    # Barajar la lista de pares
    random.shuffle(paired_files)

    # 4. Calcular puntos de corte
    total_files = len(paired_files)
    train_split_idx = math.floor(total_files * TRAIN_RATIO)
    val_split_idx = math.floor(total_files * (TRAIN_RATIO + VAL_RATIO))

    # 5. Dividir la lista de archivos
    train_files = paired_files[0:train_split_idx]
    val_files = paired_files[train_split_idx:val_split_idx]
    test_files = paired_files[val_split_idx:]

    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    # 6. Copiar archivos
    print("\nCopiando archivos a sus destinos...")
    
    counts = {'train': 0, 'val': 0, 'test': 0}

    for split, files in splits.items():
        print(f"Procesando split: {split} ({len(files)} archivos)")
        for img_path, label_path in files:
            try:
                # Definir rutas de destino
                dest_img_path = OUTPUT_DIR / split / 'images' / img_path.name
                dest_label_path = OUTPUT_DIR / split / 'labels' / label_path.name

                # Copiar
                shutil.copy(img_path, dest_img_path)
                shutil.copy(label_path, dest_label_path)
                counts[split] += 1
            except Exception as e:
                print(f"  Error copiando {img_path.name}: {e}")

    # --- 7. Resumen Final ---
    print("\n--- ¡Proceso completado! ---")
    print(f"Total de pares copiados: {len(paired_files)}")
    print(f"  - Train: {counts['train']} archivos")
    print(f"  - Val:   {counts['val']} archivos")
    print(f"  - Test:  {counts['test']} archivos")
    print(f"\nDataset listo en: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    split_dataset()