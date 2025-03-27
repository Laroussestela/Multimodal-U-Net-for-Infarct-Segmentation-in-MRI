import os

# Configuración de parámetros
VOLUME_START_AT = 20
VOLUME_SLICES = 32
IMG_SIZE = 112
IMG_HEIGHT = IMG_SIZE  # Asumimos que la imagen es cuadrada
IMG_WIDTH = IMG_SIZE  # Asumimos que la imagen es cuadrada
IMG_CHANNELS = 2  # Dos canales (ADC y DWI)
n_classes = 2  # Número de clases para la segmentación

# Directorios de datos
RAW_DATA_PATH = "/content/drive/MyDrive/Colab Notebooks/Multimodal_Unet_Infart_Segmentation"
DERIVATIVES_PATH = "/content/drive/MyDrive/Colab Notebooks/Multimodal_Unet_Infart_Segmentation"

# Ruta para guardar modelos
MODEL_SAVE_PATH = "/content/drive/MyDrive/Colab Notebooks/Multimodal_Unet_Infart_Segmentation"
