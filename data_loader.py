import numpy as np
import cv2
import nibabel as nib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import os
from config import *

def load_data():
    # Inicializar las listas para las imágenes y las máscaras
    X = np.zeros((VOLUME_SLICES * 196, IMG_SIZE, IMG_SIZE, 2))
    y = np.zeros((VOLUME_SLICES * 196, IMG_SIZE, IMG_SIZE))

    samples = []
    sample_descarte = []

    contador = 0
    for archivo in tqdm(range(1, 251)):
        path_adc = f"{RAW_DATA_PATH}/sub-strokecase{archivo:04d}/ses-0001/sub-strokecase{archivo:04d}_ses-0001_adc.nii.gz"
        path_dwi = f"{RAW_DATA_PATH}/sub-strokecase{archivo:04d}/ses-0001/sub-strokecase{archivo:04d}_ses-0001_dwi.nii.gz"
        path_msk = f"{DERIVATIVES_PATH}/sub-strokecase{archivo:04d}/ses-0001/sub-strokecase{archivo:04d}_ses-0001_msk.nii.gz"

        # Cargar las imágenes y la máscara
        archivo_adc = nib.load(path_adc)
        archivo_dwi = nib.load(path_dwi)
        archivo_msk = nib.load(path_msk)

        imagen_adc = archivo_adc.get_fdata()
        imagen_dwi = archivo_dwi.get_fdata()
        imagen_msk = archivo_msk.get_fdata()

        if imagen_adc.shape[2] > 60:
            samples.append(f'sub-strokecase{archivo:04d}')
            for j in range(VOLUME_SLICES):
                X[j + VOLUME_SLICES * contador, :, :, 0] = cv2.resize(imagen_adc[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
                X[j + VOLUME_SLICES * contador, :, :, 1] = cv2.resize(imagen_dwi[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
                y[j + VOLUME_SLICES * contador, :, :] = cv2.resize(imagen_msk[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
            contador += 1
        else:
            sample_descarte.append(f'sub-strokecase{archivo:04d}')

    # Expansión de dimensiones para las máscaras
    y_expand = np.expand_dims(y, axis=3)

    # Reescalado de las máscaras a categorías
    y_expand = to_categorical(y_expand, num_classes=n_classes)
    y_expand = y_expand.reshape((y_expand.shape[0], y_expand.shape[1], y_expand.shape[2], n_classes))

    # División en entrenamiento y validación
    X_train_adc, X_val_adc, y_train_adc, y_val_adc = train_test_split(X[:, :, :, 0], y_expand, test_size=0.20, random_state=0)
    X_train_dwi, X_val_dwi, y_train_dwi, y_val_dwi = train_test_split(X[:, :, :, 1], y_expand, test_size=0.20, random_state=0)
    X_train_multi, X_val_multi, y_train_multi, y_val_multi = train_test_split(X, y_expand, test_size=0.20, random_state=0)

    return X_train_adc, X_val_adc, y_train_adc, y_val_adc, X_train_dwi, X_val_dwi, y_train_dwi, y_val_dwi, X_train_multi, X_val_multi, y_train_multi, y_val_multi
