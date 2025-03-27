from keras.callbacks import EarlyStopping
from model import multi_unet_model
from data_loader import load_data
from tensorflow.keras.optimizers import Adam
from config import *
from segmentation_metrics import *

X_train_adc, X_val_adc, y_train_adc, y_val_adc, X_train_dwi, X_val_dwi, y_train_dwi, y_val_dwi, X_train_multi, X_val_multi, y_train_multi, y_val_multi = load_data()

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

# Callbacks
learning_rate = 0.0001
optimizer = Adam(learning_rate=learning_rate)

model_adc = get_model()
model_adc.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=[MeanIoU(num_classes=2)])

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

history_adc = model_adc.fit(X_train_adc, y_train_adc, batch_size=32, epochs=500, validation_data=(X_val_adc, y_val_adc), callbacks=[early_stopping_callback])

model_adc.save(f"{MODEL_SAVE_PATH}/modelo_ADC.h5") # Save model

# Evaluación del modelo
results = model_adc.evaluate(X_val_adc, y_val_adc, batch_size=100)
print("\nModel evaluation on the validation set:")
for metric, description in zip(results, ["Loss", "MeanIOU"]):
    print(f"{description}: {round(metric, 4)}")
