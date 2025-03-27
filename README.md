# Multimodal-U-Net-for-Infarct-Segmentation-in-MRI

## Resumen  

Este proyecto desarrolla un algoritmo de segmentación para detectar lesiones isquémicas en resonancias magnéticas cerebrales. Se basa en técnicas de procesamiento de imágenes y aprendizaje profundo, optimizando la identificación y delimitación de las áreas afectadas.  

El modelo ha sido entrenado con la base de datos del **ISLES CHALLENGE 2022**, que contiene **250 estudios de RMN** en distintas modalidades:  

- **T2 y ADC**  
- **T2 y DWI**  
- **T2 FLAIR**  
- **Máscaras de segmentación**

  ![image](https://github.com/user-attachments/assets/893d33de-da34-4298-a0ea-b46fe9e39096)

Gracias a esta aproximación, se mejora la evaluación del daño cerebral y se facilita su aplicación en entornos clínicos.  

## Resonancia Magnética  

La **resonancia magnética (RMN)** es una técnica de diagnóstico por imágenes que utiliza campos magnéticos y ondas de radio para obtener representaciones detalladas del cuerpo en tres dimensiones. A diferencia de las técnicas 2D tradicionales, permite la adquisición de cortes volumétricos que pueden analizarse en distintos planos:  

- **Sagital**: Divide el cuerpo en mitades izquierda y derecha.  
- **Frontal (Coronal)**: Separa el cuerpo en mitades anterior y posterior.  
- **Transversal (Axial)**: Divide el cuerpo en partes superior e inferior.

  ![image](https://github.com/user-attachments/assets/89cace77-44bc-4208-b044-ccc1e52331ea)

Esta capacidad de reconstrucción en múltiples planos proporciona una visión detallada y de alta resolución, ideal para la segmentación de estructuras anatómicas y la identificación de patologías.  


Para un diagnóstico preciso del ictus isquémico, es clave comparar imágenes **ADC** y **DWI**. La **DWI** permite detectar isquemias de forma temprana al revelar cambios en la difusión del agua antes de que sean visibles en otras secuencias. Sin embargo, su alta sensibilidad puede generar **falsos positivos**.  

Por otro lado, la **ADC** proporciona datos cuantitativos sobre la difusión, diferenciando entre un ictus real y otras condiciones como inflamación o tumores. Comparar ambas imágenes mejora la precisión diagnóstica y reduce errores.  

En este proyecto, se entrenaron **tres modelos U-Net** para evaluar su rendimiento:  

1. **U-Net ADC**: Entrenado solo con imágenes ADC.  
2. **U-Net DWI**: Entrenado solo con imágenes DWI.  
3. **U-Net Multimodal**: Entrenado con ambas modalidades (ADC + DWI).  

Cada modelo se entrenó con la misma configuración y datos durante **100 épocas**, asegurando condiciones equitativas para la comparación de resultados. 

## Resultados
![image](https://github.com/user-attachments/assets/1dc78d65-f3e7-47df-b8d7-a54c9f10860b)

![image](https://github.com/user-attachments/assets/b553fca5-df06-44c9-93f3-72075b12135d)

Predicciones de modelos U-Net en Strokecase0048, capa 37 tras el entreno



