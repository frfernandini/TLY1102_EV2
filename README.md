# Informe Técnico: Clasificación de Letras del Lenguaje de Señas mediante CNN y comparación con MLP

## 1. Descripción del problema de negocio
El lenguaje de señas es el principal medio de comunicación para la comunidad sorda. Sin embargo, el acceso a servicios básicos (salud, educación, atención al cliente) sigue siendo limitado debido a la escasez de intérpretes y a la falta de herramientas tecnológicas accesibles. La automatización del reconocimiento de señas mediante técnicas de aprendizaje profundo puede contribuir a reducir estas barreras, facilitando la inclusión y la interacción en entornos cotidianos.

En esta segunda etapa (EP2), el proyecto evoluciona desde un Perceptrón Multicapa (MLP) hacia una **Red Neuronal Convolucional (CNN)**. El objetivo es demostrar empíricamente cómo la capacidad de la CNN para preservar la estructura espacial de las imágenes supera las limitaciones del MLP, permitiendo una clasificación más precisa de señas visualmente similares.

## 2. Objetivos del proyecto

### 2.1 Objetivo general
Diseñar, implementar y evaluar una Red Neuronal Convolucional (CNN) para la clasificación de letras del lenguaje de señas, comparando objetivamente su rendimiento y capacidad de extracción de características frente al modelo MLP desarrollado previamente.

### 2.2 Objetivos específicos
- Mantener la variante del problema: subconjunto de 6 letras visualmente similares (A, E, M, N, S, T) del dataset Sign Language MNIST.
- Preparar los datos conservando la estructura matricial (28x28x1) para la CNN.
- Diseñar una arquitectura CNN justificando sus capas convolucionales, pooling y densas.
- Implementar técnicas de regularización (Dropout) para mitigar el sobreajuste observado en la EP1.
- Comparar el desempeño (Accuracy, F1-Score, matrices de confusión) entre el modelo MLP y la CNN.
- Visualizar e interpretar los mapas de características (Feature Maps) generados por las capas convolucionales.

## 3. KPIs (Indicadores Clave de Desempeño)
1. **Precisión global (Accuracy):** Porcentaje de clasificaciones correctas sobre el total de muestras.
2. **F1-Score por clase:** Especialmente crítico para las clases M, N y S, donde el MLP de la EP1 falló.
3. **Tasa de confusión entre pares:** Análisis de los errores sistemáticos para evaluar si la CNN logró separar las clases complejas.

## 4. Descripción de las fuentes de datos y variante
Se utilizó el dataset **Sign Language MNIST** disponible en Kaggle. Contiene imágenes de 28x28 píxeles en escala de grises.
**Variante:** Se mantiene el subconjunto de **6 letras con alta similitud visual**: A, E, M, N, S y T (etiquetas originales 0, 4, 12, 13, 18, 19). Esta selección se mantiene para garantizar una comparación justa y objetiva entre el modelo MLP anterior y la CNN actual, enfocándose en el desafío de discriminar puños y posiciones de dedos complejas.

## 5. Preparación y análisis exploratorio de datos (EDA)
- **Carga y filtrado:** Se cargaron los CSV y se filtraron las 6 clases seleccionadas, remapeándolas a un rango de 0 a 5.
- **EDA:** Se observó un balance adecuado entre clases. La visualización de imágenes confirmó la variabilidad de iluminación, postura y enfoque.
- **Preprocesamiento (Adaptado para CNN):** 
  - Las imágenes se normalizaron dividiendo por 255.
  - Las etiquetas se codificaron en One-Hot Encoding.
  - **Diferenciador clave:** A diferencia del MLP (donde las imágenes se aplanaban a un vector de 784), para la CNN los datos se redimensionaron a **`(N, 28, 28, 1)`**. Esto preserva la relación espacial de los píxeles, algo vital para las operaciones de convolución.
  - Se mantuvo la partición Train/Validation (80/20) con semilla `random_state=2026` para trazabilidad.

## 6. Metodología (CRISP-DM)
El proyecto siguió las fases de CRISP-DM: Comprensión del negocio (accesibilidad), Comprensión de los datos (Sign Language MNIST), Preparación de datos (redimensionamiento a 3D), Modelamiento (diseño CNN), Evaluación (comparación con MLP y análisis de Feature Maps) y Despliegue (documentación).

## 7. Diseño e Implementación de la CNN
Se implementó una arquitectura secuencial en TensorFlow/Keras con la siguiente estructura:

1. **Capa Convolucional 1:** `Conv2D(32, (3, 3), activation='relu')`. 32 filtros de 3x3 para extraer bordes y contornos básicos.
2. **MaxPooling 1:** `MaxPooling2D((2, 2))`. Reduce la dimensionalidad a la mitad, aportando invarianza a pequeñas traslaciones.
3. **Capa Convolucional 2:** `Conv2D(64, (3, 3), activation='relu')`. 64 filtros para extraer patrones más complejos (uniones de dedos, formas de la palma).
4. **MaxPooling 2:** `MaxPooling2D((2, 2))`. Segunda reducción espacial.
5. **Flatten:** Aplana las características extraídas.
6. **Capa Densa:** `Dense(64, activation='relu')`. Combina las características para la clasificación.
7. **Dropout:** `Dropout(0.5)`. **Decisión clave:** Se añadió para combatir el sobreajuste severo detectado en la EP1, apagando aleatoriamente el 50% de las neuronas durante el entrenamiento.
8. **Capa de Salida:** `Dense(6, activation='softmax')`.

*Justificación de hiperparámetros:* Se utilizó el optimizador **Adam** (learning rate=0.001) y batch size de 64, basándose en los resultados óptimos obtenidos durante la EP1. Se usó `EarlyStopping` (paciencia=10) para detener el entrenamiento si la pérdida de validación dejaba de mejorar.

## 8. Entrenamiento del modelo
El modelo se entrenó registrando las curvas de pérdida y precisión. 
*Nota técnica:* Se observa que el *Validation Accuracy* es superior al *Train Accuracy* durante casi todo el entrenamiento. Esto es un efecto esperado del `Dropout`: durante el entrenamiento la red se ve obligada a trabajar con la mitad de sus neuronas (haciéndola parecer "menos precisa"), pero al validar, el Dropout se desactiva y la red usa todo su potencial. Esto demuestra que el modelo generaliza bien y no está memorizando (overfitting).

## 9. Evaluación del desempeño (CNN)
El modelo CNN fue evaluado sobre el conjunto de prueba (datos no vistos):
- **Test Accuracy:** 94.42%
- **Test Loss:** 0.6847

**Reporte de Clasificación:**
| Clase | Precisión | Recall | F1-Score | Soporte |
|-------|-----------|--------|----------|---------|
| A     | 0.96      | 1.00   | 0.98     | 331     |
| E     | 0.99      | 0.94   | 0.97     | 498     |
| M     | 0.95      | 0.93   | 0.94     | 394     |
| N     | 1.00      | 0.81   | 0.89     | 291     |
| S     | 0.83      | 1.00   | 0.91     | 246     |
| T     | 0.92      | 1.00   | 0.96     | 248     |

## 10. Comparación entre MLP y CNN

| Característica | MLP (EP1) | CNN (EP2) | Mejora / Análisis |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | 81.82% | **94.42%** | **+12.6%**. Salto significativo. |
| **F1-Score (Clase M)** | 0.81 | **0.94** | La CNN entiende la forma del puño. |
| **F1-Score (Clase N)** | 0.63 | **0.89** | Se resolvió gran parte de la confusión con A. |
| **F1-Score (Clase S)** | 0.59 | **0.91** | Mejora drástica en la detección. |
| **Estructura de entrada** | Vector aplanado (784) | Matriz (28x28x1) | La CNN preserva vecindad de píxeles. |
| **Overfitting** | Severo (Train 100% / Test 81%) | Controlado | El Dropout y Pooling regularizan la red. |

**Análisis:** La CNN supera ampliamente al MLP. Mientras el MLP sufría para diferenciar las letras M, N y S (F1-Scores entre 0.59 y 0.81), la CNN logró elevarlos por encima de 0.89. Esto se debe a que los filtros convolucionales son capaces de extraer características locales (como la curvatura de un dedo o el espacio entre ellos), mientras que el MLP solo ve intensidades globales de píxeles.

## 11. Visualización de Mapas de Características (Feature Maps)
Se extrajeron las activaciones de las capas convolucionales para una muestra de la letra "A":
- **Primera Capa (Conv1):** Los filtros se activan fuertemente detectando los bordes de la mano y el contorno de los dedos. Se observa que un filtro (F4) está inactivo (fenómeno de *Dying ReLU*, normal en el entrenamiento). La red extrae características de bajo nivel.
- **Segunda Capa (Conv2):** Las activaciones son más dispersas y abstractas. La mayoría de los filtros están vacíos, lo cual es esperado: tras el MaxPooling la imagen es muy pequeña y esta capa busca patrones complejos que pueden no estar presentes en esta imagen en particular. Los pocos filtros activos detectan uniones específicas de la palma.

*Conclusión de los Feature Maps:* Se demuestra la jerarquía del aprendizaje profundo: las primeras capas ven "bordes" y las capas profundas buscan "conceptos" (formas de dedos).

## 12. Análisis de errores
A pesar de la mejora, la CNN aún presenta confusiones residuales.
- Las clases **M, N y S** siguen siendo las más complejas.
- Al visualizar los errores, se observa que las fallas ocurren cuando la mano está fuera del centro, hay sombras severas o el puño está rotado.
- Sin embargo, la tasa de error es mínima comparada con el MLP. La CNN logra clasificar correctamente imágenes que el MLP confundía sistemáticamente.

## 13. Conclusiones
- **Superioridad de la CNN:** La preservación de la estructura espacial y el uso de filtros compartidos hacen que la CNN sea muy superior para visión artificial. Se superó el cuello de botella del MLP, pasando de un 81.82% a un 94.42% de precisión.
- **Importancia del Dropout:** Fue fundamental para controlar el sobreajuste. Las curvas de aprendizaje demostraron que el modelo generaliza correctamente a datos nuevos.
- **Limitaciones:** El dataset (28x28) es muy pequeño. Aún hay margen de mejora para las letras N y S, donde el modelo tiende a ser muy sensible (alta precisión pero bajo recall, o viceversa).
- **Mejoras futuras:** Implementar *Data Augmentation* (rotaciones leves, zoom) para simular condiciones del mundo real y robustecer el modelo.

## 14. Impacto Ético
El desarrollo de herramientas de traducción automática de señas tiene un impacto social positivo directo al fomentar la inclusión. Sin embargo, es crucial considerar que el dataset utilizado (Sign Language MNIST) contiene imágenes muy homogéneas (fondos limpios, un solo tono de piel). Un modelo desplegado en el mundo real requeriría ser entrenado con un dataset diverso (diferentes etnias, iluminaciones y fondos) para evitar sesgos algorítmicos que excluyan a ciertos grupos demográficos.

## 15. Referencias
- Kaggle Dataset: Sign Language MNIST.
- Documentación de TensorFlow/Keras.
- Materiales de la asignatura TLY1102.
