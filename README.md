# Informe Técnico — EP3: Clasificación de letras del lenguaje de señas con CNN y comparación con MLP

**Asignatura:** TLY1102 — Técnicas Avanzadas de Machine Learning I
**Evaluación:** Parcial N°3 — Presentación y defensa técnica del proyecto
**Notebook:** `notebooks/TLY1102_EV3.ipynb`
**Dataset:** Sign Language MNIST (variante de 6 letras: A, E, M, N, S, T)

---

## 1. Resumen del proyecto

Se comparan dos arquitecturas de Deep Learning —un Perceptrón Multicapa (MLP) y una Red Neuronal Convolucional (CNN)— para clasificar 6 letras del alfabeto de señas que comparten la misma postura de puño cerrado y se diferencian solo por la posición del pulgar (A, E, M, N, S, T). Ambas se entrenan bajo condiciones idénticas (mismos datos, misma partición, misma semilla) y se evalúan con métricas múltiples, matriz de confusión, curvas de aprendizaje y Grad-CAM. El modelo final es una **CNN con aumento de datos**, que alcanza **99.35% de accuracy y 0.995 de F1 macro** en el conjunto de prueba.

### Resultados principales

| Modelo | Accuracy test | F1 macro | ROC-AUC | Parámetros | Tiempo entren. |
|---|---:|---:|---:|---:|---:|
| MLP base (128–64, dropout 0.3) | 78.29% | 0.7642 | 0.9532 | 109,126 | 14.8 s |
| MLP optimizada (256–128, dropout 0.2) | 81.62% | 0.7952 | 0.9581 | 234,630 | 18.5 s |
| CNN base (32/64 filtros, dropout 0.5) | 91.09% | 0.9016 | 0.9923 | 121,670 | 195.2 s |
| CNN optimizada (16/32 filtros, dropout 0.2) | 91.98% | 0.9127 | 0.9905 | 56,454 | 99.3 s |
| **CNN + aumento de datos (modelo final)** | **99.35%** | **0.9946** | **1.0000** | **56,902** | **55.3 s** |

Inferencia del modelo final: **0.20 ms por imagen** (2,008 imágenes en 0.41 s).

---

## 2. Descripción del problema de negocio

El lenguaje de señas es el principal medio de comunicación de la comunidad sorda. Sin embargo, el acceso a servicios básicos (salud, educación, atención al cliente) sigue siendo limitado por la escasez de intérpretes y la falta de herramientas tecnológicas accesibles.

Automatizar el reconocimiento de señas reduce esa barrera y habilita casos de uso concretos: señalización en mesas de ayuda, kioscos de autoatención, subtitulado asistido y material educativo accesible.

El alcance de este proyecto es acotado y explícito: **clasificación de letras estáticas** (posturas fijas), no traducción de lenguaje de señas completo. Las señas con movimiento (J y Z) quedan fuera por definición del dataset.

---

## 3. Objetivos del proyecto

### 3.1 Objetivo general
Comparar objetivamente un MLP y una CNN en la clasificación de 6 letras del lenguaje de señas, y seleccionar la arquitectura con mejor capacidad de generalización justificando la decisión con evidencia experimental.

### 3.2 Objetivos específicos
1. Reutilizar el pipeline y la variante de 6 letras del proyecto anterior, manteniendo la trazabilidad.
2. Implementar regularización (Dropout, Batch Normalization, L2, Early Stopping y aumento de datos) y medir su impacto con curvas y métricas.
3. Optimizar hiperparámetros mediante Grid Search y justificar la configuración final.
4. Auditar el modelo con múltiples métricas, matriz de confusión, Grad-CAM y análisis de sesgos.
5. Documentar fortalezas, limitaciones y mejoras del pipeline.

---

## 4. KPIs y metas

| KPI | Definición | Meta | Resultado (modelo final) | Estado |
|---|---|---:|---:|---|
| Accuracy global | Clasificaciones correctas sobre el total | ≥ 95% | 99.35% | Cumple |
| F1-Score clases críticas (M, N, S) | Equilibrio precisión/exhaustividad en las señas más parecidas | ≥ 0.90 | 0.985 / 0.993 / 1.000 | Cumple |
| Confusión entre pares (M-S, M-N, N-A) | Análisis cualitativo de los errores cruzados entre señas homólogas | Reducir la confusión respecto al MLP y que ningún par domine el error | 13 errores en 2,008 (0.65%); los pares críticos dejan de ser el foco y el error residual se concentra en E (9) y N (3) | Cumple |
| Brecha de equidad entre clases | Máxima diferencia de recall entre letras | < 0.05 | 0.018 | Cumple |

El cuarto KPI se incorporó en esta etapa porque el criterio de auditoría exige detectar si alguna clase queda sistemáticamente castigada. Se calcula como `max(recall) − min(recall)` sobre las 6 letras.

---

## 5. Fuentes de datos y variante

### 5.1 Dataset
**Sign Language MNIST** (Kaggle, `datamunge/sign-language-mnist`): imágenes de 28×28 px en escala de grises, vectorizadas en 784 columnas más la etiqueta.

| Propiedad | Valor |
|---|---|
| Muestras train / test | 27,455 / 7,172 (79% / 21%) |
| Clases | 24 letras (faltan J y Z, señas con movimiento) |
| Valores nulos | 0 |
| Duplicados exactos | 0 dentro de train y 0 dentro de test |
| Imágenes compartidas train ↔ test | **0** (verificado por hash sobre las 27,455 × 7,172 filas) |
| Rango de píxeles | 0–255 (media 159.29, desv. 48.77) |
| Brillo promedio por imagen | 54.93 – 242.55 |
| Imágenes sin variación | 0 |

### 5.2 Variante: subconjunto de 6 letras
Se conservan **A, E, M, N, S y T** (etiquetas originales 0, 4, 12, 13, 18, 19) y se remapean a 0–5 para que coincidan con las 6 neuronas de salida.

**Justificación:** son las señas con menor distancia visual entre sí (mismo puño cerrado, diferencia solo en la posición del pulgar), lo que maximiza la exigencia del problema y permite comparar MLP y CNN sobre el mismo terreno difícil.

**Volumen resultante:** 6,674 imágenes de entrenamiento (24.3% del total) y 2,008 de prueba (28.0%).

| Clase | Train | Test |
|---|---:|---:|
| A | 1,126 | 331 |
| E | 957 | 498 |
| M | 1,055 | 394 |
| N | 1,151 | 291 |
| S | 1,199 | 246 |
| T | 1,186 | 248 |

### 5.3 Trazabilidad y autorización
La pauta de la EP3 recomienda reutilizar el dataset de la Evaluación Parcial 1 y permite que el docente autorice otro conjunto. Este proyecto **mantiene el mismo dataset y la misma variante de la EP1/EP2**, lo que cumple el criterio de continuidad y comparabilidad. La decisión se deja documentada aquí y en el notebook (sección "Variante") para que quede explícita en la defensa.

---

## 6. Metodología (CRISP-DM)

| Fase | Aplicación en el proyecto |
|---|---|
| 1. Comprensión del negocio | Accesibilidad comunicacional para la comunidad sorda; alcance limitado a letras estáticas |
| 2. Comprensión de los datos | EDA general y del subconjunto: balance, iluminación, duplicados, ausencia de J/Z, sesgo de centrado |
| 3. Preparación de datos | Normalización [0,1], one-hot, partición 80/20 estratificada (semilla 123), reshape a `(N, 28, 28, 1)` |
| 4. Modelamiento | MLP y CNN parametrizadas; Grid Search; ablación de regularización |
| 5. Evaluación | Accuracy, Precision, Recall, F1, ROC-AUC, matrices de confusión, curvas de aprendizaje |
| 6. Despliegue | Exportación del modelo (`modelo_final.keras`), tabla de métricas (`metricas_modelos.csv`) y esta documentación |

La iteración entre las fases 4 y 5 es explícita: el resultado de cada experimento (Grid Search y ablación) alimenta la siguiente configuración evaluada.

---

## 7. Pipeline Integrado

### 7.1 Arquitectura

```
[1] Carga (kagglehub) ─► [2] EDA ─► [3] Preprocesamiento ─► [4] Construcción del modelo
                                                                      │
                                                                      ▼
[9] Optimización ◄─ [8] Auditoría ◄─ [7] Evaluación ◄─ [6] Regularización ◄─ [5] Entrenamiento
```

### 7.2 Etapas, componentes y propósito

| # | Etapa | Componente | Propósito |
|---|---|---|---|
| 1 | Carga de datos | `kagglehub.dataset_download` + `pd.read_csv` | Obtener los CSV sin gestión manual de archivos |
| 2 | Análisis exploratorio | Conteo por clase, histogramas de píxeles y brillo, imagen promedio por clase, duplicados | Detectar desbalance, mala iluminación, sesgo de centrado y similitud entre clases antes de modelar |
| 3 | Preprocesamiento | `preparar_datos()` | Normalizar, codificar, estratificar y adaptar la forma de entrada para cada arquitectura |
| 4 | Construcción | `crear_cnn()`, `crear_mlp()` | Encapsular la arquitectura en funciones parametrizadas por regularización, para poder experimentar sin duplicar código |
| 5 | Entrenamiento | `entrenar_y_medir()`, `entrenar_final()` | Unificar compilación, entrenamiento y medición de tiempo; garantizar condiciones comparables |
| 6 | Regularización | Ablación incremental | Aislar el aporte de cada técnica (Dropout → BatchNorm → L2 → aumento de datos) |
| 7 | Evaluación | `evaluar_modelo_completo()`, `metricas_completas()` | Medir con múltiples métricas y graficar curvas y matriz de confusión con el mismo formato |
| 8 | Auditoría | Análisis de errores, equidad por clase, `mapa_gradcam()` | Explicar *por qué* el modelo acierta o falla, no solo cuánto |
| 9 | Optimización | Bucles de Grid Search | Explorar configuraciones y seleccionar con evidencia |

### 7.3 Cómo interactúan las etapas

- La etapa 2 define qué preprocesamiento se necesita: el EDA muestra brillo heterogéneo → normalización + BatchNorm + aumento de datos.
- La etapa 3 entrega dos representaciones de los mismos datos (784 para MLP, 28×28×1 para CNN), lo que hace que la comparación de la etapa 7 sea justa.
- Las etapas 4 y 5 son funciones reutilizables: la etapa 9 las invoca en bucle, y la etapa 6 las invoca con banderas de regularización. Sin esa modularidad, cada experimento requeriría reescribir el modelo.
- La etapa 8 retroalimenta la 9: los errores observados (N→A, M→N) y los mapas de Grad-CAM informan qué revisar en la siguiente configuración.
- La etapa 9 cierra el ciclo devolviendo el modelo final, que se persiste para despliegue.

### 7.4 Evaluación crítica del pipeline

**Fortalezas**
- Modularidad real: cambiar un hiperparámetro o una técnica de regularización no exige tocar el código de evaluación.
- Condiciones comparables: misma semilla, misma partición, mismas métricas y mismo formato de reporte para MLP y CNN.
- Cobertura de auditoría poco habitual: no solo accuracy, sino curvas, matriz de confusión, equidad por clase y XAI.
- Eficiencia: el modelo final tiene 56,902 parámetros y entrena en 55 s, lo que lo hace desplegable en CPU.

**Limitaciones**
- La selección del modelo final se hizo sobre el conjunto de prueba (ver §10.4), lo que introduce un sesgo optimista.
- Una sola semilla por configuración: no hay barras de error ni pruebas de significancia.
- La validación proviene del mismo archivo que el entrenamiento, por lo que `val_accuracy` satura en 1.0 y deja de ser informativa.
- El etiquetado original y el remapeo son pasos manuales que dependen de constantes escritas en el notebook.

**Riesgos**
- **De dominio:** el dataset es homogéneo (fondo limpio, mano centrada, un fenotipo). El desempeño real en cámara será menor.
- **De selección:** optimizar sobre test con 11 configuraciones infla la métrica reportada.
- **De interpretación:** leer el 99.35% como "listo para producción" sería un error; el modelo resuelve 6 de 24 letras estáticas.

**Mejoras para la siguiente iteración**
1. Selección por validación (o validación cruzada) y test reservado para una única evaluación final.
2. Repetición con ≥ 3 semillas y reporte de media ± desviación.
3. Ampliar el espacio de búsqueda a learning rate, batch size y optimizador.
4. Aumento de datos por brillo/contraste y validación con sujetos no vistos.
5. XAI comparativa (Grad-CAM en la CNN vs. importancia de píxeles en el MLP).

---

## 8. Preparación de datos

- **Normalización:** división por 255 → rango [0,1], para estabilizar gradientes.
- **One-hot:** 6 clases → vectores de 6 posiciones, compatible con `categorical_crossentropy` y softmax.
- **Partición:** 5,339 train / 1,335 validación (80/20), estratificada, semilla 123.
- **Conjunto de prueba:** 2,008 imágenes, intactas hasta la evaluación.
- **Adaptación dimensional:** `(N, 784)` para el MLP y `(N, 28, 28, 1)` para la CNN.
- **Reproducibilidad:** semilla fija en `random`, `numpy` y `tensorflow`.

---

## 9. Modelos

### 9.1 MLP
`Dense(neuronas, relu) → Dropout → Dense(neuronas/2, relu) → Dropout → Dense(6, softmax)`
El MLP recibe la imagen aplanada, por lo que pierde la noción de vecindad entre píxeles: no puede distinguir "pulgar arriba" de "pulgar abajo" si los patrones de intensidad son similares.

### 9.2 CNN
`[Aumento de datos opcional] → Conv2D(filtros, 3×3, relu) [→ BatchNorm] → MaxPool → Conv2D(2×filtros, 3×3, relu) [→ BatchNorm] → MaxPool → Flatten → Dense(neuronas, relu) [→ BatchNorm] → Dropout → Dense(6, softmax)`

Decisiones justificadas:
- **Dos bloques convolucionales:** el primero detecta bordes y contornos; el segundo combina esos bordes en formas (uniones de dedos).
- **Filtros duplicados por bloque (16→32):** aumenta la abstracción sin disparar el costo.
- **Sin giros horizontales en el aumento de datos:** reflejar la mano cambia el significado de la seña.
- **Rotación 3%, traslación 8%, zoom 8%:** cubre variaciones de cámara sin deformar la postura.
- **Adam + categorical crossentropy:** convergencia estable sin calibración manual.
- **Batch size 64:** equilibrio entre estabilidad del gradiente y velocidad (84 actualizaciones por época).
- **Early Stopping (`val_loss`, paciencia 5–10, `restore_best_weights`):** evita sobreajuste con solo 5,339 imágenes de entrenamiento.

---

## 10. Optimización de hiperparámetros

### 10.1 Espacio de búsqueda

| Modelo | Hiperparámetros | Valores | Combinaciones |
|---|---|---|---:|
| MLP | neuronas × dropout | {64, 128, 256} × {0.2, 0.5} | 6 |
| CNN | filtros × dropout | {16, 32, 64} × {0.2, 0.5} | 6 |

Cada combinación se entrena 12 épocas sin Early Stopping para comparar velocidad de convergencia en igualdad de condiciones.

### 10.2 Resultados — MLP

| Neuronas | Dropout | val_accuracy |
|---:|---:|---:|
| 256 | 0.2 | 0.9873 |
| 128 | 0.2 | 0.9813 |
| 64 | 0.2 | 0.8562 |
| 256 | 0.5 | 0.7843 |
| 64 | 0.5 | 0.4936 |
| 128 | 0.5 | 0.3790 |

**Lectura:** el MLP es extremadamente sensible al Dropout alto. Apagar el 50% de las conexiones en una red densa destruye el acceso a píxeles específicos y produce subajuste severo (37.9% en el peor caso). Con dropout 0.2, la capacidad sí importa: 256 neuronas es la mejor configuración.

### 10.3 Resultados — CNN

| Filtros | Dropout | val_accuracy |
|---:|---:|---:|
| 16 | 0.2 | 1.0000 |
| 16 | 0.5 | 1.0000 |
| 32 | 0.2 | 1.0000 |
| 32 | 0.5 | 1.0000 |
| 64 | 0.2 | 1.0000 |
| 64 | 0.5 | 1.0000 |

**Lectura:** la CNN satura la validación en todas las configuraciones. La causa es que train y validación provienen del mismo archivo, con los mismos sujetos e iluminación, por lo que la validación deja de discriminar.

**Regla de desempate.** Al no existir diferencia empírica, se aplicó un criterio explícito y determinista, no el orden accidental de la tabla:
1. **Menor número de filtros** — menor costo computacional y menor riesgo de sobreajuste.
2. **A igualdad, menor dropout** — orden ascendente de exploración de la cuadrícula.

Resultado: **16 filtros, dropout 0.2**. Se declara que esta elección **no está respaldada por superioridad empírica** frente a las otras cinco configuraciones. Su validez se verificó después: con 16 filtros, la variante regularizada alcanzó 99.35% en test con 56,902 parámetros, frente a los 121,670 de la CNN base. Además, el modelo definitivo no depende del dropout elegido aquí, porque surge de la ablación de regularización (§11), no de este grid.

### 10.4 Selección del modelo final

La ablación de regularización se comparó por **accuracy en test**, y el modelo final se eligió por **F1 macro en test**. Esto permitió detectar la mejor variante (aumento de datos, 99.35%), pero **introduce un sesgo de selección**: se eligió el máximo de 11 configuraciones evaluadas sobre el mismo conjunto. El valor debe interpretarse como cota optimista. La corrección recomendada (selección por validación) se detalla en §14.

---

## 11. Regularización y su impacto

Se aplicaron las técnicas de forma incremental, agregando una por una sobre la anterior:

| Variante | Dropout | BatchNorm | L2 | Aumento | Train | Val | **Test** | Loss test | Épocas | Tiempo |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| Sin regularización | 0.0 | No | 0 | No | 1.0000 | 1.0000 | 0.9407 | 0.4225 | 30 | 61.9 s |
| + Dropout | 0.5 | No | 0 | No | 1.0000 | 1.0000 | 0.9124 | 0.5093 | 30 | 58.5 s |
| + BatchNorm | 0.5 | Sí | 0 | No | 1.0000 | 1.0000 | 0.9128 | 0.2995 | 29 | 81.2 s |
| + L2 | 0.5 | Sí | 1e-4 | No | 1.0000 | 1.0000 | 0.8914 | 0.5882 | 30 | 93.2 s |
| **+ Aumento de datos** | 0.5 | Sí | 1e-4 | Sí | 0.9978 | 0.9978 | **0.9935** | **0.0333** | 18 | **55.3 s** |

**Análisis por técnica**
- **Dropout:** no mejora test por sí solo en este dataset (94.07% → 91.24%), pero reduce la dependencia de neuronas individuales y es indispensable para que el aumento de datos funcione.
- **Batch Normalization:** mantiene la accuracy (91.28%) y **reduce la pérdida de 0.509 a 0.300**, es decir, mejora la calibración de la confianza. Ese es su aporte real, no el accuracy.
- **L2 (1e-4):** penaliza sin beneficio en test (89.14%) y aumenta el tiempo. Se conserva por robustez, no por métrica.
- **Aumento de datos:** el salto principal. +8.3 puntos de accuracy y **loss 0.033 (−93%)** frente a la variante sin regularización, además de converger en 18 épocas.

**Conclusión de la ablación:** en este problema el factor decisivo no es la capacidad de la red ni la penalización, sino **la variedad de los datos de entrenamiento**. Las curvas de aprendizaje confirman que la brecha train-val se cierra y que el modelo deja de memorizar.

---

## 12. Evaluación y auditoría

### 12.1 Métricas del modelo final

El conjunto de prueba (2,008 imágenes) **no participó del entrenamiento ni de la búsqueda de hiperparámetros**: proviene del CSV de test, nunca entra en `train_test_split` y se verificó que no comparte ninguna imagen con train (§5.1).

| Clase | Recall | F1 | Imágenes |
|---|---:|---:|---:|
| A | 0.9970 | 0.9985 | 331 |
| E | 0.9819 | 0.9909 | 498 |
| M | 1.0000 | 0.9850 | 394 |
| N | 0.9897 | 0.9931 | 291 |
| S | 1.0000 | 1.0000 | 246 |
| T | 1.0000 | 1.0000 | 248 |

Aciertos: 1,995 de 2,008. Errores: 13 (0.65%).

### 12.2 Matriz de confusión y análisis de errores

- Los errores residuales se concentran en **E** (9 casos), atribuibles a dedos flexionados sobre el pulgar a 28×28 px.
- Los falsos positivos de M provienen de E y N: el límite entre "cubrir tres dedos" y "cubrir dos" se difumina a baja resolución.
- No queda ninguna clase actuando como "imán" de falsos positivos, a diferencia de la CNN base donde S absorbía errores de otras letras.

### 12.3 XAI — Grad-CAM

Se implementó Grad-CAM manual sobre la última capa convolucional (`conv2d_25`): se calcula el gradiente de la clase predicha respecto al mapa de activación, se promedian los pesos por canal y se genera un mapa de calor normalizado.

**Hallazgo:** las activaciones se concentran en los **dedos flexionados y la posición del pulgar**, no en el fondo ni en los bordes del encuadre. Esto respalda que el modelo aprendió la geometría de la seña y no una coincidencia del dataset.

**Limitación:** Grad-CAM se aplicó solo a la CNN. El MLP no admite mapas de activación espaciales; para comparar explicabilidad habría que usar importancia de píxeles o SHAP.

---

## 13. Comparación MLP vs CNN

| Dimensión | MLP optimizada | CNN + aumento | Diferencia |
|---|---:|---:|---:|
| Accuracy test | 81.62% | 99.35% | +17.73 pp |
| F1 macro | 0.7952 | 0.9946 | +0.1994 |
| ROC-AUC | 0.9581 | 1.0000 | +0.0419 |
| Loss test | 0.7446 | 0.0333 | −0.7113 |
| Parámetros | 234,630 | 56,902 | −76% |
| Tiempo de entrenamiento | 18.5 s | 55.3 s | +36.8 s |
| Inferencia | — | 0.20 ms/imagen | — |
| Interpretabilidad | Baja (sin noción espacial) | Alta (Grad-CAM sobre regiones de la mano) | — |
| F1 clase N (la más difícil) | ~0.64 (MLP base) | 0.9931 | — |

**Análisis**
- La CNN supera al MLP en todas las métricas de clasificación, con **menos de la cuarta parte de los parámetros**. La ventaja no viene del tamaño, sino de la **indisponibilidad de sesgo inductivo espacial** en el MLP: los filtros comparten pesos y preservan vecindad, lo que permite detectar la posición del pulgar y los nudillos.
- El MLP tiene un techo estructural: aunque se optimice, su `loss_test` sigue alto (0.7446), señal de predicciones poco confiables incluso cuando acierta.
- La CNN es más costosa de entrenar (55–195 s vs 15–18 s), pero el costo es irrelevante frente a la ganancia de 17.7 puntos y su inferencia es más rápida que un ciclo de cámara.
- En interpretabilidad la CNN también gana: Grad-CAM permite verificar *dónde* mira la red, algo imposible en un MLP que solo ve intensidades globales.

**Decisión:** se selecciona la **CNN con aumento de datos** como arquitectura final.

---

## 14. Discusión crítica del resultado (99.35%) y sesgos del modelo

### 14.1 Por qué el 99.35% no es extrapolable

El conjunto de prueba comparte con el entrenamiento el **proceso de captura**: fondo limpio, mano centrada, iluminación controlada y un fenotipo predominante. Un modelo entrenado y evaluado en ese dominio mide *consistencia interna*, no desempeño en producción.

Tres factores inflan la cifra:
1. **Sesgo de selección:** es el máximo de 11 configuraciones evaluadas sobre el mismo test (§10.4).
2. **Ausencia de metadatos:** sin identificador de sujeto ni de sesión no se puede verificar que test sea realmente independiente, y no se puede hacer validación por sujeto.
3. **Homogeneidad del dataset:** no hay variedad de tono de piel, fondo, distancia o ángulo de cámara.

**Expectativa razonable:** en una cámara real, con fondo ruidoso y mano descentrada, se espera una caída significativa del accuracy. La comparación MLP vs CNN, en cambio, sí es sólida: la ventaja relativa se mantiene porque proviene del sesgo inductivo, no del ajuste fino.

**Variabilidad entre corridas.** La misma arquitectura CNN base, entrenada sobre los mismos datos en dos versiones del proyecto, obtuvo **94.42%** en la EP2 y **91.09%** en la EP3: una diferencia de **3.3 puntos** atribuible solo a la inicialización y al orden de los lotes. Ese es el margen de ruido real de una corrida única. Como la brecha entre el modelo con aumento de datos (99.35%) y el mejor sin aumento (94.07%) es de 5.3 puntos, la conclusión cualitativa se sostiene, pero la cifra exacta no debe citarse con precisión de centésimas.

### 14.2 Sesgos en los datos

| Sesgo | Evidencia | Mitigación aplicada | Mitigación pendiente |
|---|---|---|---|
| Iluminación | Brillo por imagen entre 54.9 y 242.6 | Normalización, BatchNorm | Aumento por brillo/contraste |
| Centrado de la mano | Imágenes promedio muestran la mano siempre al centro | Aumento con traslación y zoom | Dataset con manos descentradas |
| Fenotipo | Un solo tono de piel predominante | — | Dataset diverso en tono de piel |
| Señas dinámicas | Sin J ni Z | Alcance declarado | Arquitectura secuencial (RNN/video) |
| Balance de clases | 957–1,199 por clase en train | Partición estratificada | — (no requiere acción) |

### 14.3 Sesgos en los resultados del modelo

- **Equidad por clase:** el recall va de 0.982 (E) a 1.000 (M, S, T). La brecha es 0.018, muy por debajo de la meta de 0.05, por lo que ninguna letra queda sistemáticamente castigada. E es la única clase con errores recurrentes, y su causa es geométrica (dedos sobre el pulgar a baja resolución), no de representación.
- **Sesgo por subgrupo (no medible hoy):** no se puede estratificar el error por tono de piel, iluminación o posición porque el dataset no incluye esos atributos. Una auditoría seria requeriría etiquetar el conjunto o construir un set de validación externo con esas condiciones. Es la brecha de auditoría más importante del proyecto y se declara como limitación, no se omite.
- **Calibración:** la pérdida final (0.033) indica predicciones con alta confianza y bajo error. Sin embargo, un modelo sobreajustado al dominio puede ser **sobreconfiado fuera de él**; no se midió calibración en datos externos.

### 14.4 Consecuencia ética

En un contexto crítico (salud, emergencias) un falso negativo tiene costo alto. El modelo debe operar como **herramienta de apoyo**, con un umbral de confianza conservador y derivación a intérprete humano cuando la probabilidad sea baja. También requiere consentimiento y protocolo de privacidad para la captura de imágenes.

---

## 15. Conclusiones

1. **La CNN es la arquitectura correcta para este problema.** Supera al MLP en accuracy (+17.73 pp), F1 macro (+0.20) y ROC-AUC, con 76% menos parámetros. La causa es el sesgo inductivo espacial, no la capacidad.
2. **La regularización que importa es la que agrega variedad, no la que penaliza.** El aumento de datos produjo el salto (94.07% → 99.35%) y redujo la pérdida en 93%; L2 y Dropout aislados no mejoraron el test.
3. **La optimización fue útil pero limitada por el diseño de la validación.** El Grid Search encontró la mejor configuración del MLP, pero en la CNN la validación saturó y la elección fue por parsimonia.
4. **Se cumplieron los cuatro KPIs**, incluida la equidad por clase (brecha 0.018).
5. **El 99.35% es una cota optimista**, no un resultado de producción. El valor del proyecto está en la comparación controlada y en la auditoría, no en el número final.

---

## 16. Estructura del repositorio y reproducción

```
.
├── README.md
├── notebooks/
│   └── TLY1102_EV3.ipynb
├── models/
│   ├── modelo_final.keras
│   └── metricas_modelos.csv
├── images/
│   └── (figuras exportadas del notebook)
└── data/
    └── (descarga automática vía kagglehub)
```

**Reproducción**
1. Abrir `notebooks/TLY1102_EV3.ipynb` en Google Colab.
2. `Entorno de ejecución → Ejecutar todo` (el dataset se descarga automáticamente con `kagglehub`, es público y no requiere credenciales).
3. Dependencias: `tensorflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `kagglehub`.
4. Tiempo total aproximado en Colab CPU: 15–25 minutos (entrenamientos incluidos).
5. Salidas: `modelo_final.keras` y `metricas_modelos.csv`.

**Nota:** los archivos `modelo_final.keras` y `metricas_modelos.csv` deben moverse a `models/` una vez generados.

---

## 17. Referencias

- Kaggle. *Sign Language MNIST*. https://www.kaggle.com/datasets/datamunge/sign-language-mnist
- Selvaraju, R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV.
- Chollet, F. *Deep Learning with Python*. Manning.
- Documentación oficial de Keras y TensorFlow. https://keras.io
- Wirth, R. y Hipp, J. (2000). *CRISP-DM: Towards a Standard Process Model for Data Mining*. SIGKDD.
- Materiales de la asignatura TLY1102, Experiencia 3.
