Nuestro modelo de aprendizaje inspirado en uno de los modelos encontrados en kaggle implementa una red U-Net para la segmentación de imágenes siguiendo los siguientes pasos:

- Define la arquitectura U-Net con capas de convolución, max-pooling, upsampling y concatenación.
    - La función get_unet0 define la arquitectura U-Net específica con profundidad de 5 niveles. Utiliza bloques de (convolución + normalización por lotes + ELU) x 2 en cada nivel. La arquitectura de la red se compone de una serie de capas convolucionales, cada una seguida de una normalización de lotes y una función de activación ELU. Las capas convolucionales se utilizan para extraer características de la imagen, y se utilizan maxpoolings para reducir el tamaño de la imagen y la cantidad de características. La red también incluye capas de convolución transpuesta y concatenaciones para reconstruir la imagen segmentada a partir de las características extraídas anteriormente.

   - El modelo de salida de la red es un tensor que representa la imagen segmentada. La capa de salida utiliza una función de activación sigmoide para producir valores de píxeles en el rango de 0 a 1, que se pueden interpretar como probabilidades de que un píxel pertenezca a una de las máscaras binarias.

- Por que utilizamos U-Net?

    La razón principal por la que utilizamos U-Net para esta tarea de segmentar imágenes es porque es una de las mejores arquitecturas a la hora de segmentar imágenes:

    - Arquitectura de encoder-decoder: la U-Net tiene una estructura simétrica de encoder-decoder que permite capturar información de contexto de la imagen en diferentes escalas y luego combinarla en la etapa de decodificación para obtener una segmentación precisa.

    - Conexiones de skip: la U-Net utiliza conexiones directas de skip entre las capas de encoder y decoder, lo que permite la propagación de información detallada y de alta resolución desde la etapa de encoder a la de decoder. Esto ayuda a preservar las características de la imagen original y mejorar la precisión de la segmentación.

    - Pocas capas convolucionales: la U-Net utiliza pocas capas convolucionales, lo que reduce el riesgo de overfitting, especialmente en conjuntos de datos pequeños. Como en este caso contamos con un dataset de solo 25 imágenes, U-Net fue una buena alternativa.


- Por qué se utilizó ELU en lugar de RELU?
    - Se utilizó la función ELU en lugar de la función ReLU porque ELU tiene una salida suave para valores negativos de entrada, lo que puede ayudar a prevenir algunos problemas de la función ReLU, como neuronas "muertas" (cuando una neurona nunca se activa porque la entrada siempre es negativa) y gradientes inestables en la retropropagación del error. Además, la función ELU puede proporcionar una mejor precisión de la red neuronal, especialmente en tareas de clasificación de imágenes. Por otro lado, la función ReLU es más simple y computacionalmente más eficiente por lo que para redes grandes proporciona un entrenamiento más rápido y mejorar la capacidad de generalización de la red.


- Se definen dos funciones para el aumento de datos (data augmentation) y para la formación de lotes (batch formation) para la segmentación de imágenes con la arquitectura U-Net.


    - La función generadora `batch_generator` utiliza las funciones `flip_axis` y `form_batch` para formar lotes de datos de entrenamiento y sus correspondientes máscaras de segmentación. La función utiliza la técnica de aumento de datos (data augmentation) para aumentar la variabilidad en los datos de entrenamiento y reducir la dependencia de la red en la información específica del conjunto de datos y así mejorar el proceso de entrenamiento y disminuir el riesgo de overfitting.

    - Además, la función ha sido diseñada para ser thread-safe lo que significa que se pueden correr múltiples hilos de ejecución simultáneamente sin causar problemas de sincronización o de acceso a datos compartidos, garantizándose su comportamiento predecible y consistente, incluso cuando varios hilos de ejecución acceden a ella simultáneamente.
    
     
- Se utiliza la métrica de evaluación coeficiente de Jaccard que mide la similitud entre dos conjuntos de datos binarios. Toma dos argumentos, y_true y y_pred, que son los valores verdaderos y predichos, respectivamente, de los conjuntos binarios que se compararán. Primero se calcula la intersección entre los conjuntos y_true y y_pred, y luego calcula la suma de ambos conjuntos. Utiliza la función jaccard_coef para calcular el coeficiente de Jaccard entre y_true y y_pred y luego combina este valor con la función binary_crossentropy para obtener una medida de la diferencia entre las dos segmentaciones. 



- Por qué se disminuyó la dimensión de las convoluciones?

    - Esto puede ayudar a reducir la cantidad de parámetros necesarios en la red neuronal, lo que reduce la cantidad de cálculos necesarios por unidad de datos de entrada, lo que puede reducir el tiempo de entrenamiento y la complejidad computacionaly también puede ayudar a reducir el riesgo de overfitting.

    -  La reducción de la dimensión de las convoluciones también puede aumentar la invariancia espacial de la red neuronal. Esto se debe a que las convoluciones con un stride mayor que 1 o las capas de pooling pueden hacer que la red neuronal sea menos sensible a pequeñas variaciones en la posición de las características en las imágenes de entrada. Esto puede ayudar a mejorar la capacidad de generalización de la red neuronal.


- Intentamos hacer drop out? Dio peores o mejores resultados?


    - No se hizo Drop Out por no ser necesario dropout, porque en este caso el riesgo es de underfitting por pocos datos, para eso se hizo batch_generator, y todos los pasos anteriores que se hicieron fueron para evitar overfitting, no era necesario en este caso, sobre todo porque ELU ya resuelve la parte de lidiar con las neuronas muertas algo q dropout hace pero de forma más random.
    

