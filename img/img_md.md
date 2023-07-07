### RGB

En la primera imagen se puede apreciar como modelo favorece más al recobrado que a la precisión, encontrando bien a todos los pixeles que pertenecen a copas de los árboles y pero confundiendo a los edificios.

<!-- ###### Ruido ciudad -->

![Ruido de ciudad](new_rgb\new_new_rgb_31_ruido_ciudad.png "Ruido de ciudad")


###### Árbol borde y árbol centro

En la segunda y tercera imagen se puede notar como el modelo se le dificulta reconocer más las imágenes en los extremos. No encontrando ningún pixel en el borde y encontrando uno cuando se centra más la imágen. Esto es un problema conocido del modelo.

![Árbol borde](new_rgb\new_new_rgb_31_arbol_borde.png "Árbol borde")


<!-- ###### Árbol centro -->
![Árbol centro](new_rgb\new_new_rgb_31_arbol_centro.png "Árbol centro")

Tambien la mala calidad en imágenes rurales se fundamenta por el gran desbalance que hay en el conjunto de entrenamiento, como se puede notar en las siguientes imágenes.

###### Árbol desierto ciudad
![Árbol desierto ciudad](new_rgb\new_new_rgb_31_desierto_ciudad.png "Árbol desierto ciudad")

###### Árbol desierto no ciudad
![Árbol desierto no ciudad](new_rgb\new_new_rgb_31_desierto_no_ciudad.png "Árbol desierto no ciudad")

Como se puede notar se ve afectado con el ruido de las ciudades, pero en el desierto obtiene una segmentación casi perfecta.

### Fisrt Jacc

Al observar las imágenes y notar que no se reconoce ningún pixel se puede decir que el modelo no aprendió. 

###### Ruido ciudad
![Ruido de ciudad](first_jacc\new_first_jacc_99_ruido_ciudad.png)

###### Árbol borde
![Árbol borde](first_jacc\new_first_jacc_99_arbol_borde.png "Ruido de ciudad")


###### Árbol centro
![Árbol centro](first_jacc\new_first_jacc_99_arbol_centro.png "Árbol centro")


###### Árbol desierto ciudad
![Árbol desierto ciudad](first_jacc\new_first_jacc_99_desierto_ciudad.png "Árbol desierto ciudad")


###### Árbol desierto no ciudad
![Árbol desierto no ciudad](first_jacc\new_first_jacc_99_desierto_no_ciudad.png "Árbol desierto no ciudad")


### Jacc


###### Ruido ciudad

En esta imagen se puede apreciar como modelo favorece más al recobrado que a la precisión, encontrando bien a todos los píxeles que pertenecen a copas de los árboles y pero confundiendo a los edificios.

![Ruido de ciudad](new_jacc\new_new_jacc_99_ruido_ciudad.png "Ruido de ciudad")


###### Árbol borde y árbol centro

En la segunda y tercera imagen se puede notar como el modelo se le dificulta reconocer más las imágenes en los extremos, no detecta ningún pixel de árbol, solo da falsos positivos. No encontrando ningún pixel en el borde ni en el centro. 

![Árbol borde](new_jacc\new_new_jacc_99_arbol_borde.png "Árbol borde")

<!-- ###### Árbol centro -->
![Árbol centro](new_jacc\new_new_jacc_99_arbol_centro.png "Árbol centro")


###### Árbol desierto ciudad

En esta imagen se ve como detecta los árboles pero la precisión se ve afectada por el ruido de ciudad, . 
![Árbol desierto ciudad](new_jacc\new_new_jacc_99_desierto_ciudad.png "Árbol desierto ciudad")


###### Árbol desierto no ciudad

En el caso de las imágenes que no son de ciudades se puede considerar que hubo buen recobrado y precisión

![Árbol desierto no ciudad](new_jacc\new_new_jacc_99_desierto_no_ciudad.png "Árbol desierto no ciudad")

### Jacc NIR


###### Ruido ciudad

En la siguiente imagen no reconoce todos los pixeles que son copa de árboles que hay en esta, la precisión al igual que en los otros experimentos se ve afectada por el ruido de ciudad.

![Ruido de ciudad](new_jacc_nir\new_new_jacc_nir_99_ruido_ciudad.png "Ruido de ciudad")



###### Árbol borde y Árbol centro

En las siguientes dos imágenes detecta como árbol prácticamente toda la imagen, por lo que hay uy mala precisión.

![Árbol borde](new_jacc_nir\new_new_jacc_nir_99_arbol_borde.png "Árbol borde")

<!-- ###### Árbol centro -->
![Árbol centro](new_jacc_nir\new_new_jacc_nir_99_arbol_centro.png "Árbol centro")


###### Árbol desierto ciudad y  Árbol desierto no ciudad


En las siguientes imágenes no detecta ningún árbol. 

![Árbol desierto ciudad](new_jacc_nir\new_new_jacc_nir_99_desierto_ciudad.png "Árbol desierto ciudad")


<!-- ###### Árbol desierto no ciudad -->
![Árbol desierto no ciudad](new_jacc_nir\new_new_jacc_nir_99_desierto_no_ciudad.png "Árbol desierto no ciudad")


### NIR


###### Ruido ciudad

En esta imagen reconoce los árboles que hay, pero la precisón se ve afectada por el ruido de ciudad.

![Ruido de ciudad](new_nir\new_new_nir_31_ruido_ciudad.png "Ruido de ciudad")

###### Árbol borde y Árbol centro

En las siguinetes imágenes no detecta ningún pixel como árbol, ni del borde, ni del centro del mismo. 

![Árbol borde](new_nir\new_new_nir_31_arbol_borde.png "Árbol borde")


<!-- ###### Árbol centro -->
![Árbol centro](new_nir\new_new_nir_31_arbol_centro.png "Árbol centro")


###### Árbol desierto ciudad

En la siguiente imagen reconoce todos los pixeles que son de árbol pero la precisión se afectada por el ruido de ciudad.

![Árbol desierto ciudad](new_nir\new_new_nir_31_desierto_ciudad.png "Árbol desierto ciudad")

 

###### Árbol desierto no ciudad

En la siguiente imagen se tiene mejor precisión que en la anterior, pero algunos áreas de pixeles de árbol más pequeñas no las detecta. 

![Árbol desierto no ciudad](new_nir\new_new_nir_31_desierto_no_ciudad.png "Árbol desierto no ciudad")

