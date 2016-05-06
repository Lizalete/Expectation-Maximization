# Expectation-Maximization
Expectation Maximization in Python

INTRODUCCIÓN
El algoritmo de maximización de esperanza se usa en estadística para encontrar estimadores de máxima 
verosimilitud de parámetros en modelos probabilísticos que dependen de variables no observables. El algoritmo 
computa la esperanza de la verosimilitud mediante la inclusión de variables latentes, y después se computan 
estimadores de máxima verosimilitud de los parámetros mediante la maximización de la verosimilitud esperada (Wikiwand, 2016).

CÓDIGO EN PYTHON
De un ejemplo de implementación con un toolbox que recibía dos conjuntos, 
se modificó para que pudiera recibir tres. Los datos se extrajeron de tres bases de datos de tres 
diferentes tipos de llantos: normal, asfixia y sordera.

Se obtienen las primeras 10 columnas de cada uno de los llantos para sacar la media y covarianza para graficar 
su histograma (Imagen 4a, 5a, 6a) y gaussiana (Imagen 4b, 5b, 6b) mediante la función normpdf de la librería matplotlib.mlab 

Se hace la estimación de los nuevos parámetros de media y covarianza mediante la función de responsabilidad (gamma) 
y así graficar la nueva gaussiana 

Al final, se juntan las tres gaussianas de los datos originales y 
se comparan con las gaussianas de los datos obtenidos en el programa 
