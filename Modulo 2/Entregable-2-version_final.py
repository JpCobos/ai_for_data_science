# Entregable 1: Implementación de una técnica de aprendizaje máquina sin el
#               uso de un framework.

# Nombre: Jose Pablo Cobos Austria
# Matricula: A01274631

# Objetivo: Implementar de forma manual un algoritmo de machine learning
#           sin hacer uso de librerias o frameworks de estadistica
#           avanzada o machine learning

# Del entregable pasado retomaremos las definciones principales vistas 
# anteriormente 

# Regresion Lineal:
#   Podemos definirla como el modelo matemático que trata de encontrar la
#   relación entre variables ajustando una ecuación lineal a los datos
#   observados, teniendo variable explicativas y una variable dependiente.

#   Ya con la definicion establecida tenemos que saber recordar que la
#   la ecuacion de una línea recta esta definido por: y = m * x + b donde:
#   y es la variable dependiente
#   m es la pendiente de la ecuación
#   x es la variable independiente
#   b es la ordenada al origen

#   Entonces el objetivo de nuestra regresion lineal es ir ajustando los
#   valores de m y b para representar de la mejor manera posible la
#   relacion entre estas dos variables

# Además, el mismo dataset utilziado en el entregable anterior,
# con la diferencia de que ahora haremos uso de más variables 

# Importamos las librerias 
import imp
import pandas as pd
import numpy  as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt

# Importamos nuestro Dataset
df_main = pd.read_csv("combined_cycle_power_plant_data_set.csv")

# Este dataset fue obtenido de la siguiente 
# liga: https://archive.ics.uci.edu/ml/datasets/combined+cycle+power+plant 

# El conjunto de datos contiene 9568 puntos de datos 
# recopilados de una central eléctrica de ciclo combinado 
# durante 6 años (2006-2011), cuando la planta estaba configurada
# para funcionar a plena carga.
# Las características consisten en variables ambientales 
# promedio por hora Temperatura (T), Presión ambiental (AP), 
# Humedad relativa (RH) y Vacío de escape (V) para predecir 
# la producción de energía eléctrica neta por hora (EP) de la planta.

# El documento que se nos da es un Excel que tiene varias hojas de diferente 
# información sin embargo se extrajo nada más una de esas hojas que contiene la 
# información utilizada en este modelo

# Además, se nos indica que está que los datos ya están limpios y
# se pueden usar sin problema alguno.

# Imprimimos los datos obtenidos.
print("Información del dataset: ")
print(df_main.head())

# Para nuestras x del modelo lo que vamos a hacer va a ser dropear 
# una columna Qué es la que se va a tratar de predecir 
# y esa misma columna será nuestra nuestro valor a predecir 
x = df_main.drop(["PE"],axis = 1).values
y = df_main["PE"].values

print("\n")
# Imprimimos tanto nuestras x como las y para ver que los datos 
# sean los correspondientes.
print("Variables independiente: ")
print(x)
print("Variable dependiente: ")
print(y)
print("\n")
# A continuación hacemos uso de la función train_test_split con 
# el fin de dividir nuestros datos entre datos entrenamiento y 
# datos de Test además en esta misma función el test_size es el 
# porcentaje de datos prueba 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.3, random_state=0) # Seleccionamos el 30%

# Ahora soy finalizado nuestro manejo de datos, 
# lo que sigue es de a empezar con nuestra regresión lineal.
# Afortunadamente, una de las librerías que importamos,la sklearn, 
# ya trae incluido un modelo de regresión lineal dónde solamente 
# le tenemos que meter los datos a diferencia de cómo lo hicimos
# manual en el entregarle anterior que utilizabamos Gradiente 
# Descenediente por lo que es más sencillo lo que pueda manejar más variables.
lr = LinearRegression()

# Aquí entrenamos nuestro modelo utilizando .fit(x,y)
lr.fit(x_train,y_train)

# Ya después de haber entrenado nuestro modelo, 
# de la misma manera gracias a sklearn, 
# podemos encontrar tanto nuestra pendiente "b"
# como los coeficientes "c" de nuestra ecuación.
b = lr.intercept_ # b
c = lr.coef_      # c

print("Pendiente de la ecuación: ", b)
print("Los coeficientes de la ecuacacion: ", c, "\n")

# A continuacion realizaremos prediciones tanto con nuestros datos de prueba como 
# con los datos de entrenamiento 
y_pred_test = lr.predict(x_test)
y_pred_train = lr.predict(x_train)


# Coeficiente de determinación (R cuadrado): 
#  Podemos definir al coeficiente de determinación como la proporción
#  de la varianza total de la variable explicada por nuestra
#  regresión, su rango de valor va de 0 a 1:
#  Más cerca del 0 = nuestro modelo no es confiable 
#  Más cerca del 1 = nuestro modelo es confiable 

# Imprimos el coeficiente de determinacion para poder comprobar la confiabilidad de
# nuestro modelo con los datos train y lois datos test 
print("R^2 con datos test = "+str(r2_score(y_test,y_pred_test)))
print("R^2 con datos train = "+str(r2_score(y_train,y_pred_train)))

# Ploteamos nuestros resultados obtenidos de nuestro modelo vs los 
# datos reales
plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred_test)
plt.xlabel("Actual")
plt.ylabel("Predecido")
plt.title("Actual vs. Predecido")
plt.show()

# Y de esta grafica lo que podemos ver es que los punto como estan agrupados,
# y tienen un como compotamiento como el de una recta, significa que nuestro 
# modelo funciona correctamente 

# Y finalmente creamos un dataframe qué incluye tanto los valores actuales 
# como los valores parecidos y la diferencia que hay entre los dos para 
# poder visualizar mejor cómo es que nos fue con la regresión.
pred_y_df = pd.DataFrame({"Valores Actuales":y_test,"Valores Predecidos":y_pred_test,"Diferencia": y_test-y_pred_test})
print(pred_y_df.head())





 

