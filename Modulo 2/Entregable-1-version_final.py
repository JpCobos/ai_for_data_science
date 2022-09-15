# Entregable 1: Implementación de una técnica de aprendizaje máquina sin el
#               uso de un framework.

# Nombre: Jose Pablo Cobos Austria
# Matricula: A01274631

# Objetivo: Implementar de forma manual un algoritmo de machine learning
#           sin hacer uso de librerias o frameworks de estadistica
#           avanzada o machine learning

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
#   Se uso como base el codigo visto en clase para su elaboracion 


# Importar librerias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


__errors__ = []  # Variable global donde se guardara los errores 

# Funcion de Hipótesis:
#  Esta función lo que nos va permitir es poder evaluar la 

# Ademas, basandonos en la regresion lienar, generaremos una funcion de 
# hipotesis establecida como h(x), haciendo uso de las variables iniciales 
# de la ecuacion, la pendiente (m) y la ordenada al origen (b) , nuestros datos 
# y nos devolvera un la evaluacion de la ecuacion 
def hipotesis(parametros, ind_var):
	"""
	    Se crea una funcion de hipotesis establecida como h(x), 
		que hara uso de las variables iniciales definidas, y se ira actualizando 
		conforme avance el modelo 
    """
	acum = 0
	for i in range(len(parametros)):
		acum += parametros[i] * ind_var[i]  # h(x) = a+bx1+cx2+ ... nxn.
	return acum



# Funcion de Costo por MSE
#  Nuestra funcion de coste es la funcion que queremos optimizar, esto lo haremos 
#  gracias a el MSE que el modelo tratara de dismuir conforme avancen las epocas
#  (se hablara de ellas mas adelante). Ademas este MSE nos permite analizara la diferencia
#  que se tiene entre la y que nosotros predecimos y la y actual, por lo que mientras 
#  menos sea mejor desempeño tendra nuestro modelo
def meansqerror(parametros, ind_var, vars_dep):
	"""
        Se va acumulando los errores calculados, y los evalua en la funcion de 
		hipotesis anteriormente establecida
		ind_var son las varibales independiente y vars_dep son las dependientes
    """
	error_acum = 0

	for i in range(len(ind_var)):
		hip = hipotesis(parametros, ind_var[i])
		error_acum =+ (hip - vars_dep[i])**2 # Función de costo MSE 
	
	__errors__.append( error_acum / len(ind_var) ) 




# Funcion de Gradiente Descendient: 
#  El Gradiente Descendiente tiene la funcion de ir calculando de forma interativa (ciclos de epocas)
#  cual es el mejor valor posible para nuestras variables y de esa forma influyendo de igual 
#  manera en el MSE. Entre sus parametros tenemso Learning rate que es la velocida con la que 
#  aprende nuestro modelo e influye en el cambio de las variables para ir ajustando la recta 
def GradDes(parametros, ind_var, vars_dep, Lr):
	"""
		Esta funcion recibe los ultimos parametros actuales, tambien todas las variables, 
		dependiente e independientes para realizar el analisis del error y el 
		Learning Rate que utilizara para los calculos 
	"""
	
	gd_list = list(parametros) # Una lista de los parametros que se va a ir actualizand
	
	for j in range(len(parametros)):		
		acum = 0
		
		for i in range(len(ind_var)):
			error = hipotesis(parametros, ind_var[i]) - vars_dep[i]
			acum = acum + error * ind_var[i][j]  # El acumulador  
		gd_list[j] = parametros[j] - Lr*(1/len(ind_var))*acum  # Actuliza los parametros
	
	return gd_list

# Con nuestras funciones para la regresion hechas, lo que sigue es importar los datos de nuestro 
# dataset mediante la libreria de pandas 

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
# se pueden usar sin problema alguno y para evitar que sea complicar el modelo, 
# solo usaremos una variable dependiente "la producción de energía 
# eléctrica neta por hora (EP)" y las independientes " Vacío de escape (V) " y la
# " Presión ambiental (AP) "

# Importar dataset que utilizaremos 
df_main = pd.read_csv("combined_cycle_power_plant_data_set.csv")

# Seleccionaremos las variables que utilizaremos, del nuestro dataframe y los convierte 
# a un arreglo 
x = df_main.iloc[:, 1:3].to_numpy().tolist() # Seleccionamos las columnas de V y AP
#print(df_main.iloc[:, 1:3].to_numpy().tolist())
y = df_main["PE"].to_numpy().tolist() # Y seleccionamos la columna de los datos que 
									  # queremos predecir 


# Ejemplo base de uso del modelo en modo multivariable
# parametros = [0,0,0]
# ind_var = [[1,1],[2,2],[3,3],[4,4],[5,5]]
# vars_dep = [2,4,6,8,10]

parametros = [0,0,0] # Establecer los ceros por variable.
# 					   ej: [0, 0] = a, b | [0, 0, 0] = a, b, c
ind_var = x # Creamos nuestra variables independientes
vars_dep = y # Creamos nuestra variables dependientes
# Como mencionamos anteriormente, el learning rate es la velocidad con la que 
# nuestro modelo aprende 
Lr = 0.000001  

# Añadir un (1) como primer elemento de cada conjunto de parámetros.
for i in range(len(ind_var)):
	if isinstance(ind_var[i], list):
		ind_var[i]=  [1]+ind_var[i]
		# Comprobación de los arreglos (debug)
		# print(ind_var)
	else:
		ind_var[i]=  [1,ind_var[i]]



# Declaramos dos variables, la de las epocas que se van a realizar y la otra el valor 
# de la eopca que cambiara conforme avance el ciclo 
epoca_final = 1000
epoca = 0 

while True:  # Se corre los gradiantes descendentes hasta lograr el valor minimo 
	oldparams = list(parametros)
	
	parametros = GradDes(parametros, ind_var,vars_dep, Lr) # Se ejecuta el modelo de gradiente descendiente
	meansqerror(parametros, ind_var, vars_dep)             # Se ejecuta el la funcion de MSE 
	epoca = epoca + 1 			  				 		   # Se actuliza el vlaor de la epoa 
	print ("Epoca número: "+str(epoca)+" y el valor del MSE: "+str(__errors__[-1]*100))
	if(oldparams == parametros or epoca == epoca_final):
		break

print ("Ecuacion final obtenida: "+"\n")
print("y = " + str(parametros[0]) + " + " + str(parametros[1])+ "x1"+ " + "+ str(parametros[2]) +"x2\n")
print("Error: "+str(__errors__[-1]*100))
# Imprimimos una grafica donde se puede visulaizar el cambio del MSE conforme avanzan 
# las epocas 
plt.plot(__errors__, color = 'blue', label='Error')
plt.legend(loc='upper left')
plt.xlabel("Epocas")
plt.ylabel("Min Squared Error")
plt.title("Cambio del MSE vs epocas")
plt.show()

# Y como podemos observar en la grafica, realmente el error solo se va ajustando
# al final en cambios muy pequeños