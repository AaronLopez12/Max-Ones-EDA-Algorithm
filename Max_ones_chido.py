import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.insert(1, '../Funciones-objetivo/')
from Func_genetic_algorithms import MaxOnesFunction

def Crear_Matriz(Individuos, Caracteristicas):
	"""
	Creacion de una matriz de poblacion para los algoritmos
	EDA. 
	"""
	Matriz = np.zeros((Individuos, Caracteristicas + 1))

	for i in range(Individuos):
		for j in range(Caracteristicas):
			Matriz[i][j] = np.random.choice(range(2))
	return  Matriz

Iteraciones   = 5
N_Individuos  = 10
Seleccion_ind = 0.5*(N_Individuos)
Dimension	  = 30

Poblacion_Inicial = Crear_Matriz(N_Individuos,Dimension)

#########################
### Inicio del ciclo ####

for i in range(Iteraciones):

	# Evaluacion de la población:
	for ii in range(N_Individuos):
		Poblacion_Inicial[ii, -1] = MaxOnesFunction(Poblacion_Inicial[ii, 0:int(Dimension)])
	
	# Reordenamiento de la matriz,  reverse = True para problemas de maximizacion
	Poblacion_Inicial = np.array(sorted(Poblacion_Inicial, key = lambda x: x[-1], reverse = True))
	
	# Seleccion de los individuos:
	Poblacion_Seleccionada = np.zeros((int(Seleccion_ind), Dimension + 1))
	Poblacion_Seleccionada = Poblacion_Inicial[0:int(Seleccion_ind)]
	
	# Estimacion de media y desviación:
	Parametros = np.array([np.mean(Poblacion_Seleccionada[:,i]) for i in range(Dimension) ])
	
	# Creacion de nueva poblacion:
	Nueva_Poblacion = np.zeros((N_Individuos, Dimension + 1))

	# Creacion de la poblacion basado en los parámetros calculados
	for jj in range(N_Individuos):
		for kk in range(Dimension):
			Nueva_Poblacion[jj,kk] =  1 if np.random.uniform() < Parametros[kk] else 0 
	
	# Evaluacion de la población:
	for ll in range(N_Individuos):
		Nueva_Poblacion[ll, -1] = MaxOnesFunction(Nueva_Poblacion[ll, 0:int(Dimension)])

	# Renombre de la nueva generacion de individuos
	Poblacion_Inicial = Nueva_Poblacion.copy()
	print(Poblacion_Inicial)
	print("\n")

Poblacion_Inicial = np.array(sorted(Poblacion_Inicial, key = lambda x: x[-1], reverse = True))

print("\n##########################################################################")
print("La mejor solución encontrada para el problema de Max Ones es la siguiente:")
print(Poblacion_Inicial[0])
print("##########################################################################\n")