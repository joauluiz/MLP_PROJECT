# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:11:43 2021

@author: joao_
"""
 
import pandas 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings("ignore", category=Warning)

#Lag = 0

#Definindo quantas vezes a rede vai rodar
n = 30


#Definindo o MAE e MSE como vetores
MAE_best = np.zeros(8)
MSE_best = np.zeros(8)
MAPE_best = np.zeros(8)
Qnt_Neur_MAE = np.zeros(8)
Qnt_Neur_MAPE = np.zeros(8)
Qnt_Neur_MSE = np.zeros(8)

valor=8
C1 = 5 #camada 1
C2 = 5 #camada 2

Qnt_Neur = np.zeros(8)

val_random = -1

            

#teste 1 - IC - 200km

for vezes in range (n):
    C1=5
    val_random = val_random + 1
    while (C1<=500):
        for Lag in range(8):
            Input_train = pandas.read_excel('Treinamento.xlsx', usecols=['MP10', 'TempMedia','Umidade','DiaSemana','Feriado'], sheet_name = Lag )
            Input_train_Norm = MinMaxScaler(feature_range=(-1, 1)).fit(Input_train).transform(Input_train)
                
            Input_test = pandas.read_excel('Teste.xlsx', usecols=['MP10', 'TempMedia','Umidade','DiaSemana','Feriado'], sheet_name = Lag)
            Input_test_Norm = MinMaxScaler(feature_range=(-1, 1)).fit(Input_test).transform(Input_test)
                
                
            Output_train = pandas.read_excel('Treinamento.xlsx', usecols=['Internacao'], sheet_name = Lag)
            Output_train_Norm = MinMaxScaler(feature_range=(-1, 1)).fit(Output_train).transform(Output_train)
                
                
            Output_test = pandas.read_excel('Teste.xlsx', usecols=['Internacao'], sheet_name = Lag)
            Output_test = np.asarray(Output_test)
            Output_max = Output_test.max()
            Output_min = Output_test.min()
                
                
            mlpr = MLPRegressor(hidden_layer_sizes=(C1),
                                    max_iter=1000,
                                    learning_rate_init=0.001,
                                    validation_fraction=0.15,
                                    activation='tanh',
                                    solver='lbfgs',
                                    tol=1e-4,
                                    random_state = val_random)
            

                
            Treino = mlpr.fit(Input_train_Norm, Output_train_Norm)
            
            scores = cross_validate(mlpr, Input_train_Norm, Output_train_Norm, cv=5,
                        scoring=('neg_mean_squared_error'),
                        return_train_score=True,
                        return_estimator=True)
                
            Resultado = Treino.predict(Input_test_Norm)
            Resultado=Resultado.reshape(-1,1)
                
            Resultado = MinMaxScaler(feature_range = (Output_min, Output_max)).fit(Resultado).transform(Resultado)
                
            #        plt.plot(np.arange(len(Input_test)), Output_test, label='Valores reais')
            #        plt.plot(np.arange(len(Input_test)), Resultado, label='Valores calculados')
            #        plt.ylabel('Internações')
            #        plt.xlabel('Amostras do Testes')
            #        plt.legend()
            #        plt.show()
            
                    #Calculando o MAE e o MSE
            MAE = mean_absolute_error(Output_test, Resultado)
            MSE = mean_squared_error(Output_test, Resultado)
            MAPE = mean_absolute_percentage_error(Output_test, Resultado)
                    
            if (valor > 0):
                        
                MAE_best[Lag] = MAE
                MSE_best[Lag] = MSE
                MAPE_best[Lag] = MAPE
                        
                valor = valor - 1 
                Qnt_Neur_MAE[Lag] = C1
                Qnt_Neur_MAPE[Lag] = C1
                Qnt_Neur_MSE[Lag] = C1
                        
                plt.plot(np.arange(len(Input_test)), Output_test, label='Valores reais')
                plt.plot(np.arange(len(Input_test)), Resultado, label='Valores calculados')
                plt.ylabel('Internações')
                plt.xlabel('Amostras do Testes')
                plt.legend()
                plt.show()
                print('Grafico referente ao lag ', Lag, 'e ao MSE: ',MSE)
                        
            if (MAE < MAE_best[Lag]):
                         
                MAE_best[Lag] = MAE
                Qnt_Neur_MAE[Lag] = C1
                   
            if (MSE < MSE_best[Lag]):
                MSE_best[Lag] = MSE
                Qnt_Neur_MSE[Lag] = C1
                        
                plt.plot(np.arange(len(Input_test)), Output_test, label='Valores reais')
                plt.plot(np.arange(len(Input_test)), Resultado, label='Valores calculados')
                plt.ylabel('Internações')
                plt.xlabel('Amostras do Testes')
                plt.legend()
                plt.show()
                print('Grafico referente ao lag ', Lag, 'e ao MSE: ',MSE)
                        
            if (MAPE < MAPE_best[Lag]):
                MAPE_best[Lag] = MAPE   
                Qnt_Neur_MAPE[Lag] = C1
                    
        Lag=0
        C1 = C1 + 5
                
         
    
for i in range(8):
    print('O resultado do MAE para o LAG' ,i,' é: ',MAE_best[i], ' [',Qnt_Neur_MAE[i],' Neurônios]' )
    print('O resultado do MSE para o LAG' ,i,' é: ',MSE_best[i], ' [',Qnt_Neur_MSE[i],' Neurônios]' )
    print('O resultado do MAPE para o LAG' ,i,' é: ',MAPE_best[i], ' [',Qnt_Neur_MAPE[i],' Neurônios]\n' )