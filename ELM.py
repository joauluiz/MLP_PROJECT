# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 16:46:40 2021

@author: joao_
"""

import pandas 
import numpy as np
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

def tanh (v):
    return (np.exp(v)-np.exp(-v))/(np.exp(v)+np.exp(-v))

def logistic (v):
    return ( 1 /(1+np.exp(-v)))

#Definindo o número de neurônios na camada escondida
num_neur_hidden = 0
#Definindo quantas vezes a rede vai rodar
n = 1
var=1


Teste = 0
#Definindo o MAE e MSE como vetores
MAE_best = np.zeros(8)
MSE_best = np.zeros(8)
Resultado_best = np.zeros(8)
Qnt_Neur_MAE = np.zeros(8)
Qnt_Neur_MSE = np.zeros(8)


valor=8

MAE1 = 1000000000
MSE1 = 1000000000



for vezes in range (n):
        while (num_neur_hidden<=300):
            num_neur_hidden = num_neur_hidden + 5
            Lag = 0
            var=1
            for Lag in range(8):
                #Fazendo a leitura dos Inputs e transformando em arrays, colocando o Bias e normalizando
                Input_train = pandas.read_excel('Treinamento.xlsx', usecols=['MP10', 'TempMedia','Umidade','DiaSemana','Feriado'], sheet_name = Lag) #Colunas que servirão como inputs, já normalizadas
                Input_train = np.asarray(Input_train) #Transformação dos dados em matriz/vetor
                Input_train = np.concatenate((Input_train, np.ones((len(Input_train),1))), axis=1)
                Input_train = MinMaxScaler(feature_range=(-1, 1)).fit(Input_train).transform(Input_train)
                
                
                #Fazendo a leitura dos Inputs e transformando em arrays, colocando o Bias e normalizando
                Input_test = pandas.read_excel('Teste.xlsx', usecols=['MP10', 'TempMedia','Umidade','DiaSemana','Feriado'], sheet_name = Lag) #Colunas que servirão como inputs, já normalizadas
                Input_test = np.asarray(Input_test) #Transformação dos dados em matriz/vetor
                Input_test = np.concatenate((Input_test, np.ones((len(Input_test),1))), axis=1)
                Input_test = MinMaxScaler(feature_range=(-1, 1)).fit(Input_test).transform(Input_test)
                
                
                #Fazendo a leitura dos Outputs e transformando em arrays
                output_train = pandas.read_excel('Treinamento.xlsx', usecols=['Internacao'], sheet_name = Lag) #Variável em que ficarão os outputs na fase de treinamento
                output_train = np.asarray(output_train)
                output_max = output_train.max()
                output_min = output_train.min()
                output_train = MinMaxScaler(feature_range=(-1, 1)).fit(output_train).transform(output_train)
                
                
                output_test = pandas.read_excel('Teste.xlsx', usecols=['Internacao'], sheet_name = Lag) #Variável em que ficarão os outputs na fase de treinamento
                output_test = np.asarray(output_test)
                
                kfolds = KFold(n_splits=5, random_state=16, shuffle=True)
                
                for train_index, test_index in kfolds.split( Input_train, output_train):
                    X_train_folds, X_test_folds = Input_train[train_index], Input_train[test_index]
                    y_train_folds, y_test_folds = output_train[train_index], output_train[test_index]
                   
                    Teste = Teste + 1
                    #Criando os pesos de forma aleatória entre os Inputs e a camada escondida
                    w = np.random.uniform(-1,1,[len(X_train_folds[0]),num_neur_hidden])
                    
                    if(var==1):
                        u=w
                        var=var+1
                         
                    #Multiplicação entre a camada de entrada e os pesos, já sendo colocado os valores na função de ativação durante o treinamento
                    Mult_Input_train = logistic((np.dot(X_train_folds, u)))
                    
                    #Colocando o Bias na camada intermediária na fase de treinamento
                    Mult_Input_train = np.concatenate ((Mult_Input_train,np.ones((len(Mult_Input_train),1))),axis=1)
                    
                    #Encontrando a inversa generalizada
                    Mult_Input_train_gen = np.linalg.pinv(Mult_Input_train)
                    
                    #Encontrando os pesos entre a camada escondidade e a de saída
                    beta = Mult_Input_train_gen.dot(y_train_folds)
                    
                    
                    
                    #Multiplicação entre a camada de entrada e os pesos, já sendo colocado os valores na função de ativação durante o teste
                    Mult_Input_test = logistic(np.dot(X_test_folds, u))
                   
                    
                    
                    #Colocando o Bias na camada intermediária na fase de teste
                    Mult_Input_test = np.concatenate ((Mult_Input_test,np.ones((len(Mult_Input_test),1))),axis=1)
                    
                    #Encontrando os os resultados 
                    resultado = tanh((np.dot(Mult_Input_test, beta)))
                    
                    #Realizando a desnormalização dos dados
                    resultado = MinMaxScaler(feature_range = (output_min, output_max)).fit(resultado).transform(resultado)
                    y_test_folds = MinMaxScaler(feature_range = (output_min, output_max)).fit(y_test_folds).transform(y_test_folds)
                    
                    #Calculando o MAE e o MSE
                    MAE = mean_absolute_error(y_test_folds, resultado)
                    MSE = mean_squared_error(y_test_folds, resultado)
                    
                    #print("O resultado do MAE para o Teste ",Teste, " foi de:", MAE)
                    #print("O resultado do MSE para o Teste ",Teste, " foi de:", MSE)
                    
                    if(MAE<MAE1):
                        MAE1=MAE
                        beta_a_ser_usado = beta
                        
                    if(MSE<MSE1):
                        MSE1=MSE
                           
                MAE1 = 1000000000
                MSE1 = 1000000000
                
                #Multiplicação entre a camada de entrada e os pesos, já sendo colocado os valores na função de ativação durante o teste
                Mult_Input_test = logistic(np.dot(Input_test, u))
                
                #Colocando o Bias na camada intermediária na fase de teste
                Mult_Input_test = np.concatenate ((Mult_Input_test,np.ones((len(Mult_Input_test),1))),axis=1)
                
                #Encontrando os os resultados 
                resultado = logistic((np.dot(Mult_Input_test, beta_a_ser_usado)))
                
                #Realizando a desnormalização dos dados
                resultado = MinMaxScaler(feature_range = (output_min, output_max)).fit(resultado).transform(resultado)
                #output_test = MinMaxScaler(feature_range = (output_min, output_max)).fit(output_test).transform(output_test)
                
                #Calculando o MAE e o MSE
                MAE = mean_absolute_error(output_test, resultado)
                MSE = mean_squared_error(output_test, resultado)   
                
                        
                if (valor > 0):
                    
                    MAE_best[Lag] = MAE
                    MSE_best[Lag] = MSE
                    Qnt_Neur_MAE [Lag] = num_neur_hidden
                    valor = valor - 1 
                    plt.plot(np.arange(len(Input_test)), output_test, label='Valores reais')
                    plt.plot(np.arange(len(Input_test)), resultado, label='Valores calculados')
                    plt.ylabel('Internações')
                    plt.xlabel('Amostras do Testes')
                    plt.legend()
                    plt.show()
                    print('Grafico referente ao lag ', Lag, 'e ao MAE: ',MAE)
                    
                    
                    
                if (MAE < MAE_best[Lag]):
                     
                    MAE_best[Lag] = MAE
                    Qnt_Neur_MAE [Lag] = num_neur_hidden
                    
                    plt.plot(np.arange(len(Input_test)), output_test, label='Valores reais')
                    plt.plot(np.arange(len(Input_test)), resultado, label='Valores calculados')
                    plt.ylabel('Internações')
                    plt.xlabel('Amostras do Testes')
                    plt.legend()
                    plt.show()
                    print('Grafico referente ao lag ', Lag, 'e ao MAE: ',MAE)
                    
                if (MSE < MSE_best[Lag]):
                     
                    MSE_best[Lag] = MSE
                    Qnt_Neur_MSE [Lag] = num_neur_hidden
         
            
    
    #Plotando os gráficos dos resultados calculados pela rede versus os valores reais
#    plt.plot(np.arange(len(Input_test)), output_test, label='Valores reais')
#    plt.plot(np.arange(len(Input_test)), resultado, label='Valores calculados')
#    plt.ylabel('Internações')
#    plt.xlabel('Amostras do Testes')
#    plt.legend()
#    plt.show()
#    
for i in range(8):
    print('O resultado do MAE para o LAG' ,i,' é: ',MAE_best[i], ' [',Qnt_Neur_MAE[i],' Neurônios]' )
    print('O resultado do MSE para o LAG' ,i,' é: ',MSE_best[i], ' [',Qnt_Neur_MSE[i],' Neurônios]' )