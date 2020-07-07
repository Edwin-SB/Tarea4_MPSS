#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy
import numpy as np
from scipy import stats
import pandas as pd
from scipy import signal
from scipy import integrate
import matplotlib.pyplot as plt

#Se crean los dataframes de los archivos xy.csv
datos = pd.read_csv('bits10k.csv')
df = pd.DataFrame(datos)
C_Bits = df['0'].to_numpy()
print(C_Bits)

'''
#1. Pregunta#1 Esquema de modulación BPSK para los bits presentados. 
'''
# Frecuencia de operación
f = 1000 # 1 kHz
N=10000
# Duración del período de cada onda
T = 1/f # 1 ms

# Número de puntos de muestreo por período
p = 50 

# Puntos de muestreo para cada período
tp = np.linspace(0, T, p)

# Creación de la forma de onda
seno = np.sin(2*np.pi * f * tp)
coseno= np.cos(2*np.pi * f * tp)

# Visualización de la forma de onda de la portadora
plt.plot(tp, seno)
plt.xlabel('Tiempo / s')
plt.savefig('ondaSen.png')

plt.plot(tp, coseno)
plt.xlabel('Tiempo / s')
plt.savefig('ondaCos.png')
plt.show()

# Frecuencia de muestreo
fs = p/T # 50 kHz

# Creación de la línea temporal para toda la señal Tx
t = np.linspace(0, N*T, N*p)
print('T.shape:',t.shape)
print('N*p:',N*p)
senal = np.zeros(t.shape)# inicializa vector senal vacio del tamaño N*p

# enumerate acomoda los bits
# b*seno si b es  0 da vector ceros pero si b es 1 da seno   
#va a depostitar k-decimo segmoento de señal una forma de onda que depende del bit asociado 
for k, b in enumerate(C_Bits):
    if b == 1:
        senal[k*p:(k+1)*p] = seno
    else:
        senal[k*p:(k+1)*p] = -seno


# Visualización de los primeros bits modulados
print('Antes canal Ruidoso' )
pb = 5
plt.figure()
plt.plot(senal[0:pb*p])
plt.savefig('Tx.png')# senal transmitida 

'''
# Pregunta #2 
'''
# Potencia intantánea
Pinst = senal**2

# Potencia promedio: promedio temporal de la potencia instantánea
Ps = integrate.trapz(Pinst, t) / (T*N)

'''
#Pregunta#3. Simular un canal ruidoso del tipo AWGN (ruido aditivo blanco gaussiano).
'''
#Relacion señal-ruido deseada
SNR_lista = [-2,-1,0,1,2,3]
SNR = np.array(SNR_lista)

# potencia del ruido

Pn1 = Ps / 10**(SNR[0] / 10)
Pn2 = Ps / 10**(SNR[1] / 10)
Pn3 = Ps / 10**(SNR[2] / 10)
Pn4 = Ps / 10**(SNR[3] / 10)
Pn5 = Ps / 10**(SNR[4] / 10)
Pn6 = Ps / 10**(SNR[5] / 10)


print('Pn1:',Pn1)
print('Pn2:',Pn2)
print('Pn3:',Pn3)
print('Pn4:',Pn4)
print('Pn5:',Pn5)
print('Pn6:',Pn6)


#Desviacion estandar del ruido 
sigma1 = np.sqrt(Pn1)
sigma2 = np.sqrt(Pn2)
sigma3 = np.sqrt(Pn3)
sigma4 = np.sqrt(Pn4)
sigma5 = np.sqrt(Pn5)
sigma6 = np.sqrt(Pn6)

print('Sigma: ',sigma1,sigma2,sigma3,sigma4,sigma5,sigma6)
# Crear ruido
# .normal nos ayuda a generar ruido normal 
ruido1 = np.random.normal(0, sigma1, senal.shape)
ruido2 = np.random.normal(0, sigma2, senal.shape)
ruido3 = np.random.normal(0, sigma3 , senal.shape)
ruido4 = np.random.normal(0, sigma4, senal.shape)
ruido5 = np.random.normal(0, sigma5, senal.shape)
ruido6 = np.random.normal(0, sigma6, senal.shape)


# "El canal": señal recibida
print
Rx1 = senal + ruido1
Rx2 = senal + ruido2
Rx3 = senal + ruido3
Rx4 = senal + ruido4
Rx5 = senal + ruido5
Rx6 = senal + ruido6



# Visualización de los pirmeros bits recibidos
print('Despues canal Ruidoso')
pb = 5
plt.figure()
plt.plot(Rx1[0:pb*p])# desde 0 hasta 5*p, p es el numero de punto de muestreo por cada periodo
plt.savefig('Rx1.png')# punto 4

pb = 5
plt.figure()
plt.plot(Rx2[0:pb*p])# desde 0 hasta 5*p, p es el numero de punto de muestreo por cada periodo
plt.savefig('Rx2.png')# punto 4

pb = 5
plt.figure()
plt.plot(Rx3[0:pb*p])# desde 0 hasta 5*p, p es el numero de punto de muestreo por cada periodo
plt.savefig('Rx3.png')# punto 4

pb = 5
plt.figure()
plt.plot(Rx4[0:pb*p])# desde 0 hasta 5*p, p es el numero de punto de muestreo por cada periodo
plt.savefig('Rx4.png')# punto 4

pb = 5
plt.figure()
plt.plot(Rx5[0:pb*p])# desde 0 hasta 5*p, p es el numero de punto de muestreo por cada periodo
plt.savefig('Rx5.png')# punto 4

pb = 5
plt.figure()
plt.plot(Rx6[0:pb*p])# desde 0 hasta 5*p, p es el numero de punto de muestreo por cada periodo
plt.savefig('Rx6.png')# punto 4

'''
# Pregunta #4. Demodular y decodificar la señal.
'''
# amplitud onda seudo energia
Es = np.sum(seno**2)

# Inicialización del vector de bits recibidos
bitsRx1 = np.zeros(C_Bits.shape)
bitsRx2 = np.zeros(C_Bits.shape)
bitsRx3 = np.zeros(C_Bits.shape)
bitsRx4 = np.zeros(C_Bits.shape)
bitsRx5 = np.zeros(C_Bits.shape)
bitsRx6 = np.zeros(C_Bits.shape)


# Decodificación de la señal por detección de energía Rx1
for k in range(len(bitsRx1)):
  E = np.sum(Rx1[k*p:(k+1)*p] * seno)
  if E > 0:
    bitsRx1[k] = 1
  else:
    bitsRx1[k] = 0
print('bitsRx1',bitsRx1)    
# Decodificación de la señal por detección de energía Rx2
for k in range(len(bitsRx2)):
  E = np.sum(Rx2[k*p:(k+1)*p] * seno)
  if E > 0:
    bitsRx2[k] = 1
  else:
    bitsRx2[k] = 0
    
# Decodificación de la señal por detección de energía Rx3
for k in range(len(bitsRx3)):
  E = np.sum(Rx3[k*p:(k+1)*p] * seno)
  if E > 0:
    bitsRx3[k] = 1
  else:
    bitsRx3[k] = 0
    
# Decodificación de la señal por detección de energía Rx4
for k in range(len(bitsRx4)):
  E = np.sum(Rx4[k*p:(k+1)*p] * seno)
  if E > 0:
    bitsRx4[k] = 1
  else:
    bitsRx4[k] = 0
# Decodificación de la señal por detección de energía Rx5
for k in range(len(bitsRx5)):
  E = np.sum(Rx5[k*p:(k+1)*p] * seno)
  if E > 0:
    bitsRx5[k] = 1
  else:
    bitsRx5[k] = 0
# Decodificación de la señal por detección de energía Rx6
for k in range(len(bitsRx6)):
  E = np.sum(Rx6[k*p:(k+1)*p] * seno)
  if E > 0:
    bitsRx6[k] = 1
  else:
    bitsRx6[k] = 0
print('Decodificados',bitsRx1,bitsRx2,bitsRx3,bitsRx4,bitsRx5,bitsRx6)  
# Contar errores
err1 = np.sum(np.abs(C_Bits - bitsRx1))
err2 = np.sum(np.abs(C_Bits - bitsRx2))
err3 = np.sum(np.abs(C_Bits - bitsRx3))
err4 = np.sum(np.abs(C_Bits - bitsRx4))
err5 = np.sum(np.abs(C_Bits - bitsRx5))
err6 = np.sum(np.abs(C_Bits - bitsRx6))
print('error1:',err1)
print('error2:',err2)
print('error3:',err3)
print('error4:',err4)
print('error5:',err5)
print('error6:',err6)
# Tasa de error de bits (BER, bit error rate)
BER1 = err1/N
BER2 = err2/N
BER3 = err3/N
BER4 = err4/N
BER5 = err5/N
BER6 = err6/N
V_BER_list = [BER1,BER2,BER3,BER4,BER5,BER6]
V_BER = np.array(V_BER_list)
print('Tasa error:',BER1)
print('Tasa error:',BER2)
print('Tasa error:',BER3)
print('Tasa error:',BER4)
print('Tasa error:',BER5)
print('Tasa error:',BER6) 
print('Vector Tasa de error:',V_BER_list)

print('Hay un total de {} errores en {} bits para una tasa de error de {}.'.format(err1, N, BER1))
print('Hay un total de {} errores en {} bits para una tasa de error de {}.'.format(err2, N, BER2))
print('Hay un total de {} errores en {} bits para una tasa de error de {}.'.format(err3, N, BER3))
print('Hay un total de {} errores en {} bits para una tasa de error de {}.'.format(err4, N, BER4))
print('Hay un total de {} errores en {} bits para una tasa de error de {}.'.format(err5, N, BER5))
print('Hay un total de {} errores en {} bits para una tasa de error de {}.'.format(err6, N, BER6))


print('Densidad espectral de potencia de la señal con el método de Welch ')
fw, PSD = signal.welch(senal, fs, nperseg=1024)
plt.semilogy(fw, PSD)
plt.savefig('PSD.png')

print('Densidad espectral de potencia de la señal con el método de Welch ')
fw, PSD = signal.welch(Rx1, fs, nperseg=1024)
plt.figure()
plt.semilogy(fw, PSD)
plt.savefig('PSD1.png')

fw, PSD = signal.welch(Rx2, fs, nperseg=1024)
plt.figure()
plt.semilogy(fw, PSD)
plt.savefig('PSD2.png')

fw, PSD = signal.welch(Rx3, fs, nperseg=1024)
plt.figure()
plt.semilogy(fw, PSD)
plt.savefig('PSD3.png')

fw, PSD = signal.welch(Rx4, fs, nperseg=1024)
plt.figure()
plt.semilogy(fw, PSD)
plt.savefig('PSD4.png')

fw, PSD = signal.welch(Rx5, fs, nperseg=1024)
plt.figure()
plt.semilogy(fw, PSD)
plt.savefig('PSD5.png')

fw, PSD = signal.welch(Rx6, fs, nperseg=1024)
plt.figure()
plt.semilogy(fw, PSD)
plt.savefig('PSD6.png')
plt.show()

plt.figure()
plt.plot(V_BER,SNR)
plt.xlabel('BER')
plt.ylabel('SNR')
plt.title('BER versus SNR')
plt.show()


# In[ ]:





# In[ ]:




