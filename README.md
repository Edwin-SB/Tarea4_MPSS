# Tarea4_MPSS
## Edwin Somarribas Barahona
## B16453

### Pregunta#1 Crear un esquema de modulación BPSK para los bits presentados
Para la realizacion de esta tarea se toman en cuenta parametros como:
 Frecuencia de operación
f = 1000 
#### Duración del período de cada onda
T = 1/f # 1 ms
#### Número de puntos de muestreo por período
p = 50 

#### Puntos de muestreo para cada período
tp = np.linspace(0, T, p)
por medio de la libreria numpy se definen los puntos de muestreo para la frecuencia y tambien se define en este caso la señal senoidal, adicionalmente se agrega la señal coseno
como para efectos de comprobación.

#### Creación de la forma de onda
seno = np.sin(2*np.pi * f * tp)
coseno= np.cos(2*np.pi * f * tp)
 
 Como se puede observar en la siguiente imagen 
![Imagen1](https://github.com/Edwin-SB/Tarea4_MPSS/blob/master/ondaSen.png)
![Imagen1](https://github.com/Edwin-SB/Tarea4_MPSS/blob/master/ondaCos.png)

####  Se unaFrecuencia de muestreo definida por 
fs = p/T # 50 kHz
#### Creación de la línea temporal para toda la señal Tx
Por medio de numpy y de linspace, se define una señal, llamada senal, que es un vector donde se alojan los valores,del tamaño N*p
t = np.linspace(0, N*T, N*p)
senal = np.zeros(t.shape)# inicializa vector senal vacio del tamaño N*p

La funcion enumerate acomoda los bits, se va a depostitar k-decimo segmoento de señal una forma de onda que depende del bit asociado , con un for, si el bit es 1 la funcion es seno en caso contrario es -sen.
b*seno si b es  0 da vector ceros pero si b es 1 da seno 

for k, b in enumerate(C_Bits):
    if b == 1:
        senal[k*p:(k+1)*p] = seno
    else:
        senal[k*p:(k+1)*p] = -seno

Esto se ve reflejado en la siguiente imagen 
![Imagen1](https://github.com/Edwin-SB/Tarea4_MPSS/blob/master/Tx.png)

### Pregunta#2 Calcular la potencia promedio de la señal modulada generada.

#### Potencia intantánea
Pinst = senal**2
  

##### Potencia promedio: promedio temporal de la potencia instantánea
Ps = integrate.trapz(Pinst, t) / (T*N)
Dando como resultado una potencia promedio PS
Ps = 0.4899519799039596

### Pregunta#3 Simular un canal ruidoso del tipo AWGN (ruido aditivo blanco gaussiano) con una relación señal a ruido (SNR) desde -2 hasta 3 dB.
Primeramente se define los 6 SNR para cada uno de los 6 valores (-2,3), luego se calcula las 6 potencias del ruido 
Pn(1,2,3,4,5,6) = Ps / 10**(SNR[0] / 10)
Dando como resultados:

Pn1: 0.7765215575826299

Pn2: 0.6168129980599599

Pn3: 0.4899519799039596

Pn4: 0.38918269129677885

Pn5: 0.30913880016301537

Pn6: 0.24555767741827042

Se calcula la desviacion estandar del ruido para cada valor del vector SNR, mediante la potencia del ruido y se crean los respectivos canales de ruido
sigma1 = np.sqrt(Pn1)
Rx(1,2,3,4,5,6) = senal + ruido(1,2,3,4,5,6)

Las siguiente imagenes dan como resultado grafico dichos canales despues del ruido 
![Imagen1](https://github.com/Edwin-SB/Tarea4_MPSS/blob/master/Rx1.png)
![Imagen1](https://github.com/Edwin-SB/Tarea4_MPSS/blob/master/Rx2.png)
![Imagen1](https://github.com/Edwin-SB/Tarea4_MPSS/blob/master/Rx3.png)
![Imagen1](https://github.com/Edwin-SB/Tarea4_MPSS/blob/master/Rx4.png)
![Imagen1](https://github.com/Edwin-SB/Tarea4_MPSS/blob/master/Rx5.png)
![Imagen1](https://github.com/Edwin-SB/Tarea4_MPSS/blob/master/Rx6.png)

### Pregunta#4 Graficar la densidad espectral de potencia de la señal con el método de Welch (SciPy), antes y después del canal ruidoso.


(Antes)

![Imagen1](https://github.com/Edwin-SB/Tarea4_MPSS/blob/master/PSD.png)

(Despues)

![Imagen1](https://github.com/Edwin-SB/Tarea4_MPSS/blob/master/PSD1.png)
![Imagen1](https://github.com/Edwin-SB/Tarea4_MPSS/blob/master/PSD2.png)
![Imagen1](https://github.com/Edwin-SB/Tarea4_MPSS/blob/master/PSD3.png)
![Imagen1](https://github.com/Edwin-SB/Tarea4_MPSS/blob/master/PSD4.png)
![Imagen1](https://github.com/Edwin-SB/Tarea4_MPSS/blob/master/PSD5.png)
![Imagen1](https://github.com/Edwin-SB/Tarea4_MPSS/blob/master/PSD6.png)


### Pregunta#5 Demodular y decodificar la señal y hacer un conteo de la tasa de error de bits (BER, bit error rate) para cada nivel SNR
Para esto inicialmente se Inicialización del vector de bits recibidos  para cada una de las entradas del vector bitsRx
bitsRx1 = np.zeros(C_Bits.shape)

Luego  se Decodificación de la señal por detección de energía Rx, para cada una de las entradas del vector RX
si la energia es mayor a 0  escribe en el vector bitsRx un 1
for k in range(len(bitsRx1)):
  E = np.sum(Rx1[k*p:(k+1)*p] * seno)
  if E > 0:
    bitsRx1[k] = 1
  else:
    bitsRx1[k] = 0
    
Posteriormente se Cuentan los errores para el vector err
err = np.sum(np.abs(C_Bits - bitsRx))

Se calcula la Tasa de error de bits (BER, bit error rate) para el vector BER
BER = err/N

### Pregunta#6 Graficar BER versus SNR.
![Imagen1](https://github.com/Edwin-SB/Tarea4_MPSS/blob/master/PSD1.png)




    
    
