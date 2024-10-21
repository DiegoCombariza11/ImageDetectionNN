#tensorflow es de google y hay resti de modelos ya hechos 
import tensorflow as tf
import numpy as np

# los datos de entrenamiento, la entrada y lo que se supone debe salir :p
celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# se usa keras para hacer todo más fácil y bonito :p
capa = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([capa])


#configuramos el modelo para que aprenda, le decimos qué
#  optimizador y que función de perdida usar, si es mas chikita
# mejor, eso significa qué tanto va a variar al optener una respuesta
# adam es un optimizador que se usa mucho, y mean_squared_error 
# una función que dice que es preferible muchos errores chikitos
# que pocos errores grandes

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

#entrenamos

#acá le decimos que entrene, le damos los datos de entrada y salida
# y le decimos cuantas veces queremos que lo haga (1000), cada una
# significa que va a intentar encontrar la respuesta correcta
# y ajustar los valores de la capa para que la respuesta sea la correcta
# si ya lo encontró pues, ya no cambia mucho
# verbose=false pa que no imprima tantas cosas
print("Entrenando...")
historial = modelo.fit(celsius_q, fahrenheit_a, epochs=1000, verbose=False)

print("Modelo entrenado!")


#grafiquita para ver el error con cada iteración

import matplotlib.pyplot as plt
#cada iteración se conoce como época
plt.xlabel("# Iteración")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])



# para este punto debería estár funcionando 1 A

print("Hagamos una predicción!")
#le pasamos los datitos de entrada y nos da la respuesta wiiii
resultado = modelo.predict([100.0])
print("El resultado es " + str(resultado) + " fahrenheit!")


#podemos ver los variables que se ajustaron en la capa

print("Variables internas del modelo")
print(capa.get_weights())
