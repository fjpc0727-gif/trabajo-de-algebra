import numpy as np

# red neuronal para predecir si un estudiante aprueba
# usando operaciones matriciales - algebra lineal aplicada a IA

np.random.seed(7)

# datos: cada fila es un estudiante
# columnas: parcial1, parcial2, horas de estudio (valores entre 0 y 1)
estudiantes = np.array([
    [0.9, 0.85, 0.9],
    [0.2, 0.25, 0.15],
    [0.75, 0.65, 0.8],
    [0.1, 0.15, 0.1],
    [0.85, 0.9, 0.75],
    [0.3, 0.2, 0.35],
    [0.6, 0.7, 0.65],
    [0.15, 0.1, 0.2],
])

# resultado real: 1 aprueba, 0 reprueba
resultados = np.array([[1],[0],[1],[0],[1],[0],[1],[0]])

print("=" * 52)
print("  RED NEURONAL CON OPERACIONES MATRICIALES")
print("  Aplicacion: Prediccion de rendimiento estudiantil")
print("=" * 52)
print()
print("Datos de entrada (Parcial1 / Parcial2 / Horas):")
nombres = ["Carlos","Pedro","Maria","Juan","Ana","Luis","Sofia","Diego"]
for i, e in enumerate(estudiantes):
    estado = "Aprueba" if resultados[i][0] == 1 else "Reprueba"
    print(f"  {nombres[i]:<8} P1={e[0]:.2f}  P2={e[1]:.2f}  Horas={e[2]:.2f}  -> {estado}")
print()

# pesos iniciales aleatorios - estas son las matrices que la red va a aprender
# capa 1: 3 entradas -> 5 neuronas ocultas
W1 = np.random.randn(3, 5) * 0.3
b1 = np.zeros((1, 5))

# capa 2: 5 neuronas -> 1 salida
W2 = np.random.randn(5, 1) * 0.3
b2 = np.zeros((1, 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)


lr = 0.8
errores = []

print("Entrenando la red neuronal...")
print()

for i in range(6000):
    # propagacion hacia adelante
    # multiplicacion matricial: estudiantes x W1
    Z1 = np.dot(estudiantes, W1) + b1
    A1 = sigmoid(Z1)

    # multiplicacion matricial: A1 x W2
    Z2 = np.dot(A1, W2) + b2
    salida = sigmoid(Z2)

    # calculo 
    error = np.mean((resultados - salida) ** 2)
    errores.append(error)

    # retropropagacion - se usan transpuestas matriciales
    d_salida = -(resultados - salida) * derivada_sigmoid(Z2)
    dW2 = np.dot(A1.T, d_salida)
    db2 = np.sum(d_salida, axis=0, keepdims=True)

    d_oculta = np.dot(d_salida, W2.T) * derivada_sigmoid(Z1)
    dW1 = np.dot(estudiantes.T, d_oculta)
    db1 = np.sum(d_oculta, axis=0, keepdims=True)

    # actualizar pesos
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if (i + 1) % 1500 == 0:
        print(f"  Iteracion {i+1:5d}  |  Error: {error:.5f}")

print()

# resultados finales
Z1f = np.dot(estudiantes, W1) + b1
A1f = sigmoid(Z1f)
Z2f = np.dot(A1f, W2) + b2
predicciones = sigmoid(Z2f)

print("=" * 52)
print("RESULTADOS FINALES")
print("=" * 52)
print(f"{'Nombre':<10} {'Prediccion':>12} {'Resultado':>12} {'Real':>10} {'OK':>4}")
print("-" * 52)

aciertos = 0
for i in range(len(estudiantes)):
    pred = predicciones[i][0]
    resultado = "Aprueba" if pred >= 0.5 else "Reprueba"
    real = "Aprueba" if resultados[i][0] == 1 else "Reprueba"
    ok = "si" if resultado == real else "no"
    if ok == "si":
        aciertos += 1
    print(f"{nombres[i]:<10} {pred:>12.4f} {resultado:>12} {real:>10} {ok:>4}")

print("-" * 52)
print(f"Precision: {aciertos}/{len(estudiantes)} estudiantes correctos ({aciertos/len(estudiantes)*100:.0f}%)")
print(f"Error final: {errores[-1]:.6f}")
print()

# probar con un estudiante nuevo
print("=" * 52)
print("PRUEBA CON ESTUDIANTE NUEVO")
print("=" * 52)
nuevo_estudiante = np.array([[0.7, 0.75, 0.8]])
print(f"  Parcial 1:          {nuevo_estudiante[0,0]*100:.0f} puntos")
print(f"  Parcial 2:          {nuevo_estudiante[0,1]*100:.0f} puntos")
print(f"  Horas semanales:    {nuevo_estudiante[0,2]*10:.0f} horas")
print()

# operacion matricial paso a paso
Z1_n = np.dot(nuevo_estudiante, W1) + b1
A1_n = sigmoid(Z1_n)
Z2_n = np.dot(A1_n, W2) + b2
pred_n = sigmoid(Z2_n)[0][0]

print("Operaciones matriciales internas:")
print(f"  Z1 = estudiante x W1  ->  forma: {Z1_n.shape}")
print(f"  A1 = sigmoid(Z1)      ->  {np.round(A1_n, 3)}")
print(f"  Z2 = A1 x W2          ->  {np.round(Z2_n, 4)}")
print(f"  Prediccion final      ->  {pred_n:.4f}")
print()

if pred_n >= 0.5:
    print(f">> El estudiante APRUEBA con {pred_n*100:.1f}% de confianza")
else:
    print(f">> El estudiante REPRUEBA con {(1-pred_n)*100:.1f}% de confianza")

print()
print("=" * 52)
print("OPERACIONES MATRICIALES USADAS:")
print("  - Multiplicacion:  Z = X . W")
print("  - Transpuesta:     W.T  (en retropropagacion)")
print("  - Suma:            Z + b  (bias)")
print("  - Producto punto:  np.dot()")
print("=" * 52)