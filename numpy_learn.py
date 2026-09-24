import numpy as np

data = np.array([
    [10.0, 200.0],
    [12.5, 180.0],
    [15.0, 190.0],
    [9.0, 210.0],
    [11.0, 205.0]
])

print("форма матрицы", data.shape)

speeds = data[:, 0]
print("все скорости", speeds)


max_values = np.max(data, axis=0)
print("наибольшие (скорость и давление)", max_values)


normalized_data = data / max_values
print("нормализованные данные: \n", normalized_data)


experimental_results = np.array([  # скорость + давление + температура
    [10.0, 200.0, 55.0],
    [12.0, 180.0, 62.0],
    [15.0, 190.0, 70.0],
    [9.0,  210.0, 48.0],
    [11.0, 205.0, 59.0],
    [14.0, 195.0, 68.0]
])

X = experimental_results[:,:2]
print(X.shape)

Y = experimental_results[:, 2:]
print(Y)

T_mean = np.mean(Y)
print(T_mean)