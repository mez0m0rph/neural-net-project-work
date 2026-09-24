import torch
import numpy as np

np_array = np.array([[1.0, 2.0], [3.0, 4.0]])

tensor = torch.from_numpy(np_array)

print(tensor)

x = torch.tensor(3.0, requires_grad=True)  # тензор нулевого порядка
# requires_grad = отслеживать все изменения с этой переменной. Под капотом строится граф вычислений

y = x ** 2
# pytorch запишет в граф вычислений "y зависит от x через операцию возведения в степень"

y.backward()  # backpropagation 
# pytorch по цепочке берет производную у каждой переменной, у которой стоял флаг requires_grad=True

print("производная y по точке 3 равна: ", x.grad)  # grad = gradient (производная)



p = torch.tensor(2.0, requires_grad=True)
T = 2 * (p**3) + 5 * p
T.backward()

print(p.grad)