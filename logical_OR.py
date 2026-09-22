w1 = 0.5
w2 = 0.5
bias = -0.25

def step_activation(z):
    if z > 0:
        return 1
    else:
        return 0

def perceptron_or(x1, x2):
    z = x1 * w1 + x2 * w2 + bias
    return step_activation(z)

inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

print("x1 x2 | OR")
for x1, x2 in inputs:
    y = perceptron_or(x1, x2)
    print(f"{x1}  {x2}  |  {y}")