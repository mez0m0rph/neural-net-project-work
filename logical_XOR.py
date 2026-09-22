# XOR = AND(OR(x1, x2), NAND(x1, x2))

def step_activation(z):
    if z > 0:
        return 1
    else:
        return 0

def perceptron_or(x1, x2):
    w1, w2, bias = 0.5, 0.5, -0.25
    z = x1 * w1 + x2 * w2 + bias
    return step_activation(z)

def perceptron_and(x1, x2):
    w1, w2, bias = 0.5, 0.5, -0.5
    z = w1 * x1 + w2 * x2 + bias
    return step_activation(z)

def perceptron_nand(x1, x2):
    return 1 - perceptron_and(x1, x2)

def perceptron_xor(x1, x2):
    a = perceptron_or(x1, x2)
    b = perceptron_nand(x1, x2)
    return perceptron_and(a, b)

inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

print("x1 x2 | XOR")
for x1, x2 in inputs:
    y = perceptron_xor(x1, x2)
    print(f"{x1}  {x2}  |  {y}")