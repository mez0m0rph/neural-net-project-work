import sympy

# f(x) = (x-3) ** 2

def slope(x):  # slope - значение производной (вычисляет по функции, а не передается числом)
    return 2 * (x - 3)

print("slope: ", slope(7))

x = sympy.symbols("x")
print(sympy.solve(2 * (x - 3), x))

def gradient_descent(slope, start, steps, lr=0.1):
    x = float(start)
    for i in range(steps):
        g = slope(x)
        x = x - lr * g
    return x

print(gradient_descent(slope, 900, 3))