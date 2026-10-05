import torch
from kan import KAN
import prop_data_2 as data_prop
import os
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kan_model = KAN(width=[2, 34, 4], grid=5, k=3, device=str(device))
pikan_model = KAN(width=[2, 34, 4], grid=4, k=3, device=str(device))

kan_model.load_state_dict(torch.load('./best_model/best_kan_model.pth', map_location=device))
pikan_model.load_state_dict(torch.load('./best_model/best_pikan_model.pth', map_location=device))

property_names = ['Young Modulus', 'UTS', 'Elongation', 'Yield Strength']

print("="*75)
print("               MATHEMATICAL FORMULAS FROM BASE KAN MODEL")
print("="*75)
try:
    sys.stdout = open(os.devnull, 'w')
    for l in range(len(kan_model.acts)):
        for i in range(kan_model.width[l]):
            for j in range(kan_model.width[l+1]):
                kan_model.fix_symbolic(l, i, j, 'poly')
    sys.stdout = sys.__stdout__
    
    for i, name in enumerate(property_names):
        print(f"{name:<15} (Scaled) = {kan_model.symbolic_formula()[0][i]}")
except Exception as e:
    sys.stdout = sys.__stdout__
    print(f"KAN symbolic extraction failed: {e}")

print("\n" + "="*75)
print("               MATHEMATICAL FORMULAS FROM PHYSICS-INFORMED PIKAN MODEL")
print("="*75)
try:
    sys.stdout = open(os.devnull, 'w')
    for l in range(len(pikan_model.acts)):
        for i in range(pikan_model.width[l]):
            for j in range(pikan_model.width[l+1]):
                pikan_model.fix_symbolic(l, i, j, 'poly')
    sys.stdout = sys.__stdout__
    
    for i, name in enumerate(property_names):
        print(f"{name:<15} (Scaled) = {pikan_model.symbolic_formula()[0][i]}")
except Exception as e:
    sys.stdout = sys.__stdout__
    print(f"PIKAN symbolic extraction failed: {e}")
print("="*75)