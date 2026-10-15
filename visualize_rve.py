import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def convert_csv_to_nn_matrix(csv_path='grid_rve_dataset.csv', resolution=256):
    data = pd.read_csv(csv_path)
    X = data['X'].values
    Y = data['Y'].values
    Stress = data['Mises_Stress'].values
    
    grid_x, grid_y = np.mgrid[0:10:complex(resolution), 0:10:complex(resolution)]
    
    grid_stress = griddata((X, Y), Stress, (grid_x, grid_y), method='linear', fill_value=0)
    
    np.save('rve_input_matrix.npy', grid_stress)
    print("Matrix shape: " + str(grid_stress.shape))
    
    plt.figure(figsize=(6, 6))
    plt.imshow(grid_stress.T, extent=(0, 10, 0, 10), origin='lower', cmap='jet')
    plt.colorbar(label='Mises Stress (MPa)')
    plt.title('2D RVE Regular Stress Map')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.savefig('rve_stress_map.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    convert_csv_to_nn_matrix()