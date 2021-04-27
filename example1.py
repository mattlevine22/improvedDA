import numpy as np
from DA import VAR3D, ENKF
import pdb

## Experimental parameters
H = np.array([[1, 0, 0]])
settings = {
            'H': H,
            'dt': 0.01,
            'obs_noise_sd': 1,
            'dynamics_rhs': 'L63',
            'integrator': 'RK45'
            }

## run vanilla EnKF
output_dir = 'output/EnKF_vanilla'
enkf = ENKF(output_dir=output_dir, T=10, **settings)
enkf.test_filter()

## run 3DVAR with basic K
output_dir = 'output/K_basic'
nu = 0.1 # design parameter
Kbasic = H.T / (1 + nu)
var3d = VAR3D(K=Kbasic, output_dir=output_dir, T=5, **settings)
var3d.test_filter()

## run 3DVAR with a favorite K
output_dir = 'output/K_favorite'
Kfavorite = np.array([[0.08603118, 0.12466607, 0.00351079]]).T
var3d = VAR3D(K=Kfavorite, output_dir=output_dir, T=5, **settings)
var3d.test_filter()

## run 3DVAR with adaptive K
output_dir = 'output/K_learning'

var3d = VAR3D(K=Kbasic, output_dir=output_dir, T=50, **settings)
var3d.train_filter()
Kstar = var3d.K_vec_runningmean[-1]
print('Kstar = ', Kstar)

# run 3DVAR with steady state of K_adaptive
output_dir = 'output/K_learned'
var3d = VAR3D(K=Kstar, output_dir=output_dir, T=5, **settings)
var3d.test_filter()
