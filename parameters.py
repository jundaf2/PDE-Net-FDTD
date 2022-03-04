from numpy import *
class _parameters:
    # PERMEABILITY: float = 1
    # PERMITIVITY: float = 1
    SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light
    VACUUM_PERMEABILITY: float = 4e-7 * pi  # vacuum permeability
    VACUUM_PERMITIVITY: float = 1.0 / (VACUUM_PERMEABILITY * SPEED_LIGHT ** 2)
    FREQUENCY: float = 1e6
    DELTA_X: float = SPEED_LIGHT / FREQUENCY / 20
    DELTA_Y: float = DELTA_X
    DELTA_T: float = DELTA_X / (2 * SPEED_LIGHT)
    DIM: int = 100
    STEP: int = 50
    BATCH: int = 20

_options = {  # defalt options
        '--precision': 'float',
        'boundary': 'periodic',
        '--constraint': 'free',
        '--gpu': 0,
        '--kernel_size': 7, '--max_order': 5,
        '--xn': '50', '--yn': '50',
        '--interp_degree': 2, '--interp_mesh_size': 5,
        '--nonlinear_interp_degree': 4, '--nonlinear_interp_mesh_size': 20,
        '--nonlinear_interp_mesh_bound': 15,
        '--nonlinear_coefficient': 15,
        '--batch_size': 20, '--teststepnum': 20,
        '--maxiter': 5000,
        '--dt': _parameters.DELTA_T,
        '--dx': _parameters.DELTA_X,
        '--layer': list(range(0, 20)),
        '--recordfile': 'convergence',
        '--recordcycle': 100, '--savecycle': 10000,
        '--repeatnum': 5,
    }