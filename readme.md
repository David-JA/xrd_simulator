# xrd_simulator

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/xrd-simulator.svg)](https://pypi.org/project/xrd-simulator/)
[![CI](https://github.com/FABLE-3DXRD/xrd_simulator/actions/workflows/python-package-run-tests-linux-py38.yml/badge.svg)](https://github.com/FABLE-3DXRD/xrd_simulator/actions/workflows/python-package-run-tests-linux-py38.yml)
[![Docs](https://github.com/FABLE-3DXRD/xrd_simulator/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/FABLE-3DXRD/xrd_simulator/actions/workflows/pages/pages-build-deployment/)
[![PyPI version](https://badge.fury.io/py/xrd-simulator.svg)](https://pypi.org/project/xrd-simulator/)
[![Conda platforms](https://anaconda.org/conda-forge/xrd_simulator/badges/platforms.svg)](https://anaconda.org/conda-forge/xrd_simulator/)
[![Conda release](https://anaconda.org/conda-forge/xrd_simulator/badges/latest_release_relative_date.svg)](https://anaconda.org/conda-forge/xrd_simulator/)

**xrd_simulator** is a Python package for simulating X-ray diffraction from polycrystalline materials in 3D. It models a complete diffraction experiment: a monochromatic X‑ray beam illuminates a 3D polycrystalline sample while it undergoes a rigid‑body motion, and the scattered signal is collected by a 2D area detector. The sample can have arbitrary morphology (tetrahedral mesh) and orientation/texture, and the simulation accounts for geometric effects, Lorentz and polarization factors, and structure factors (from CIF files).

The package is designed to support a wide range of experimental geometries, including scanning‑3DXRD, full‑field 3DXRD, helical scans, and powder‑like diffraction. It is especially useful for studying the influence of sample microstructure on diffraction patterns and for optimising measurement strategies.

If you use this software in your research, please cite:

> Henningsson, A. & Hall, S. A. (2023). *xrd_simulator: 3D X-ray diffraction simulation software supporting 3D polycrystalline microstructure morphology descriptions*. J. Appl. Cryst. 56, 282–292. [https://doi.org/10.1107/S1600576722011001](https://doi.org/10.1107/S1600576722011001)

---

## Features

- **Arbitrary beam shape** – defined as a convex polyhedron.
- **Flexible detector geometry** – rectangular, pixelated, with configurable point‑spread function.
- **3D polycrystalline sample** – represented by a tetrahedral mesh; each element can be a single crystal with its own orientation, strain, and phase.
- **Rigid‑body motion** – rotation around an arbitrary axis and translation, parameterised by time.
- **Efficient diffraction computation** – vectorised solution of the Laue condition over many grains and reflections; parallel processing support.
- **Multiple intensity corrections** – Lorentz, polarization, and structure factors.
- **Rendering options** – centroid deposition, geometric projection, and scintillator‑blur simulation.
- **Template functions** – quick setup of common experiments (e.g., scanning‑3DXRD) and sample generation (e.g., from orientation distribution functions).
- **Save/load** – all objects can be serialised (pickle/dill) for reuse.

---

## Installation

### Conda (recommended)

The easiest way is to install from the `conda-forge` channel, which handles all dependencies automatically:

```bash
conda create -n xrd_sim python=3.9
conda activate xrd_sim
conda install -c conda-forge xrd_simulator
```

### Pip

You can also install via pip, but note that external dependencies of `pygalmesh` must be preinstalled on your system (see [pygalmesh documentation](https://github.com/nschloe/pygalmesh#installation)):

```bash
pip install xrd-simulator
```

### From source

```bash
git clone https://github.com/FABLE-3DXRD/xrd_simulator.git
cd xrd_simulator
python setup.py install
```

---

## Quick start

The simulation is built around four core objects:

- `Beam` – the X‑ray beam geometry and properties.
- `Detector` – the 2D area detector.
- `Polycrystal` – the sample (mesh + phases + orientations + strains).
- `RigidBodyMotion` – the motion of the sample during exposure.

Here is a minimal example (adapted from the documentation):

```python
import numpy as np
from xrd_simulator.beam import Beam
from xrd_simulator.detector import Detector
from xrd_simulator.mesh import TetraMesh
from xrd_simulator.phase import Phase
from xrd_simulator.polycrystal import Polycrystal
from xrd_simulator.motion import RigidBodyMotion

# 1. Define a box‑shaped beam
beam_vertices = np.array([
    [-1e6, -500, -500],
    [-1e6,  500, -500],
    [-1e6,  500,  500],
    [-1e6, -500,  500],
    [ 1e6, -500, -500],
    [ 1e6,  500, -500],
    [ 1e6,  500,  500],
    [ 1e6, -500,  500]])
beam = Beam(beam_vertices,
            xray_propagation_direction=[1,0,0],
            wavelength=0.28523,
            polarization_vector=[0,1,0])

# 2. Define a flat detector
detector = Detector(pixel_size_z=75.0, pixel_size_y=55.0,
                    det_corner_0=[142938.3, -38400, -38400],
                    det_corner_1=[142938.3,  38400, -38400],
                    det_corner_2=[142938.3, -38400,  38400])

# 3. Create a mesh (sphere from level set)
mesh = TetraMesh.generate_mesh_from_levelset(
    level_set=lambda x: np.linalg.norm(x) - 768.0,
    bounding_radius=769.0,
    max_cell_circumradius=450.)

# 4. Define a material phase (quartz)
quartz = Phase(unit_cell=[4.926, 4.926, 5.4189, 90, 90, 120],
               sgname='P3221',
               path_to_cif_file='quartz.cif')  # optional, for structure factors

# 5. Assign random orientations to each mesh element
from scipy.spatial.transform import Rotation as R
orientation = R.random(mesh.number_of_elements).as_matrix()
strain = np.zeros((3,3))          # zero strain
polycrystal = Polycrystal(mesh, orientation, strain, phases=quartz)

# 6. Define a sample motion (rotate by 1° around an axis)
motion = RigidBodyMotion(rotation_axis=[0, 1/np.sqrt(2), -1/np.sqrt(2)],
                         rotation_angle=np.radians(1.0),
                         translation=[123, -153.3, 3.42])

# 7. Compute diffraction
polycrystal.diffract(beam, detector, motion,
                     min_bragg_angle=0,
                     max_bragg_angle=None,   # automatically estimated
                     verbose=True,
                     number_of_processes=4)

# 8. Render the detector frame
img = detector.render(frames_to_render=0,
                      lorentz=True,
                      polarization=True,
                      structure_factor=True,
                      method='centroid')

# 9. Save/visualise
import matplotlib.pyplot as plt
plt.imsave('diffraction.png', img, cmap='gray')
detector.save('my_detector.det')
polycrystal.save('my_polycrystal.pc', save_mesh_as_xdmf=True)  # XDMF for Paraview
```

For more advanced examples, including the use of templates (`s3dxrd`, `polycrystal_from_odf`) and multiphase samples, see the [documentation](https://FABLE-3DXRD.github.io/xrd_simulator/) or the `examples/` directory in the source repository.

---

## Module overview

| Module            | Main classes / functions                               | Purpose |
|-------------------|--------------------------------------------------------|---------|
| `beam`            | `Beam`                                                 | X‑ray beam geometry and wave properties. |
| `detector`        | `Detector`                                             | 2D pixel detector, ray‑to‑detector projection, rendering. |
| `mesh`            | `TetraMesh`                                            | Tetrahedral mesh with geometric queries (normals, centroids, bounding spheres). |
| `phase`           | `Phase`                                                | Crystallographic phase: unit cell, space group, Miller indices, structure factors. |
| `motion`          | `RigidBodyMotion`, `_RodriguezRotator`                 | Rigid‑body rotation + translation, parameterised by time. |
| `laue`            | `find_solutions_to_tangens_half_angle_equation`, etc. | Time‑dependent Laue condition solver. |
| `polycrystal`     | `Polycrystal`                                          | Combines mesh, orientations, strains, phases; provides `diffract()` and `transform()`. |
| `scattering_unit` | `ScatteringUnit`                                       | Represents a single diffraction event (geometry, wavevectors, intensity factors). |
| `templates`       | `s3dxrd`, `polycrystal_from_odf`, `get_uniform_powder_sample` | High‑level helpers for common setups. |
| `utils`           | various                                                | Geometry utilities, CIF reading, strain‑B‑matrix conversions, progress bar, etc. |

---

## Dependencies

- Python ≥ 3.8
- `numpy`, `scipy`, `dill`, `pandas`
- `pygalmesh` and `meshio` (for mesh generation)
- `xfab` (for crystallographic calculations)
- `CifFile` (for reading CIF)
- `numba` (optional, speeds up some geometric clipping)

All dependencies are automatically resolved when installing via conda‑forge.

---

## Documentation

Full documentation is available at [https://FABLE-3DXRD.github.io/xrd_simulator/](https://FABLE-3DXRD.github.io/xrd_simulator/).

---

## Citation

If you find this package useful, please cite the corresponding paper:

> Henningsson, A. & Hall, S. A. (2023). *xrd_simulator: 3D X-ray diffraction simulation software supporting 3D polycrystalline microstructure morphology descriptions*. J. Appl. Cryst. 56, 282–292. [https://doi.org/10.1107/S1600576722011001](https://doi.org/10.1107/S1600576722011001)

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.

---

## Acknowledgements

`xrd_simulator` builds upon the excellent libraries [`xfab`](https://github.com/FABLE-3DXRD/xfab) and [`pygalmesh`](https://github.com/nschloe/pygalmesh). We thank their developers for making these tools available.