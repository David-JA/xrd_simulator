"""The detector module is used to represent a 2D area detector. After diffraction from a
:class:`xrd_simulator.polycrystal.Polycrystal` has been computed, the detector can render
the scattering as a pixelated image via the :func:`xrd_simulator.detector.Detector.render`
function.

Here is a minimal example of how to instantiate a detector object and save it to disc:

    Examples:
        .. literalinclude:: examples/example_init_detector.py

Below follows a detailed description of the detector class attributes and functions.

"""

from typing import Any, Dict, List, Optional, Tuple, Union

import dill
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from scipy.special import j1

from xrd_simulator import utils
from xrd_simulator.cuda import get_selected_device
from xrd_simulator.utils import ensure_numpy, ensure_torch, return_device_memory

torch.set_default_dtype(torch.float64)


class Detector:
    """Represents a rectangular 2D area detector.

    The detector collects :class:`xrd_simulator.scattering_unit.ScatteringUnit`
    during diffraction from a :class:`xrd_simulator.polycrystal.Polycrystal`.
    The detector implements various rendering of the scattering as 2D pixelated
    frames. The detector geometry is described by specifying the locations of
    three detector corners. The detector is described in the laboratory
    coordinate system.

    Parameters
    ----------
    det_corner_0 : numpy.ndarray
        Detector corner 3D coordinates, shape ``(3,)``. The origin of the
        detector is at this corner.
    det_corner_1 : numpy.ndarray
        Detector corner 3D coordinates, shape ``(3,)``. Defines the y-axis
        direction of the detector.
    det_corner_2 : numpy.ndarray
        Detector corner 3D coordinates, shape ``(3,)``. Defines the z-axis
        direction of the detector.
    n_pixels : int or tuple, optional
        Number of pixels. Can be a single int (square detector) or tuple
        ``(n_z, n_y)``. If provided, pixel sizes are calculated automatically
        from detector dimensions. Mutually exclusive with ``pixel_size``.
    pixel_size : float or tuple, optional
        Pixel side length in microns. Can be a single value (square pixels)
        or tuple ``(pixel_size_z, pixel_size_y)`` for rectangular pixels.
        ``zdhat`` is the unit vector from ``det_corner_0`` towards
        ``det_corner_2``. ``ydhat`` is the unit vector from ``det_corner_0``
        towards ``det_corner_1``.
    gaussian_sigma : float, optional
        Standard deviation of the Gaussian point spread function in pixels.
        Default is 1.0.
    kernel_threshold : float, optional
        Intensity threshold used to determine the kernel size. The kernel will
        extend until the Gaussian tail drops below this value. Default is 0.02.
    max_gaussian_kernel_radius : int
        Maximum radius of the Gaussian kernel. Default is 2 providing a 5x5 kernel, which is consistent with the Airy method.
        When the Gaussian kernel radius is larger than this value, the kernel will be truncated.
    use_lorentz : bool, optional
        Whether to apply the Lorentz factor correction. Default is ``True``.
    use_polarization : bool, optional
        Whether to apply the polarization factor correction. Default is ``True``.
    use_structure_factor : bool, optional
        Whether to apply the structure factor. Default is ``True``.

    Attributes
    ----------
    pixel_size_z : float
        Pixel side length along ``zdhat`` (rectangular pixels) in microns.
    pixel_size_y : float
        Pixel side length along ``ydhat`` (rectangular pixels) in microns.
    det_corner_0 : torch.Tensor
        Detector corner 3D coordinates, shape ``(3,)``. The origin of the
        detector.
    det_corner_1 : torch.Tensor
        Detector corner 3D coordinates, shape ``(3,)``.
    det_corner_2 : torch.Tensor
        Detector corner 3D coordinates, shape ``(3,)``.
    frames : list
        Analytical diffraction frames.
    zdhat : torch.Tensor
        Detector basis vector along the z-axis direction.
    ydhat : torch.Tensor
        Detector basis vector along the y-axis direction.
    normal : torch.Tensor
        Detector normal, formed as the cross product:
        ``torch.cross(self.zdhat, self.ydhat)``.
    zmax : torch.Tensor
        Detector width along ``zdhat``.
    ymax : torch.Tensor
        Detector height along ``ydhat``.
    pixel_coordinates : torch.Tensor
        Real space 3D detector pixel coordinates, shape ``(n_z, n_y, 3)``.
    gaussian_sigma : float
        Standard deviation of the Gaussian point spread function.
    kernel_threshold : float
        Threshold value that determines the kernel size.
    gaussian_kernel : torch.Tensor
        Pre-computed normalized 2D Gaussian kernel.
    lorentz_factor : bool
        Whether the Lorentz factor correction is applied.
    polarization_factor : bool
        Whether the polarization factor correction is applied.
    structure_factor : bool
        Whether the structure factor is applied.
    """

    # ------------------------------------------------------------------
    # 1. Core lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        det_corner_0: npt.NDArray,
        det_corner_1: npt.NDArray,
        det_corner_2: npt.NDArray,
        n_pixels: int | tuple = None,
        pixel_size: float | tuple = None,
        gaussian_sigma: float = 1.0,
        kernel_threshold: float = 0.02,
        max_gaussian_kernel_radius: int = 2,
        use_lorentz: bool = True,
        use_polarization: bool = True,
        use_structure_factor: bool = True,
    ):
        self.det_corner_0 = ensure_torch(det_corner_0)
        self.det_corner_1 = ensure_torch(det_corner_1)
        self.det_corner_2 = ensure_torch(det_corner_2)

        self.zmax = torch.linalg.norm(self.det_corner_2 - self.det_corner_0)
        self.ymax = torch.linalg.norm(self.det_corner_1 - self.det_corner_0)

        # Determine pixel sizes from either n_pixels or pixel_size
        if n_pixels is not None:
            if pixel_size is not None:
                raise ValueError("Cannot specify both n_pixels and pixel_size")
            if isinstance(n_pixels, (tuple, list)):
                n_z, n_y = n_pixels
            else:
                n_z = n_y = n_pixels
            self.pixel_size_z = self.zmax / n_z
            self.pixel_size_y = self.ymax / n_y
        elif pixel_size is not None:
            if isinstance(pixel_size, (tuple, list)):
                self.pixel_size_z = ensure_torch(pixel_size[0])
                self.pixel_size_y = ensure_torch(pixel_size[1])
            else:
                self.pixel_size_z = ensure_torch(pixel_size)
                self.pixel_size_y = ensure_torch(pixel_size)
        else:
            raise ValueError("Must specify either n_pixels=(n_z, n_y) or pixel_size")

        self.zdhat = (self.det_corner_2 - self.det_corner_0) / self.zmax
        self.ydhat = (self.det_corner_1 - self.det_corner_0) / self.ymax
        self.normal = torch.linalg.cross(self.zdhat, self.ydhat)
        self.normal = self.normal / torch.linalg.norm(self.normal)
        self.frames = []
        self.pixel_coordinates = self._get_pixel_coordinates()
        self.gaussian_sigma = gaussian_sigma
        self.kernel_threshold = kernel_threshold
        self._max_gaussian_kernel_radius = max_gaussian_kernel_radius
        self.gaussian_kernel = self._generate_gaussian_kernel()

        self.lorentz_factor = use_lorentz
        self.polarization_factor = use_polarization
        self.structure_factor = use_structure_factor

    def save(self, path: str) -> None:
        """Save detector to disk.

        Parameters
        ----------
        path : str
            Output file path. The ``.det`` extension is added if missing.
        """
        if not path.endswith(".det"):
            path = path + ".det"
        with open(path, "wb") as f:
            dill.dump(self, f, dill.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "Detector":
        """Load detector from disk.

        Parameters
        ----------
        path : str
            Path to ``.det`` file.

        Returns
        -------
        Detector
            Loaded Detector instance.

        Raises
        ------
        ValueError
            If file extension is not ``.det``.
        """
        if not path.endswith(".det"):
            raise ValueError("The loaded motion file must end with .det")
        with open(path, "rb") as f:
            loaded = dill.load(f)
            loaded.normal = ensure_torch(loaded.normal)
            loaded.det_corner_0 = ensure_torch(loaded.det_corner_0)
            loaded.det_corner_1 = ensure_torch(loaded.det_corner_1)
            loaded.det_corner_2 = ensure_torch(loaded.det_corner_2)
            loaded.zdhat = ensure_torch(loaded.zdhat)
            loaded.ydhat = ensure_torch(loaded.ydhat)
            loaded.zmax = ensure_torch(loaded.zmax)
            loaded.ymax = ensure_torch(loaded.ymax)
            loaded.pixel_size_z = ensure_torch(loaded.pixel_size_z)
            loaded.pixel_size_y = ensure_torch(loaded.pixel_size_y)
            return loaded

    # ------------------------------------------------------------------
    # 2. High-level public API
    # ------------------------------------------------------------------

    def render(
        self,
        peaks_dict: Dict[str, Union[torch.Tensor, List[str]]],
        frames_to_render: int = 1,
        method: str = "auto",
        macro_grain_limit=None,  # depends on the detector pixel size
        micro_grain_limit=0.1**3,  # in cubic microns
        render_dtype: torch.dtype | None = None,
        verbose: bool | None = None,
    ) -> tuple[torch.Tensor, Dict[str, Union[torch.Tensor, List[str]]]]:
        """Render diffraction frames using different peak profile methods.

        Three physically motivated rendering methods are provided, each
        targeting a different grain-size regime.  See the individual
        rendering methods for detailed physical justification.

        - ``'nano'``:  Airy-disk profiles for sub-micron crystallites
          where the finite-lattice assumption breaks down and the
          Fourier transform of the lattice is no longer a delta.
          See :meth:`_render_airy_peaks`.
        - ``'micro'``: Gaussian point-spread for medium grains (~0.1–10 µm)
          where the instrument response dominates the spot shape.
          See :meth:`_render_gauss_peaks`.
        - ``'macro'``: Volume projection for grains larger than the pixel
          size, capturing crystal morphology and strain.
          See :meth:`_render_projected_volumes`.
        - ``'auto'``:  Automatically selects per-peak based on grain volume
          (``macro_grain_limit`` / ``micro_grain_limit`` thresholds).

        Parameters
        ----------
        peaks_dict : dict
            Peak data dictionary containing ``'peaks'`` tensor and metadata
            columns.
        frames_to_render : int, optional
            Number of frames to generate. The diffraction time domain ``[0, 1]``
            is divided into ``frames_to_render`` equal bins, and each peak is
            assigned to the frame corresponding to its diffraction time. When
            set to 1, all peaks are rendered into a single integrated frame.
            Values less than 1 are clamped to 1. Default is 1.
        method : str, optional
            Rendering method: ``'micro'``, ``'nano'``, ``'macro'``, or ``'auto'``.
            Default is ``'micro'``.
        macro_grain_limit : float, optional
            Volume threshold (cubic microns) above which grains use the macro
            rendering method. Default is ``pixel_size**3``.
        micro_grain_limit : float, optional
            Volume threshold (cubic microns) below which grains use the nano
            rendering method. Default is ``0.001`` (0.1³ cubic microns).
        render_dtype : torch.dtype, optional
            Data type for rendered frames. If ``None``, uses the dtype of the
            peaks tensor.
        verbose : bool or None, optional
            If ``True``, print progress messages during rendering. If ``None``,
            inherits the ``verbose`` flag from ``peaks_dict`` (set by
            ``polycrystal.diffract``). Defaults to ``True`` if neither is set.

        Returns
        -------
        diffraction_frames : torch.Tensor
            Rendered diffraction frames with shape ``(frames, height, width)``.
        peaks_dict : dict
            Updated peak data dictionary with detector intersection columns
            (``zd``, ``yd``, ``incident_angle``, ``frame``,
            ``intensity_factors``) appended and non-visible / invalid peaks
            filtered out.  Identical to the output of
            :meth:`peaks_detector_intersection`.

        Raises
        ------
        ValueError
            If an invalid method is specified.
        """

        target_dtype = (
            render_dtype if render_dtype is not None else peaks_dict["peaks"].dtype
        )
        if verbose is None:
            verbose = peaks_dict.get("verbose", True)
        self._verbose = verbose  # Store for use in rendering methods

        if frames_to_render < 1:
            frames_to_render = 1
            if self._verbose:
                print("[Detector.render] frames_to_render < 1, setting to 1")

        peaks_dict = self.peaks_detector_intersection(
            peaks_dict, frames_to_render, verbose
        )
        peaks = peaks_dict["peaks"]
        beam = peaks_dict.get("beam", None)
        lab_mesh = peaks_dict.get("mesh_lab", None)
        rigid_body_motion = peaks_dict.get("rigid_body_motion", None)

        if method == "macro":
            diffraction_frames = self._render_projected_volumes(
                peaks, beam, lab_mesh, rigid_body_motion, frames_to_render, target_dtype
            )
        elif method == "micro":
            diffraction_frames = self._render_gauss_peaks(
                peaks, frames_to_render, target_dtype
            )
        elif method == "nano":
            diffraction_frames = self._render_airy_peaks(
                peaks, frames_to_render, target_dtype
            )
        elif method == "auto":
            if macro_grain_limit is None:
                macro_grain_limit = (min(self.pixel_size_y, self.pixel_size_z)) ** 3
            large_grains_mask = peaks[:, 21] >= macro_grain_limit
            small_grains_mask = peaks[:, 21] < micro_grain_limit
            medium_grains_mask = (~large_grains_mask) & (~small_grains_mask)

            diffraction_frames = torch.zeros(
                (
                    frames_to_render,
                    self.pixel_coordinates.shape[0],
                    self.pixel_coordinates.shape[1],
                ),
                device=peaks.device,
            )

            # Collect contributions (each returns frames_to_render-length tensors)
            contrib_macro = self._render_projected_volumes(
                peaks[large_grains_mask],
                beam,
                lab_mesh,
                rigid_body_motion,
                frames_to_render,
                target_dtype,
            )
            contrib_micro = self._render_gauss_peaks(
                peaks[medium_grains_mask], frames_to_render, target_dtype
            )
            contrib_nano = self._render_airy_peaks(
                peaks[small_grains_mask], frames_to_render, target_dtype
            )

            # Ensure dtype/device matching and add up
            diffraction_frames = diffraction_frames.to(contrib_nano.device)
            diffraction_frames += contrib_nano
            diffraction_frames += contrib_macro
            diffraction_frames += contrib_micro
        else:
            raise ValueError(
                f"Invalid method: {method}. Must be one of: 'micro', 'nano', 'macro', or 'auto'"
            )

        return diffraction_frames, peaks_dict

    def contains(self, zd: torch.Tensor, yd: torch.Tensor) -> torch.Tensor:
        """Check if detector coordinates are within bounds.

        Parameters
        ----------
        zd : torch.Tensor
            Z-axis detector coordinates.
        yd : torch.Tensor
            Y-axis detector coordinates.

        Returns
        -------
        torch.Tensor
            Boolean tensor indicating which coordinates are within detector
            bounds.
        """
        return (zd >= 0) & (zd <= self.zmax) & (yd >= 0) & (yd <= self.ymax)

    # ------------------------------------------------------------------
    # 3. Geometry & coordinate helpers
    # ------------------------------------------------------------------

    def _get_pixel_coordinates(self) -> torch.Tensor:
        """Calculate real-space coordinates for each detector pixel.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(Z, Y, 3)`` containing 3D coordinates for each
            pixel center.
        """
        # Get device from det_corner_0 to ensure all tensors are on same device
        dev = self.det_corner_0.device

        zds = torch.arange(0, self.zmax, self.pixel_size_z, device=dev)
        yds = torch.arange(0, self.ymax, self.pixel_size_y, device=dev)
        Z, Y = torch.meshgrid(zds, yds, indexing="ij")
        Zds = torch.zeros((len(zds), len(yds), 3), device=dev)
        Yds = torch.zeros((len(zds), len(yds), 3), device=dev)
        for i in range(3):
            Zds[:, :, i] = Z
            Yds[:, :, i] = Y
        pixel_coordinates = (
            self.det_corner_0.reshape(1, 1, 3)
            + Zds * self.zdhat.reshape(1, 1, 3)
            + Yds * self.ydhat.reshape(1, 1, 3)
        )
        return pixel_coordinates

    def _get_intersection(
        self, ray_direction: torch.Tensor, source_point: torch.Tensor
    ) -> torch.Tensor:
        """Get detector intersection coordinates for rays.

        Parameters
        ----------
        ray_direction : torch.Tensor
            Direction vectors for each ray, shape ``(N, 3)`` or ``(3,)``.
        source_point : torch.Tensor
            Origin points for each ray, shape ``(N, 3)`` or ``(3,)``.

        Returns
        -------
        torch.Tensor
            Intersection coordinates ``(zd, yd)`` and incident angles, shape
            ``(N, 3)``.
        """
        # Handle single vector inputs by adding batch dimension
        if ray_direction.dim() == 1:
            ray_direction = ray_direction.unsqueeze(0)
        if source_point.dim() == 1:
            source_point = source_point.unsqueeze(0)

        s = torch.matmul(self.det_corner_0 - source_point, self.normal) / torch.matmul(
            ray_direction, self.normal
        )

        intersection = source_point + ray_direction * s.unsqueeze(1)
        intersection[s < 0] = np.nan
        zd = torch.matmul(intersection - self.det_corner_0, self.zdhat)
        yd = torch.matmul(intersection - self.det_corner_0, self.ydhat)

        ray_dir_norm = ray_direction / torch.norm(ray_direction, dim=1).unsqueeze(-1)
        normal_norm = self.normal / torch.linalg.norm(self.normal)
        cosine_theta = torch.matmul(ray_dir_norm, -normal_norm)
        incident_angle_deg = torch.arccos(cosine_theta) * (180 / torch.pi)
        return torch.stack((zd, yd, incident_angle_deg), dim=1)

    def _get_wrapping_cone(
        self, k: torch.Tensor, source_point: torch.Tensor
    ) -> torch.Tensor:
        """Compute cone that wraps detector corners around wavevector.

        Parameters
        ----------
        k : torch.Tensor
            Central wavevector of cone, shape ``(3,)``.
        source_point : torch.Tensor
            Cone vertex point, shape ``(3,)``.

        Returns
        -------
        torch.Tensor
            Half-angle of cone opening in radians.
        """
        # Ensure k and source_point are torch tensors
        k = ensure_torch(k)
        source_point = ensure_torch(source_point)

        fourth_corner_of_detector = self.det_corner_2 + (
            self.det_corner_1 - self.det_corner_0[:]
        )
        geom_mat = torch.zeros((3, 4))
        for i, det_corner in enumerate(
            [
                self.det_corner_0,
                self.det_corner_1,
                self.det_corner_2,
                fourth_corner_of_detector,
            ]
        ):
            geom_mat[:, i] = det_corner - source_point
        normalised_local_coord_geom_mat = geom_mat / torch.linalg.norm(geom_mat, axis=0)
        cone_opening = torch.arccos(
            torch.matmul(normalised_local_coord_geom_mat.T, k / torch.linalg.norm(k))
        )
        return torch.max(cone_opening) / 2.0

    def _detector_coordinate_to_pixel_index(
        self, zd: float, yd: float
    ) -> Tuple[int, int]:
        """Convert detector coordinates to pixel indices.

        Parameters
        ----------
        zd : float
            Z-axis detector coordinate in microns.
        yd : float
            Y-axis detector coordinate in microns.

        Returns
        -------
        tuple of int
            ``(row_index, col_index)`` pixel indices.
        """
        row_index = int(zd / self.pixel_size_z)
        col_index = int(yd / self.pixel_size_y)
        return row_index, col_index

    def _get_projected_bounding_box(
        self, proj_context: Dict[str, Any]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Compute bounding detector pixel indices for a convex hull projection.

        Parameters
        ----------
        proj_context : dict
            Dictionary containing:

            - ``'convex_hull'``: ConvexHull object to project.
            - ``'scattered_wave_vector'``: Direction of scattered wave.

        Returns
        -------
        tuple of int or None
            ``(min_row, max_row, min_col, max_col)`` indices for slicing the
            detector frame array. Returns ``None`` if the projection does not
            intersect the detector.
        """
        # Get vertices from convex hull (numpy) and convert to torch
        hull = proj_context["convex_hull"]
        vertices = ensure_torch(hull.points[hull.vertices])

        # Reshape scattered_wave_vector to (N,3) by repeating for each vertex
        scattered_vec = proj_context["scattered_wave_vector"]
        scattered_vec = scattered_vec.unsqueeze(0).repeat(vertices.shape[0], 1)

        # Get intersections - stays in torch
        projected_vertices = self._get_intersection(scattered_vec, vertices)

        # Calculate bounds using torch operations
        zmax = self.zmax
        ymax = self.ymax

        # Get z and y columns, filter out NaN values for min/max
        z_vals = projected_vertices[:, 0]
        y_vals = projected_vertices[:, 1]
        z_valid = z_vals[~torch.isnan(z_vals)]
        y_valid = y_vals[~torch.isnan(y_vals)]

        if z_valid.numel() == 0 or y_valid.numel() == 0:
            return None

        min_zd = max(float(z_valid.min().item()), 0)
        max_zd = min(float(z_valid.max().item()), float(zmax))
        min_yd = max(float(y_valid.min().item()), 0)
        max_yd = min(float(y_valid.max().item()), float(ymax))

        if min_zd > max_zd or min_yd > max_yd:
            return None

        min_row_indx, min_col_indx = self._detector_coordinate_to_pixel_index(
            min_zd, min_yd
        )
        max_row_indx, max_col_indx = self._detector_coordinate_to_pixel_index(
            max_zd, max_yd
        )

        max_row_indx = min(max_row_indx + 1, int(zmax / self.pixel_size_z))
        max_col_indx = min(max_col_indx + 1, int(ymax / self.pixel_size_y))

        return min_row_indx, max_row_indx, min_col_indx, max_col_indx

    # ------------------------------------------------------------------
    # 4. Peak / frame preprocessing
    # ------------------------------------------------------------------

    def peaks_detector_intersection(
        self,
        incoming_wavevector,
        pixel_zd_coord,
        pixel_yd_coord,
        scattering_origin=np.array([0, 0, 0]),
    ):
        """Compute bragg angle and azimuth angle  for a detector coordinate.

        Args:
            pixel_zd_coord (:obj:`float`): Coordinate in microns along detector zd axis.
            pixel_yd_coord (:obj:`float`): Coordinate in microns along detector yd axis.
            scattering_origin (obj:`numpy array`): Origin of diffraction in microns. Defaults to np.array([0, 0, 0]).

        Returns:
            (:obj:`tuple`) Bragg angle theta and azimuth angle eta (measured from det_corner_1 - det_corner_0 axis) in radians
        """
        # TODO: unit test
        khat = incoming_wavevector / np.linalg.norm(incoming_wavevector)
        kp = (
            self.det_corner_0
            + pixel_zd_coord * self.zdhat
            + pixel_yd_coord * self.ydhat
            - scattering_origin
        )
        kprimehat = kp / np.linalg.norm(kp)
        theta = np.arccos(khat.dot(kprimehat)) / 2.0
        korthogonal = kprimehat - (khat * kprimehat.dot(khat))
        eta = np.arccos(self.zdhat.dot(korthogonal) / np.linalg.norm(korthogonal))
        eta *= np.sign((np.cross(self.zdhat, korthogonal)).dot(-incoming_wavevector))
        return theta, eta

    def get_intersection(self, ray_direction, source_point):
        """Get detector intersection in detector coordinates of a single ray originating from source_point.

        Args:
            ray_direction (:obj:`numpy array`): Vector in direction of the xray propagation
            source_point (:obj:`numpy array`): Origin of the ray.

        Returns:
            (:obj:`tuple`) zd, yd in detector plane coordinates.

        """

        s = (self.det_corner_0 - source_point).dot(self.normal) / ray_direction.dot(
            self.normal
        )

        intersection = source_point + ray_direction * s[:, np.newaxis]

        # such that backwards rays are not considered to intersect the detector
        # i.e only rays that can intersect the detector plane by propagating
        # forward along the photon path are considered.
        intersection[s < 0] = np.nan

        zd = np.dot(intersection - self.det_corner_0, self.zdhat)
        yd = np.dot(intersection - self.det_corner_0, self.ydhat)
        return np.array([zd, yd]).T

    def contains(self, zd, yd):
        """Determine if the detector coordinate zd,yd lies within the detector bounds.

        Args:
            zd (:obj:`float`): Detector z coordinate
            yd (:obj:`float`): Detector y coordinate

        Returns:
            (:obj:`boolean`) True if the zd,yd is within the detector bounds.

        """

        return (zd >= 0) & (zd <= self.zmax) & (yd >= 0) & (yd <= self.ymax)

    def project(self, scattering_unit, box):
        """Compute parametric projection of scattering region unto detector.

        Args:
            scattering_unit (:obj:`xrd_simulator.ScatteringUnit`): The scattering region.
            box (:obj:`tuple` of :obj:`int`): indices of the detector frame over which to compute the projection.
                i.e the subgrid of the detector is taken as: array[[box[0]:box[1], box[2]:box[3]].

        Returns:
            (:obj:`numpy array`) clip lengths between scattering_unit polyhedron and rays traced from the detector.

        """

        ray_points = self.pixel_coordinates[
            box[0] : box[1], box[2] : box[3], :
        ].reshape((box[1] - box[0]) * (box[3] - box[2]), 3)

        plane_normals = scattering_unit.convex_hull.equations[:, 0:3]
        plane_ofsets = scattering_unit.convex_hull.equations[:, 3].reshape(
            scattering_unit.convex_hull.equations.shape[0], 1
        )
        plane_points = -np.multiply(plane_ofsets, plane_normals)

        ray_points = np.ascontiguousarray(ray_points)
        ray_direction = np.ascontiguousarray(
            scattering_unit.scattered_wave_vector
            / np.linalg.norm(scattering_unit.scattered_wave_vector)
        )
        plane_points = np.ascontiguousarray(plane_points)
        plane_normals = np.ascontiguousarray(plane_normals)

        clip_lengths = utils._clip_line_with_convex_polyhedron(
            ray_points, ray_direction, plane_points, plane_normals
        )
        clip_lengths = clip_lengths.reshape(box[1] - box[0], box[3] - box[2])

        return clip_lengths

    def get_wrapping_cone(self, k, source_point):
        """Compute the cone around a wavevector such that the cone wraps the detector corners.

        Args:
            k (:obj:`numpy array`): Wavevector forming the central axis of cone ```shape=(3,)```.
            source_point (:obj:`numpy array`): Origin of the wavevector ```shape=(3,)```.

        Returns:
            (:obj:`float`) Cone opening angle divided by two (radians), corresponding to a maximum bragg angle after
                which scattering will systematically miss the detector.

        """
        fourth_corner_of_detector = self.det_corner_2 + (
            self.det_corner_1 - self.det_corner_0[:]
        )
        geom_mat = np.zeros((3, 4))
        for i, det_corner in enumerate(
            [
                self.det_corner_0,
                self.det_corner_1,
                self.det_corner_2,
                fourth_corner_of_detector,
            ]
        ):
            geom_mat[:, i] = det_corner - source_point
        normalised_local_coord_geom_mat = geom_mat / np.linalg.norm(geom_mat, axis=0)
        cone_opening = np.arccos(
            np.dot(normalised_local_coord_geom_mat.T, k / np.linalg.norm(k))
        )  # These are two time Bragg angles
        return np.max(cone_opening) / 2.0

    def save(self, path):
        """Save the detector object to disc (via pickling).

        Args:
            path (:obj:`str`): File path at which to save, ending with the desired filename.

        """
        if not path.endswith(".det"):
            path = path + ".det"
        with open(path, "wb") as f:
            dill.dump(self, f, dill.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        """Load the detector object from disc (via pickling).

        Args:
            path (:obj:`str`): File path at which to load, ending with the desired filename.

        .. warning::
            This function will unpickle data from the provied path. The pickle module
            is not intended to be secure against erroneous or maliciously constructed data.
            Never unpickle data received from an untrusted or unauthenticated source.

        """
        if not path.endswith(".det"):
            raise ValueError("The loaded motion file must end with .det")
        with open(path, "rb") as f:
            return dill.load(f)

    def _get_point_spread_function_kernel(self):
        """Render the point_spread_function onto a grid of shape specified by point_spread_kernel_shape."""
        sz, sy = self.point_spread_kernel_shape
        axz = np.linspace(-(sz - 1) / 2.0, (sz - 1) / 2.0, sz)
        axy = np.linspace(-(sy - 1) / 2.0, (sy - 1) / 2.0, sy)
        Z, Y = np.meshgrid(axz, axy, indexing="ij")
        kernel = np.zeros(self.point_spread_kernel_shape)
        for i in range(Z.shape[0]):
            for j in range(Y.shape[1]):
                kernel[i, j] = self.point_spread_function(Z[i, j], Y[i, j])

        assert (
            len(kernel[kernel < 0]) == 0
        ), "Point spread function must be strictly positive, but negative values were found."
        assert (
            np.sum(kernel) > 1e-8
        ), "The integrated value of the point spread function over the defined kernel domain is close to zero."

        return kernel / np.sum(kernel)

    def _get_pixel_coordinates(self):
        zds = np.arange(0, self.zmax, self.pixel_size_z)
        yds = np.arange(0, self.ymax, self.pixel_size_y)
        Z, Y = np.meshgrid(zds, yds, indexing="ij")
        Zds = np.zeros((len(zds), len(yds), 3))
        Yds = np.zeros((len(zds), len(yds), 3))
        for i in range(3):
            Zds[:, :, i] = Z
            Yds[:, :, i] = Y
        pixel_coordinates = (
            self.det_corner_0.reshape(1, 1, 3)
            + Zds * self.zdhat.reshape(1, 1, 3)
            + Yds * self.ydhat.reshape(1, 1, 3)
        )
        return pixel_coordinates

    def _centroid_render(
        self, scattering_unit, frame, lorentz, polarization, structure_factor
    ):
        """Simple deposit of intensity for each scattering_unit onto the detector by tracing a line from the
        sample scattering region centroid to the detector plane. The intensity is deposited into a single
        detector pixel regardless of the geometrical shape of the scattering_unit.
        """
        zd, yd = scattering_unit.zd, scattering_unit.yd

        if self.contains(zd, yd):
            intensity_scaling_factor = self._get_intensity_factor(
                scattering_unit, lorentz, polarization, structure_factor
            )
            row, col = self._detector_coordinate_to_pixel_index(zd, yd)
            if np.isinf(intensity_scaling_factor):
                frame[row, col] += np.inf
            else:
                frame[row, col] += scattering_unit.volume * intensity_scaling_factor

    def _centroid_render_with_scintillator(
        self, scattering_unit, frame, lorentz, polarization, structure_factor
    ):
        """Simple deposit of intensity for each scattering_unit onto the detector by tracing a line from the
        sample scattering region centroid to the detector plane. The intensity is deposited by placing the detector
        point spread function at the hit location and rendering it unto the detector grid.

        NOTE: this is different from self._centroid_render which applies the point spread function as a post-proccessing
        step using convolution. Here the point spread is simulated to take place in the scintillator, before reaching the
        chip.
        """
        detector_distance = torch.linalg.norm(self.det_corner_0)
        incident_angles_rad = incident_angles * torch.pi / 180
        R = detector_distance / torch.cos(incident_angles_rad)

        # Convert FWHM in radians to pixel scale
        fwhm_pixels = fwhm_rad * R / self.pixel_size_z

        # The Airy pattern has FWHM ≈ 0.843 * first_zero_radius
        # Therefore: first_zero_radius = FWHM / 0.843 ≈ FWHM * 1.1861
        # This ensures the rendered Airy disk has the correct FWHM for Scherrer recovery
        AIRY_FWHM_TO_FIRST_ZERO = 1.1861  # = 1 / 0.843
        first_zero_pixels = fwhm_pixels * AIRY_FWHM_TO_FIRST_ZERO

        sigma = self.gaussian_sigma
        max_first_zero = first_zero_pixels.max()
        threshold = self.kernel_threshold / 2

        # Calculate kernel radius - need to capture several Airy rings
        # Airy pattern decays as 1/x^3 in the tails, so we need larger extent
        # for small crystallites (broad patterns)
        # Use ~4 times the first zero to capture sufficient rings
        radius = torch.tensor(max(4.0 * float(max_first_zero), 8.0))

        # Ensure we also capture Gaussian tails (cap at 128 pixels)
        max_airy_radius = 128
        while radius <= max_airy_radius:
            g_val = torch.exp(-(radius**2) / (2 * sigma**2))
            if g_val < threshold:
                break
            radius = radius * 2
        if radius > max_airy_radius:
            if self._verbose:
                print(
                    f"Airy kernel radius should be {radius.item()} but limited to {max_airy_radius}"
                )
            radius = torch.tensor(max_airy_radius)

        ax = torch.arange(-radius, radius + 1, dtype=torch.float64)
        yy, zz = torch.meshgrid(ax, ax, indexing="ij")
        r = torch.sqrt(yy**2 + zz**2)

        kernels = []
        max_size = 0

        # First zero of J_1(x) occurs at x ≈ 3.8317
        FIRST_ZERO_J1 = 3.8317

        def airy_profile(
            r: npt.NDArray, first_zero: float, sigma: float
        ) -> npt.NDArray:
            """Compute Airy disk profile convolved with Gaussian PSF.

            The Airy intensity pattern is: ``I(x) = [2*J_1(x)/x]^2``
            where ``x = 3.8317 * r / first_zero_radius``.

            The first zero of J_1(x) is at x = 3.8317, so the pattern's
            first zero occurs at r = first_zero_radius.

            For r=0, the pattern has value 1 (use L'Hôpital's rule).

            Parameters
            ----------
            r : numpy.ndarray
                Radial distances in pixels.
            first_zero : float
                First zero radius of Airy pattern in pixels.
            sigma : float
                Gaussian width for instrument broadening within the Airy kernel.

            Returns
            -------
            numpy.ndarray
                Computed Airy disk values (normalized later).
            """
            # Scale r so that first zero occurs at r = first_zero
            # x = 3.8317 * r / first_zero, so at r = first_zero, x = 3.8317
            x = np.where(r == 0, 1e-10, FIRST_ZERO_J1 * r / first_zero)

            # Airy pattern: [2*J_1(x)/x]^2
            # Handle small x separately to avoid numerical issues
            airy = np.where(
                np.abs(x) < 1e-8,
                1.0,  # At center, the jinc function approaches 1
                (2 * j1(x) / x) ** 2,
            )

            # For very small crystallites (broad patterns), the Airy disk dominates
            # For large crystallites (narrow patterns), convolve with Gaussian PSF
            if first_zero < sigma:
                # Pattern is sharper than PSF - convolve with Gaussian
                # Simple approximation: multiply by Gaussian to smooth
                gaussian = np.exp(-(r**2) / (2 * sigma**2))
                return airy * gaussian
            else:
                return airy

        r_np = r.cpu().numpy()
        sigma_np = float(sigma)

        for first_zero in first_zero_pixels:
            first_zero_np = float(first_zero)

            # For very large crystallites (very small first_zero), use Gaussian kernel
            if first_zero_np < 0.5:
                kernel = self.gaussian_kernel
            else:
                kernel = ensure_torch(airy_profile(r_np, first_zero_np, sigma_np))

            kernel = kernel / kernel.sum()

            # Crop to significant region using binary search
            center = kernel.shape[0] // 2
            left, right = 1, center
            while left < right:
                mid = (left + right) // 2
                window = kernel[
                    center - mid : center + mid + 1, center - mid : center + mid + 1
                ]
                if window.sum() >= 1 - self.kernel_threshold:
                    right = mid
                else:
                    left = mid + 1

            crop_r = left
            cropped = kernel[
                center - crop_r : center + crop_r + 1,
                center - crop_r : center + crop_r + 1,
            ]

            kernels.append(cropped.unsqueeze(0).unsqueeze(0))
            max_size = max(max_size, 2 * crop_r + 1)

        # Pad all kernels to same size
        padded_kernels = []
        for k in kernels:
            if k.shape[-1] < max_size:
                pad = (max_size - k.shape[-1]) // 2
                k_padded = F.pad(k, (pad, pad, pad, pad), mode="constant", value=0)
                padded_kernels.append(k_padded.to(device=first_zero_pixels.device))
            else:
                padded_kernels.append(k.to(device=first_zero_pixels.device))

        result = torch.cat(padded_kernels, dim=0)
        return result

    def _deposit_kernels_batch(
        self,
        tensor: torch.Tensor,
        kernels: torch.Tensor,
        centers_z: torch.Tensor,
        centers_y: torch.Tensor,
        render_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Deposit multiple kernels onto a tensor with sub-pixel precision.

        Uses bilinear interpolation (4 neighbors) for sub-pixel positioning,
        which is memory-efficient while preserving positional accuracy.

        Parameters
        ----------
        tensor : torch.Tensor
            Target tensor to deposit onto, shape ``(H, W)``.
        kernels : torch.Tensor
            Batch of kernels to deposit, shape ``(N, 1, H, W)``.
        centers_z : torch.Tensor
            Z coordinates for kernel centers (in pixels, sub-pixel precision).
        centers_y : torch.Tensor
            Y coordinates for kernel centers (in pixels, sub-pixel precision).
        render_dtype : torch.dtype
            Data type for output values.

        Returns
        -------
        torch.Tensor
            Updated tensor with deposited kernels.
        """
        N = kernels.shape[0]
        kh, kw = kernels.shape[-2:]
        half_h, half_w = kh // 2, kw // 2
        H, W = tensor.shape

        # Create kernel pixel offsets (relative to center)
        offsets_z = (
            torch.arange(kh, device=kernels.device, dtype=torch.float64) - half_h
        )
        offsets_y = (
            torch.arange(kw, device=kernels.device, dtype=torch.float64) - half_w
        )

        # Build all sub-pixel positions: (N, kh, kw)
        all_z = centers_z.view(N, 1, 1) + offsets_z.view(1, kh, 1)
        all_y = centers_y.view(N, 1, 1) + offsets_y.view(1, 1, kw)

        # Expand to full grid and flatten
        all_z = all_z.expand(N, kh, kw).reshape(-1)
        all_y = all_y.expand(N, kh, kw).reshape(-1)

        # Flatten kernel values
        values = kernels.view(N, -1).reshape(-1)

        # Bilinear interpolation: split into integer and fractional parts
        z0 = all_z.floor().long()
        y0 = all_y.floor().long()
        z1 = z0 + 1
        y1 = y0 + 1

        # Fractional parts for weighting
        fz = all_z - z0.float()
        fy = all_y - y0.float()

        # Bilinear weights for 4 corners
        w00 = (1 - fz) * (1 - fy)  # top-left
        w01 = (1 - fz) * fy  # top-right
        w10 = fz * (1 - fy)  # bottom-left
        w11 = fz * fy  # bottom-right

        # Deposit to all 4 corners with their weights
        for zi, yi, w in [(z0, y0, w00), (z0, y1, w01), (z1, y0, w10), (z1, y1, w11)]:
            valid = (zi >= 0) & (zi < H) & (yi >= 0) & (yi < W)
            if valid.any():
                tensor.index_put_(
                    (zi[valid], yi[valid]),
                    (values[valid] * w[valid]).to(render_dtype),
                    accumulate=True,
                )

        return tensor

    # 5.c Projected volumes
    def _render_projected_volumes(
        self,
        peaks,
        beam,
        mesh_lab,
        rigid_body_motion,
        frames_to_render,
        render_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Render projected crystal volumes — the ``'macro'`` method.

        **Physical motivation (grains larger than the pixel size)**

        When a grain's linear dimension exceeds the detector pixel size,
        its diffraction spot is no longer point-like: different parts of
        the crystal illuminate different detector pixels, so the observed
        peak reproduces the *geometric shadow* of the illuminated crystal
        volume projected along the scattered wave-vector.  The per-pixel
        intensity is proportional to the path length of the scattered ray
        through the grain, i.e. the diffracting volume behind that pixel.

        This regime is relevant for coarse-grained metals and geological
        samples where individual grains span tens to hundreds of microns.
        The resulting images capture grain morphology, intra-granular
        orientation gradients, and spatial strain variations that are
        invisible to the point-like methods.

        **Implementation**

        For each peak the grain's tetrahedral mesh is rotated to its
        diffraction time, intersected with the beam footprint to form a
        convex hull, and then ray-traced: parallel rays along the
        scattered wave-vector are clipped against the hull to obtain
        per-pixel path lengths.  A Gaussian PSF convolution is applied
        afterwards.

        Parameters
        ----------
        peaks : torch.Tensor
            Processed peaks tensor with detector intersections and intensity
            factors.
        beam : Beam
            Beam object for computing convex hull intersections.
        mesh_lab : Mesh
            Mesh object containing tetrahedral element geometry.
        rigid_body_motion : RigidBodyMotion
            RigidBodyMotion object for transforming vertices.
        frames_to_render : int
            Number of frames to generate.
        render_dtype : torch.dtype
            Data type for rendered frames.

        Returns
        -------
        torch.Tensor
            Rendered diffraction frames with projected volumes, shape
            ``(frames, height, width)``.
        """
        frames_bundle = peaks[:, 28].unique()  # frame column is index 28

        # Early exit for empty peak list
        if peaks.shape[0] == 0:
            return torch.zeros(
                (
                    frames_to_render,
                    self.pixel_coordinates.shape[0],
                    self.pixel_coordinates.shape[1],
                ),
                device=self.det_corner_0.device,
                dtype=render_dtype,
            )

        device = peaks.device
        scattered_vectors = peaks[:, 13:16]

        # Get element indices from peaks (column 0: grain_index which maps to element)
        element_indices = peaks[:, 0].long()
        times = peaks[:, 6]
        # Initialize all frames to zeros so we can write directly by frame index
        diffraction_frames = torch.zeros(
            (
                frames_to_render,
                self.pixel_coordinates.shape[0],
                self.pixel_coordinates.shape[1],
            ),
            device=device,
            dtype=render_dtype,
        )

        # Get vertices for each peak and apply rigid body motion
        node_indices = mesh_lab.enod[element_indices]
        vertices = ensure_torch(mesh_lab.coord[node_indices])  # Shape: (N_peaks, 4, 3)

        # Apply rigid body motion to all vertices at their corresponding times
        # RigidBodyMotion handles (N, 4, 3) shape with per-vector times (N,)
        rotated_vertices = rigid_body_motion(vertices, times)  # Shape: (N_peaks, 4, 3)

        total_peaks = peaks.shape[0]
        processed_peaks = 0

        for frame_index in frames_bundle:
            # Initialize frame on same device as peaks
            frames = torch.zeros(
                (self.pixel_coordinates.shape[0], self.pixel_coordinates.shape[1]),
                device=device,
                dtype=render_dtype,
            )

            # Process each peak in this frame
            frame_mask = peaks[:, 28] == frame_index
            frame_peak_indices = torch.where(frame_mask)[0]

            for i, peak_idx in enumerate(frame_peak_indices):
                processed_peaks += 1
                # Show progress for verbose mode using a single continuous bar
                if self._verbose and total_peaks > 0:
                    progress_fraction = processed_peaks / total_peaks
                    utils._print_progress(
                        progress_fraction,
                        f"[Macro] Peak {processed_peaks}/{total_peaks}",
                    )

                # Compute convex hull intersection with beam
                hull = beam._intersect(rotated_vertices[peak_idx])

                if hull is not None:
                    # Create minimal projection context
                    proj_context = {
                        "convex_hull": hull,
                        "scattered_wave_vector": scattered_vectors[peak_idx],
                    }

                    # Project volume using only necessary data
                    box = self._get_projected_bounding_box(proj_context)
                    if box is not None:
                        # Keep projection in numpy for computation
                        projection = self._project_convex_hull(proj_context, box)
                        # Use pre-calculated intensity from peaks tensor
                        intensity = peaks[peak_idx, 29]  # intensity column is index 29

                        # Note: Infinite intensity peaks should already be filtered out
                        # by peaks_detector_intersection. This check is defensive.
                        if torch.isfinite(intensity):
                            projection_torch = ensure_torch(projection)
                            # projection is path length, multiply by pixel area to get volume
                            projected_volume = (
                                projection_torch * self.pixel_size_z * self.pixel_size_y
                            )
                            # Multiply intensity per volume by projected volume
                            result = (intensity * projected_volume).to(render_dtype)
                            frames[box[0] : box[1], box[2] : box[3]] += result

            # Apply Gaussian convolution for point spread function
            frames = self._conv2d_gaussian_kernel(frames)
            diffraction_frames[frame_index.long()] = frames

        return diffraction_frames

    def _project_convex_hull(
        self, proj_context: Dict[str, Any], box: Tuple[int, int, int, int]
    ) -> npt.NDArray:
        """Project convex hull onto detector region.

        Swithces between a sample driven and detector driven approcimation depending on
        the detector pixel size and the convex hull size. When the convex hull is large
        compared to the pixel size, the detector driven approach is used. This means that
        rays traced from pixel centroids are used as projection lines. When the convex
        hull is small compared to the pixel size, the sample driven approach is used. This
        means that points are sampled from the interior of the hull and used as projection lines.

        Parameters
        ----------
        proj_context : dict
            Dictionary containing:

            - ``'convex_hull'``: ConvexHull object to project.
            - ``'scattered_wave_vector'``: Direction of scattered wave.

        box : tuple of int
            ``(min_z, max_z, min_y, max_y)`` bounds for projection region.

        Returns
        -------
        torch.Tensor
            Array of projected fractional contribution to each detector pixel
            in the box. The sum total contribution is proportional to the
            volume of the convex hull.
        """

        # ConvexHull from scipy returns numpy, convert to torch
        hull = proj_context["convex_hull"]
        plane_normals = ensure_torch(hull.equations[:, 0:3])
        plane_offsets = ensure_torch(hull.equations[:, 3]).reshape(
            hull.equations.shape[0], 1
        )
        plane_points = -plane_offsets * plane_normals

        n_pixels_in_box = (box[1] - box[0]) * (box[3] - box[2])

        # Ray direction in torch
        scattered_vec = proj_context["scattered_wave_vector"]
        ray_direction = scattered_vec / torch.linalg.norm(scattered_vec)

        if n_pixels_in_box == 1:
            # tetra is fully contained in a detector pixel, use the volume of the tetra as intensity
            return torch.ones(1, 1) * proj_context["convex_hull"].volume
        elif (
            n_pixels_in_box <= 4
        ):  # tetra is small compared to the pixel size, adapt to sampling approach.
            # This is a decent approximation when the tetras are small in relation to the pixel size
            n_samples = (
                7 * n_pixels_in_box
            )  # TODO the ad-hoc number 7 here should perhaps be a parameter to give controll over the accuracy of the statistics.
            ray_points = utils._sample_convex_hull_3d(
                proj_context["convex_hull"], n_samples
            )
            ray_points.reshape(n_samples, 3)
            ray_points = ensure_torch(ray_points)

            clip_lengths = utils._clip_line_with_convex_polyhedron(
                ray_points, ray_direction, plane_points, plane_normals
            )
            # now map the clip lengths to the detector pixels
            ray_directions = torch.tile(ray_direction, (n_samples, 1))
            out = self._get_intersection(ray_directions, ray_points)
            zd = out[:, 0]
            yd = out[:, 1]

            row_index = ensure_numpy(zd / self.pixel_size_z).astype(int) - box[0]
            col_index = ensure_numpy(yd / self.pixel_size_y).astype(int) - box[2]

            box_intensities = np.zeros((box[1] - box[0], box[3] - box[2]))
            np.add.at(box_intensities, (row_index, col_index), clip_lengths)
            box_intensities /= box_intensities.sum()
            box_intensities *= proj_context["convex_hull"].volume

            return ensure_torch(box_intensities)

        else:
            # This is a good approximation when the tetras are large in relation to the pixel size
            # Get pixel coordinates and project on pixel centorid ray-lines
            pixel_coords = self.pixel_coordinates[box[0] : box[1], box[2] : box[3], :]
            ray_points = pixel_coords.reshape((box[1] - box[0]) * (box[3] - box[2]), 3)

            clip_lengths = utils._clip_line_with_convex_polyhedron(
                ray_points, ray_direction, plane_points, plane_normals
            )

            # ensure that scattering is proportional to the material volume
            # that is scattering.
            clip_lengths /= clip_lengths.sum()
            clip_lengths *= proj_context["convex_hull"].volume
            clip_lengths = clip_lengths.reshape(box[1] - box[0], box[3] - box[2])

        return clip_lengths

    # ------------------------------------------------------------------
    # 6. Convolution / kernel helpers
    # ------------------------------------------------------------------
    def _conv2d_gaussian_kernel(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply the point spread function to detector frames using Gaussian convolution.

        Parameters
        ----------
        frames : torch.Tensor
            Input frames to convolve, shape ``(H, W)``, ``(N, H, W)``, or
            ``(N, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Convolved frames with same shape as input.
        """
        frames = ensure_torch(frames)

        input_dims = frames.ndim
        if input_dims == 2:
            frames = frames.unsqueeze(0).unsqueeze(0)
        elif input_dims == 3:
            frames = frames.unsqueeze(1)

        kernel = self.gaussian_kernel.to(frames.dtype).unsqueeze(0).unsqueeze(0)
        kernel_padding = kernel.shape[-1] // 2

        with torch.no_grad():
            output = torch.nn.functional.conv2d(
                frames, weight=kernel, padding=kernel_padding
            )

        if input_dims == 2:
            output = output.squeeze(0).squeeze(0)
        elif input_dims == 3:
            output = output.squeeze(1)

        return output

    def _generate_gaussian_kernel(self) -> torch.Tensor:
        """Generate a normalized 2D Gaussian kernel with dynamic size.

        The kernel size is determined by the ``kernel_threshold`` attribute,
        expanding until the Gaussian tail drops below the threshold.

        Returns
        -------
        torch.Tensor
            2D Gaussian kernel of shape ``(H, W)``, normalized to sum to 1.
        """
        # Get device from det_corner_0 to ensure kernel is on same device
        dev = self.det_corner_0.device

        sigma = ensure_torch(self.gaussian_sigma).to(dev)
        threshold = self.kernel_threshold

        radius = 1
        while radius <= self._max_gaussian_kernel_radius:
            value = torch.exp(-(radius**2) / (2 * sigma**2))
            if value < threshold / 2:
                break
            radius += 1
        radius = min(radius, self._max_gaussian_kernel_radius)

        kernel_size = 2 * radius + 1

        ax = torch.arange(kernel_size, dtype=torch.float64, device=dev) - radius
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= kernel.sum()

        return kernel  # Return as (H,W) instead of (1,1,H,W)

    # ------------------------------------------------------------------
    # 7. Deprecated methods (clearly separated)
    # ------------------------------------------------------------------
    # DEPRECATED METHODS - TO BE REMOVED IN FUTURE VERSION

    def _ray_detector_intersection(
        self, origin: np.ndarray, direction: np.ndarray
    ) -> Optional[np.ndarray]:
        """Find where a ray intersects the detector plane.

        .. deprecated::
            This method is deprecated and will be removed in a future version.

        Parameters
        ----------
        origin : numpy.ndarray
            Ray origin point, shape ``(3,)``.
        direction : numpy.ndarray
            Normalized ray direction, shape ``(3,)``.

        Returns
        -------
        numpy.ndarray or None
            3D intersection point, or ``None`` if no intersection exists.
        """
        # Detector plane equation: dot(point - det_corner_0, normal) = 0
        # Ray equation: point = origin + t * direction

        normal = ensure_numpy(self.normal)
        corner = ensure_numpy(self.det_corner_0)

        denom = np.dot(direction, normal)

        if abs(denom) < 1e-10:
            return None  # Ray parallel to plane

        t = np.dot(corner - origin, normal) / denom

        if t < 0:
            return None  # Intersection behind ray origin

        intersection = origin + t * direction
        return intersection

    def _world_to_pixel_coords(
        self, world_point: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Convert 3D world coordinates to 2D pixel coordinates.

        .. deprecated::
            This method is deprecated and will be removed in a future version.

        Parameters
        ----------
        world_point : numpy.ndarray
            3D point in lab frame, shape ``(3,)``.

        Returns
        -------
        tuple of float or None
            ``(z_pixel, y_pixel)`` coordinates, or ``None`` if outside detector.
        """
        corner = ensure_numpy(self.det_corner_0)
        zdhat = ensure_numpy(self.zdhat)
        ydhat = ensure_numpy(self.ydhat)

        # Vector from detector corner to point
        v = world_point - corner

        # Project onto detector axes
        z_coord = np.dot(v, zdhat)
        y_coord = np.dot(v, ydhat)

        # Convert to pixel coordinates
        z_pixel = z_coord / float(self.pixel_size_z)
        y_pixel = y_coord / float(self.pixel_size_y)

        # Check bounds
        if (
            z_pixel < 0
            or z_pixel >= self.pixel_coordinates.shape[0]
            or y_pixel < 0
            or y_pixel >= self.pixel_coordinates.shape[1]
        ):
            return None

        return (z_pixel, y_pixel)

    def _compute_pixel_path_length(
        self, ray_point: np.ndarray, ray_direction: np.ndarray, hull
    ) -> float:
        """Compute exact path length of ray through convex hull.

        .. deprecated::
            This method is deprecated and will be removed in a future version.

        Parameters
        ----------
        ray_point : numpy.ndarray
            3D starting point of ray (pixel position).
        ray_direction : numpy.ndarray
            Normalized ray direction (scattered wave direction).
        hull : ConvexHull
            ConvexHull object from beam intersection.

        Returns
        -------
        float
            Path length through hull in microns.
        """
        # Use existing utility function for ray-polyhedron intersection
        from scipy.spatial import ConvexHull as SciPyConvexHull

        # Get hull faces (planes)
        hull_points = ensure_numpy(hull.points)
        try:
            scipy_hull = SciPyConvexHull(hull_points)
        except:
            return 0.0

        # Get plane normals and points
        plane_normals = scipy_hull.equations[:, :3]  # (n_faces, 3)
        plane_offsets = scipy_hull.equations[:, 3]  # (n_faces,)

        # Plane points (any point on each plane)
        plane_points = hull_points[scipy_hull.simplices[:, 0]]  # (n_faces, 3)

        # Convert to torch for utility function
        ray_point_torch = ensure_torch(ray_point).unsqueeze(0)  # (1, 3)
        ray_direction_torch = ensure_torch(ray_direction).unsqueeze(0)  # (1, 3)
        plane_points_torch = ensure_torch(plane_points)
        plane_normals_torch = ensure_torch(plane_normals)

        # Compute clip length
        clip_lengths = utils._clip_line_with_convex_polyhedron(
            ray_point_torch,
            ray_direction_torch,
            plane_points_torch,
            plane_normals_torch,
        )

        return float(clip_lengths[0])
