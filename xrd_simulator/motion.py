"""The motion module is used to represent a rigid body motion.

During diffraction from a :class:`xrd_simulator.polycrystal.Polycrystal`, the
:class:`xrd_simulator.motion.RigidBodyMotion` object describes how the sample
is translating and rotating. The motion can be used to update the polycrystal
position via the :func:`xrd_simulator.polycrystal.Polycrystal.transform`
function.

Examples
--------
Here is a minimal example of how to instantiate a rigid body motion object,
apply the motion to a pointcloud and save the motion to disc:

.. literalinclude:: examples/example_init_motion.py

Below follows a detailed description of the RigidBodyMotion class attributes
and functions.
运动模块用于表示刚体运动。在 :class:`xrd_simulator.polycrystal.Polycrystal` 的衍射过程中，
:class:`xrd_simulator.motion.RigidBodyMotion` 对象描述了样品如何平移和旋转。该运动可通过
:func:`xrd_simulator.polycrystal.Polycrystal.transform` 函数来更新多晶体的位置。

以下是如何实例化一个刚体运动对象、将运动应用于点云并将运动保存到磁盘的最小示例：

    示例：
        .. literalinclude:: examples/example_init_motion.py

下面是 RigidBodyMotion 类属性和函数的详细描述。
"""

import dill
import torch
from xrd_simulator.utils import ensure_torch

torch.set_default_dtype(torch.float64)


class RigidBodyMotion:
    """Rigid body transformation by Euler axis rotation and translation.

    A rigid body motion is defined in the laboratory coordinates system.

    The motion is parametric in the interval ``time=[0, 1]`` and will perform a
    rigid body transformation of a point ``x`` by linearly uniformly rotating
    it from ``[0, rotation_angle]`` and translating ``[0, translation]``. I.e.,
    if called at a time ``time=t``, the motion will first rotate the point
    ``t * rotation_angle`` radians around ``rotation_axis`` and next translate
    the point by the vector ``t * translation``.

    Parameters
    ----------
    rotation_axis : numpy.ndarray or torch.Tensor
        Rotation axis, shape ``(3,)``.
    rotation_angle : float
        Radians for final rotation when ``time=1``.
    translation : numpy.ndarray or torch.Tensor
        Translation vector, shape ``(3,)``.
    origin : numpy.ndarray or torch.Tensor, optional
        Point in space about which the rigid body motion is defined. Defaults
        to the origin ``(0, 0, 0)``. All translations are executed in relation
        to the origin and all rotations are rotations about the point of origin.
        Shape ``(3,)``.

    Attributes
    ----------
    rotation_axis : torch.Tensor
        Rotation axis, shape ``(3,)``.
    rotation_angle : torch.Tensor
        Radians for final rotation when ``time=1``.
    translation : torch.Tensor
        Translation vector, shape ``(3,)``.
    origin : torch.Tensor
        Point in space about which the rigid body motion is defined,
        shape ``(3,)``.            
    通过欧拉轴旋转和平移对欧几里得点进行刚体变换。

    刚体运动在实验室坐标系中定义。

    该运动在时间区间 time=[0,1] 内是参数化的，它通过将点x从[0, rotation_angle]进行线性均匀旋转，
    并从[0, translation]进行平移来执行刚体变换。也就是说，如果在时间 t 调用，运动将首先围绕 `rotation_axis` 
    旋转点 `t*rotation_angle` 弧度，然后通过向量 `t*translation` 平移该点。

    参数：
        rotation_axis (:obj:`numpy array`): 旋转轴，形状为 ``shape=(3,)``
        rotation_angle (:obj:`float`): 最终旋转的弧度值，当 time=1 时。
        translation (:obj:`numpy array`): 平移向量，形状为 ``shape=(3,)``
        origin (:obj:`numpy array`): 定义刚体运动的空间点，默认为原点 (0,0,0)。
            所有平移都相对于此原点执行，所有旋转也都是围绕此原点的旋转。形状为 ``shape=(3,)``

    属性：
        rotation_axis (:obj:`numpy array`): 旋转轴，形状为 ``shape=(3,)``
        rotation_angle (:obj:`float`): 最终旋转的弧度值，当 time=1 时。
        translation (:obj:`numpy array`): 平移向量，形状为 ``shape=(3,)``
        origin (:obj:`numpy array`): 定义刚体运动的空间点，默认为原点 (0,0,0)。
            所有平移都相对于此原点执行，所有旋转也都是围绕此原点的旋转。形状为 ``shape=(3,)``
    """

    def __init__(
        self, rotation_axis, rotation_angle, translation, origin=torch.zeros((3,))
    ):
        assert (
            rotation_angle < torch.pi and rotation_angle > 0
        ), "The rotation angle must be in [0 pi]" # 旋转角度必须在 [0, pi] 范围内
        self.rotator = _RodriguezRotator(rotation_axis)
        self.rotation_axis = ensure_torch(rotation_axis)
        self.rotation_angle = ensure_torch(rotation_angle)
        self.translation = ensure_torch(translation)
        self.origin = ensure_torch(origin)

    def __call__(self, vectors, time):
        """Find the transformation of a set of points at a prescribed time.

        Calling this method executes the rigid body motion with respect to the
        currently set origin.

        Parameters
        ----------
        vectors : numpy.ndarray or torch.Tensor
            A set of points to be transformed, shape ``(3, N)``, ``(N, 3)``, or
            ``(N, 4, 3)``.
        time : float, numpy.ndarray, or torch.Tensor
            Time to compute for. Can be scalar or shape ``(N,)`` for per-vector
            times.

        Returns
        -------
        torch.Tensor
            Transformed vectors with shape ``(3, N)``, ``(N, 3)``, or
            ``(N, 4, 3)``.
        """
        # assert time <= 1 and time >= 0, "The rigid body motion is only valid on the interval time=[0,1]" # 刚体运动仅在时间区间 time=[0,1] 内有效
        vectors = ensure_torch(vectors)
        time = ensure_torch(time)

        if len(vectors.shape) == 1:
            translation = self.translation
            origin = self.origin
            centered_vectors = vectors - origin
            centered_rotated_vectors = self.rotator(
                centered_vectors, self.rotation_angle * time
            )
            rotated_vectors = centered_rotated_vectors + origin
            return torch.squeeze(rotated_vectors + translation * time)

        elif len(vectors.shape) == 2:
            translation = self.translation.reshape(1, 3)
            origin = self.origin.reshape(1, 3) 
            centered_vectors = vectors - origin
            centered_rotated_vectors = self.rotator(
                centered_vectors, self.rotation_angle * time
            )
            rotated_vectors = centered_rotated_vectors + origin
            if time.ndim == 0 or (time.ndim == 1 and len(time) == 1):
                return rotated_vectors + translation * time

            return rotated_vectors + translation * time.unsqueeze(-1)

        elif len(vectors.shape) == 3:
            # Handle (N, M, 3) input shapes - typically (N_peaks, 4_vertices, 3_coords)
            N, M, _ = vectors.shape
            translation = self.translation.reshape(1, 3)
            origin = self.origin.reshape(1, 3)
            centered_vectors = vectors - origin
            centered_rotated_vectors = self.rotator(
                centered_vectors.reshape(-1, 3),
                self.rotation_angle * torch.tile(time, (M, 1)).T.reshape(-1),
            ).reshape(N, M, 3)
            rotated_vectors = centered_rotated_vectors + origin
            # Don't squeeze - preserve shape for proper indexing downstream
            return (
                rotated_vectors
                + translation * ensure_torch(time)[:, torch.newaxis, torch.newaxis]
            )

    def rotate(self, vectors, time):
        """Find the rotational transformation of a set of vectors.

        This function only applies the rigid body rotation and will not respect
        the origin of the motion. This function is intended for rotation of
        diffraction and wavevectors. Use the ``__call__`` method to perform a
        physical rigid body motion respecting the origin.

        Parameters
        ----------
        vectors : numpy.ndarray or torch.Tensor
            A set of points in 3D Euclidean space to be rotated, shape
            ``(3, N)`` or ``(N, 3)``.
        time : float, numpy.ndarray, or torch.Tensor
            Time to compute for. Can be scalar or shape ``(N,)`` for per-vector
            times.

        Returns
        -------
        torch.Tensor
            Transformed vectors of shape ``(3, N)`` or ``(N, 3)``.
                    在指定时间找到一组向量的旋转变换。

        注意：此函数仅应用刚体旋转，而不会考虑运动的原点！此函数旨在用于衍射向量
        和波向量的旋转。要执行考虑原点的物理刚体运动，请使用 __call__ 方法。

        参数：
            vectors (:obj:`numpy array`): 要旋转的3D欧几里得空间中的点集，形状为 (``shape=(3,N)``)
            time (:obj:`float`): 用于计算的时间点。

        返回：
            变换后的向量 (:obj:`numpy array`)，形状为 ``shape=(3,N)``。
        """
        # assert time <= 1 and time >= 0, "The rigid body motion is only valid on the interval time=[0,1]"
        time = ensure_torch(time)
        rotated_vectors = self.rotator(vectors, self.rotation_angle * time)
        return rotated_vectors

    def translate(self, vectors, time):
        """Find the translational transformation of a set of points.

        This function only applies the rigid body translation.

        Parameters
        ----------
        vectors : numpy.ndarray or torch.Tensor
            A set of points in 3D Euclidean space to be translated, shape
            ``(3, N)`` or ``(N, 3)``.
        time : float
            Time to compute for.

        Returns
        -------
        torch.Tensor
            Transformed vectors of shape ``(3, N)`` or ``(N, 3)``.
        在指定时间找到一组点的平移变换。

        注意：此函数仅应用刚体平移。

        参数：
            vectors (:obj:`numpy array`): 要旋转的3D欧几里得空间中的点集，形状为 (``shape=(3,N)``)
            time (:obj:`float`): 用于计算的时间点。

        返回：
            变换后的向量 (:obj:`numpy array`)，形状为 ``shape=(3,N)``。
        """
        assert (
            time <= 1 and time >= 0
        ), "The rigid body motion is only valid on the interval time=[0,1]"
        
        vectors = ensure_torch(vectors)
        
        if len(vectors.shape) > 1:
            translation = self.translation.reshape(3, 1)
        else:
            translation = self.translation

        return vectors + translation * time

    def inverse(self):
        """Create an instance of the inverse motion.

        The inverse is defined by negative translation and rotation axis
        vectors.

        Returns
        -------
        RigidBodyMotion
            The inverse motion with a reversed rotation and translation.
        创建一个逆运动的实例，由负的平移向量和负的旋转轴向量定义。

        返回：
            (:obj:`xrd_simulator.RigidBodyMotion`) 具有相反旋转和平移的逆运动对象。
        """
        return RigidBodyMotion(
            -self.rotation_axis.clone(),
            self.rotation_angle,
            -self.translation.clone(),
            self.origin.clone(),
        )

    def save(self, path):
        """Save the motion object to disc via pickling.

        Parameters
        ----------
        path : str
            File path at which to save, ending with the desired filename.
            The ``.motion`` extension is added if not present.
        """
        if not path.endswith(".motion"):
            path = path + ".motion"
        with open(path, "wb") as f:
            dill.dump(self, f, dill.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        """Load the motion object from disc via pickling.

        Parameters
        ----------
        path : str
            File path at which to load, ending with the desired filename.

        Returns
        -------
        RigidBodyMotion
            Loaded motion object.

        Raises
        ------
        ValueError
            If the file does not end with ``.motion``.

        Warnings
        --------
        This function will unpickle data from the provided path. The pickle
        module is not intended to be secure against erroneous or maliciously
        constructed data. Never unpickle data received from an untrusted or
        unauthenticated source.
                    从磁盘加载运动对象（通过 pickling）。

        参数：
            path (:obj:`str`): 用于加载的文件路径，以所需的文件名结尾。

        .. 警告::
            此函数将从提供的路径中反序列化（unpickle）数据。pickle 模块
            不能保证免受错误或恶意构造的数据的攻击。
            永远不要反序列化从不受信任或未经身份验证的来源接收的数据。
        """
        if not path.endswith(".motion"):
            raise ValueError("The loaded motion file must end with .motion")
        with open(path, "rb") as f:
            return dill.load(f)


class _RodriguezRotator(object):
    """Object for rotating vectors around a unit normal rotation axis.

    Parameters
    ----------
    rotation_axis : numpy.ndarray or torch.Tensor
        A unit vector in 3D Euclidean space, shape ``(3,)``.

    Attributes
    ----------
    rotation_axis : torch.Tensor
        A unit vector in 3D Euclidean space, shape ``(3,)``.
    K : torch.Tensor
        Skew-symmetric cross-product matrix, shape ``(3, 3)``.
    K2 : torch.Tensor
        Square of the skew-symmetric matrix, shape ``(3, 3)``.        
    用于在由单位法线 `rotation_axis` 描述的平面中旋转向量的对象。

    参数：
        rotation_axis (:obj:`numpy array`): 3D欧几里得空间中的单位向量，形状为 (``shape=(3,)``)

    属性：
        rotation_axis (:obj:`numpy array`): 3D欧几里得空间中的单位向量，形状为 (``shape=(3,)``)
        K (:obj:`numpy array`): 形状为 (``shape=(3,3)``)
        K2 (:obj:`numpy array`): 形状为 (``shape=(3,3)``)
        I (:obj:`numpy array`): 形状为 (``shape=(3,3)``)
    """

    def __init__(self, rotation_axis):
        rotation_axis = ensure_torch(rotation_axis)
        assert torch.allclose(
            torch.linalg.norm(rotation_axis), ensure_torch(1.0)
        ), "The rotation axis must be length unity."
        self.rotation_axis = rotation_axis
        rx, ry, rz = self.rotation_axis
        self.K = ensure_torch([[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]])
        self.K2 = torch.matmul(self.K, self.K)

    # def get_rotation_matrix(self, rotation_angle):
    #     """Get the rotation matrix for a given rotation angle."""
    #     identity_matrix = torch.eye(3, dtype=self.K.dtype).unsqueeze(2)
    #     sin_term = torch.sin(rotation_angle) * self.K.unsqueeze(2)
    #     cos_term = (1 - torch.cos(rotation_angle)) * self.K2.unsqueeze(2)

    #     rotation_matrix = identity_matrix + sin_term + cos_term
    #     rotation_matrix = rotation_matrix.permute(2, 0, 1)

    #     return rotation_matrix
    
    def get_rotation_matrix(self, rotation_angle):
        """Compute 3x3 rotation matrices for one or many rotation angles.

        Parameters
        ----------
        rotation_angle : torch.Tensor
            Scalar tensor or shape ``(N,)``.

        Returns
        -------
        torch.Tensor
            If scalar, returns shape ``(3, 3)``. If N angles, returns shape
            ``(N, 3, 3)``.
        """
        # Ensure rotation_angle is (..., 1, 1) for broadcasting
        # rotation_angle: scalar → shape (1,1,1)
        #                 vector (N,) → shape (N,1,1)
        rot = rotation_angle[..., None, None]

        # K and K2 are 3×3 tensors — add batch dims for broadcasting
        K  = self.K[None, :, :]      # (1,3,3)
        K2 = self.K2[None, :, :]     # (1,3,3)

        # Identity matrix, broadcastable
        I = torch.eye(3, dtype=self.K.dtype, device=self.K.device)[None, :, :]  # (1,3,3)

        # Rodrigues’ formula
        sin_term = torch.sin(rot) * K
        cos_term = (1 - torch.cos(rot)) * K2

        R = I + sin_term + cos_term  # shape: (N,3,3) or (1,3,3)

        # Return correct shape:
        # if input was scalar → return (3,3)
        if rotation_angle.ndim == 0:
            return R[0]
        else:
            return R


    def __call__(self, vectors, rotation_angle):
        """Rotate vectors around the rotation axis.

        Parameters
        ----------
        vectors : numpy.ndarray or torch.Tensor
            A set of vectors in 3D Euclidean space to be rotated, shape
            ``(N, 3)`` or ``(3,)`` for single vector.
        rotation_angle : float, numpy.ndarray, or torch.Tensor
            Radians to rotate vectors around the rotation axis (positive
            rotation). Can be scalar or shape ``(N,)`` for per-vector angles.

        Returns
        -------
        torch.Tensor
            Rotated vectors of shape ``(N, 3)`` or ``(3,)`` for single vector.
        """

        R = self.get_rotation_matrix(rotation_angle)
        vectors = ensure_torch(vectors)

        if len(vectors.shape) == 1:
            vectors = vectors[None, :]
        
        # Handle both scalar and vector rotation_angle cases
        if rotation_angle.ndim == 0:
            # Scalar case: R is (3,3), vectors is (N,3) -> output (N,3)
            return torch.squeeze(torch.matmul(R, vectors[:, :, None]))
        else:
            # Vector case: R is (N,3,3), vectors is (N,3) -> output (N,3)
            # Use bmm: (N,3,3) × (N,3,1) -> (N,3,1) -> squeeze to (N,3)
            return torch.bmm(R, vectors.unsqueeze(-1)).squeeze(-1)
