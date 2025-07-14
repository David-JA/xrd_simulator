"""The motion module is used to represent a rigid body motion. During diffraction from a
:class:`xrd_simulator.polycrystal.Polycrystal` the :class:`xrd_simulator.motion.RigidBodyMotion`
object describes how the sample is translating and rotating. The motion can be used to update the
polycrystal position via the :func:`xrd_simulator.polycrystal.Polycrystal.transform` function.

Here is a minimal example of how to instantiate a rigid body motion object, apply the motion to a pointcloud
and save the motion to disc:

    Examples:
        .. literalinclude:: examples/example_init_motion.py

Below follows a detailed description of the RigidBodyMotion class attributes and functions.

运动模块用于表示刚体运动。在 :class:`xrd_simulator.polycrystal.Polycrystal` 的衍射过程中，
:class:`xrd_simulator.motion.RigidBodyMotion` 对象描述了样品如何平移和旋转。该运动可通过
:func:`xrd_simulator.polycrystal.Polycrystal.transform` 函数来更新多晶体的位置。

以下是如何实例化一个刚体运动对象、将运动应用于点云并将运动保存到磁盘的最小示例：

    示例：
        .. literalinclude:: examples/example_init_motion.py

下面是 RigidBodyMotion 类属性和函数的详细描述。
"""
import numpy as np
import dill

class RigidBodyMotion():
    """Rigid body transformation of euclidean points by an euler axis rotation and a translation.

    A rigid body motion is defined in the laboratory coordinates system.

    The Motion is parametric in the interval time=[0,1] and will perform a rigid body transformation
    of a point x by linearly uniformly rotating it from [0, rotation_angle] and translating [0, translation].
    I.e if called at a time time=t the motion will first rotate the point ``t*rotation_angle`` radians
    around ``rotation_axis`` and next translate the point by the vector ``t*translation``.

    Args:
        rotation_axis (:obj:`numpy array`): Rotation axis ``shape=(3,)``
        rotation_angle (:obj:`float`): Radians for final rotation, when time=1.
        translation (:obj:`numpy array`):  Translation vector ``shape=(3,)``
        origin (:obj:`numpy array`): Point in space about which the rigid body motion is
            defined Defaults to the origin (0,0,0). All translations are executed in relation
            to the origin and all rotation are rotations about the point of origin. ``shape=(3,)``

    Attributes:
        rotation_axis (:obj:`numpy array`): Rotation axis ``shape=(3,)``
        rotation_angle (:obj:`float`): Radians for final rotation, when time=1.
        translation (:obj:`numpy array`):  Translation vector ``shape=(3,)``
        origin (:obj:`numpy array`): Point in space about which the rigid body motion is
            defined Defaults to the origin (0,0,0). All translations are executed in relation
            to the origin and all rotation are rotations about the point of origin. ``shape=(3,)``
            
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

    def __init__(self, rotation_axis, rotation_angle, translation, origin=np.zeros((3,))):
        assert rotation_angle < np.pi and rotation_angle > 0, "The rotation angle must be in [0 pi]" # 旋转角度必须在 [0, pi] 范围内
        self.rotator = _RodriguezRotator(rotation_axis)
        self.rotation_axis = rotation_axis
        self.rotation_angle = rotation_angle
        self.translation = translation
        self.origin = origin

    def __call__(self, vectors, time):
        """Find the transformation of a set of points at a prescribed time.

        Calling this method will execute the rigid body motion with respect to the
            currently set origin.

        Args:
            vectors (:obj:`numpy array`): A set of points to be transformed (``shape=(3,N)``)
            time (:obj:`float`): Time to compute for.

        Returns:
            Transformed vectors (:obj:`numpy array`) of ``shape=(3,N)``.
            
        在指定时间找到一组点的变换。

        调用此方法将相对于当前设置的原点执行刚体运动。

        参数：
            vectors (:obj:`numpy array`): 要变换的点集，形状为 (``shape=(3,N)``)
            time (:obj:`float`): 用于计算的时间点。

        返回：
            变换后的向量 (:obj:`numpy array`)，形状为 ``shape=(3,N)``。
        """
        #assert time <= 1 and time >= 0, "The rigid body motion is only valid on the interval time=[0,1]" # 刚体运动仅在时间区间 time=[0,1] 内有效
        
        if len(vectors.shape) == 1:
            translation = self.translation
            origin = self.origin
            centered_vectors = vectors - origin
            centered_rotated_vectors  =  self.rotator(centered_vectors, self.rotation_angle * time)
            rotated_vectors = centered_rotated_vectors + origin
            return np.squeeze(rotated_vectors + translation * time)
        
        elif len(vectors.shape) == 2:
            translation = self.translation.reshape(1,3)
            origin = self.origin.reshape(1,3)       
            centered_vectors = vectors - origin
            centered_rotated_vectors  =  self.rotator(centered_vectors, self.rotation_angle * time)
            rotated_vectors = centered_rotated_vectors + origin
            if np.isscalar(time):
                return rotated_vectors + translation * time
            return np.squeeze(rotated_vectors + translation * np.array(time)[:,np.newaxis])
        
        elif len(vectors.shape) == 3:
            translation = self.translation.reshape(1,3)
            origin = self.origin.reshape(1,3)
            centered_vectors = vectors - origin
            centered_rotated_vectors  =  self.rotator(centered_vectors.reshape(-1,3), self.rotation_angle * np.tile(time,(4,1)).T.reshape(-1)).reshape(-1,4,3)
            rotated_vectors = centered_rotated_vectors + origin      
            return np.squeeze(rotated_vectors + translation * np.array(time)[:,np.newaxis,np.newaxis])
    
    def rotate(self, vectors, time):
        """Find the rotational transformation of a set of vectors at a prescribed time.

        NOTE: This function only applies the rigid body rotation and will not respect the
            origin of the motion! This function is intended for rotation of diffraction
            and wavevectors. Use the __call__ method to perform a physical rigidbody motion
            respecting the origin.

        Args:
            vectors (:obj:`numpy array`): A set of points in 3d euclidean space to be rotated (``shape=(3,N)``)
            time (:obj:`float`): Time to compute for.

        Returns:
            Transformed vectors (:obj:`numpy array`) of ``shape=(3,N)``.
            
        在指定时间找到一组向量的旋转变换。

        注意：此函数仅应用刚体旋转，而不会考虑运动的原点！此函数旨在用于衍射向量
        和波向量的旋转。要执行考虑原点的物理刚体运动，请使用 __call__ 方法。

        参数：
            vectors (:obj:`numpy array`): 要旋转的3D欧几里得空间中的点集，形状为 (``shape=(3,N)``)
            time (:obj:`float`): 用于计算的时间点。

        返回：
            变换后的向量 (:obj:`numpy array`)，形状为 ``shape=(3,N)``。
        """
        #assert time <= 1 and time >= 0, "The rigid body motion is only valid on the interval time=[0,1]"
        rotated_vectors  = self.rotator(vectors, self.rotation_angle * time)
        return rotated_vectors

    def translate(self, vectors, time):
        """Find the translational transformation of a set of points at a prescribed time.

        NOTE: This function only applies the rigid body translation.

        Args:
            vectors (:obj:`numpy array`): A set of points in 3d euclidean space to be rotated (``shape=(3,N)``)
            time (:obj:`float`): Time to compute for.

        Returns:
            Transformed vectors (:obj:`numpy array`) of ``shape=(3,N)``.

        在指定时间找到一组点的平移变换。

        注意：此函数仅应用刚体平移。

        参数：
            vectors (:obj:`numpy array`): 要旋转的3D欧几里得空间中的点集，形状为 (``shape=(3,N)``)
            time (:obj:`float`): 用于计算的时间点。

        返回：
            变换后的向量 (:obj:`numpy array`)，形状为 ``shape=(3,N)``。
        """
        assert time <= 1 and time >= 0, "The rigid body motion is only valid on the interval time=[0,1]"
        if len(vectors.shape) > 1:
            translation = self.translation.reshape(3, 1)
        else:
            translation = self.translation

        return vectors + translation * time

    def inverse(self):
        """Create an instance of the inverse motion, defined by negative translation- and rotation-axis vectors.

        Returns:
            (:obj:`xrd_simulator.RigidBodyMotion`) The inverse motion with a reversed rotation and translation.

        创建一个逆运动的实例，由负的平移向量和负的旋转轴向量定义。

        返回：
            (:obj:`xrd_simulator.RigidBodyMotion`) 具有相反旋转和平移的逆运动对象。
        """
        return RigidBodyMotion(-self.rotation_axis.copy(), self.rotation_angle, -self.translation.copy(), self.origin.copy())

    def save(self, path):
        """Save the motion object to disc (via pickling).

        Args:
            path (:obj:`str`): File path at which to save, ending with the desired filename.
            
        将运动对象保存到磁盘（通过 pickling）。

        参数：
            path (:obj:`str`): 用于保存的文件路径，以所需的文件名结尾。
        """
        if not path.endswith(".motion"):
            path = path + ".motion"
        with open(path, "wb") as f:
            dill.dump(self, f, dill.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        """Load the motion object from disc (via pickling).

        Args:
            path (:obj:`str`): File path at which to load, ending with the desired filename.

        .. warning::
            This function will unpickle data from the provied path. The pickle module
            is not intended to be secure against erroneous or maliciously constructed data.
            Never unpickle data received from an untrusted or unauthenticated source.
            
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
        with open(path, 'rb') as f:
            return dill.load(f)

class _RodriguezRotator(object):
    """Object for rotating vectors in the plane described by yhe unit normal rotation_axis.

    Args:
        rotation_axis (:obj:`numpy array`): A unit vector in 3d euclidean space (``shape=(3,)``)

    Attributes:
        rotation_axis (:obj:`numpy array`): A unit vector in 3d euclidean space (``shape=(3,)``)
        K (:obj:`numpy array`): (``shape=(3,3)``)
        K2 (:obj:`numpy array`): (``shape=(3,3)``)
        I (:obj:`numpy array`): (``shape=(3,3)``)
        
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
        assert np.allclose(np.linalg.norm(rotation_axis),
                           1), "The rotation axis must be length unity."
        self.rotation_axis = rotation_axis
        rx, ry, rz = self.rotation_axis
        self.K = np.array([[0, -rz, ry],
                           [rz, 0, -rx],
                           [-ry, rx, 0]])
        self.K2 = self.K.dot(self.K)

    def get_rotation_matrix(self, rotation_angle):
        return np.squeeze((np.eye(3, 3)[:,:,np.newaxis] + np.sin(rotation_angle) * self.K[:,:,np.newaxis] + (1 - np.cos(rotation_angle)) * self.K2[:,:,np.newaxis]).transpose(2,0,1))

    def __call__(self, vectors, rotation_angle):
        """Rotate a vector in the plane described by v1 and v2 towards v2 a fraction s=[0,1].

        Args:
            vectors (:obj:`numpy array`): A set of vectors in 3d euclidean space to be rotated (``shape=(3,N)``)
            rotation_angle (:obj:`float`): Radians to rotate vectors around the rotation_axis (positive rotation).

        Returns:
            Rotated vectors (:obj:`numpy array`) of ``shape=(3,N)``.
            
        在由v1和v2描述的平面中，将一个向量向v2旋转一个分数s=[0,1]。 (注释原文似有误，应为围绕旋转轴旋转)
        (修正后含义) 围绕 `rotation_axis` 旋转一组向量。

        参数：
            vectors (:obj:`numpy array`): 要旋转的3D欧几里得空间中的向量集，形状为 (``shape=(3,N)``)
            rotation_angle (:obj:`float`): 围绕 `rotation_axis` 旋转向量的弧度值（正向旋转）。

        返回：
            旋转后的向量 (:obj:`numpy array`)，形状为 ``shape=(3,N)``。
        """

        R = self.get_rotation_matrix(rotation_angle)
        
        if len(vectors.shape)==1:
            vectors = vectors[np.newaxis,:]
        
        return np.matmul(R,vectors[:,:,np.newaxis])[:,:,0] # Syntax valid for the rotation fo the G vectors from the grains
                                                           # 此语法对于旋转晶粒的G向量有效