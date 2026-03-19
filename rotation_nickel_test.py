import numpy as np
import os
from xrd_simulator.beam import Beam
from xrd_simulator.detector import Detector
from xrd_simulator.mesh import TetraMesh
from xrd_simulator.phase import Phase
from xrd_simulator.polycrystal import Polycrystal
from xrd_simulator.motion import RigidBodyMotion
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib.colors import PowerNorm

# --- 创建用于保存结果的文件夹 ---
output_folder = 'tilt_angle_study'
os.makedirs(output_folder, exist_ok=True)
print(f"结果将保存到文件夹: '{output_folder}/'")

# --- 定义光束和探测器 (在循环外定义，因为它们不变) ---
beam_edge = 2500.
beam = Beam(
    np.array([
        [-5e6, -beam_edge, -beam_edge], [-5e6, beam_edge, -beam_edge], [-5e6, beam_edge, beam_edge], [-5e6, -beam_edge, beam_edge],
        [5e6, -beam_edge, -beam_edge], [5e6, beam_edge, -beam_edge], [5e6, beam_edge, beam_edge], [5e6, -beam_edge, beam_edge]
    ]),
    xray_propagation_direction=np.array([1., 0., 0.]),
    wavelength=0.1258,
    polarization_vector=np.array([0., 1., 0.])
)
detector_distance_mm = 1324.9
pixel_size_um = 139.0
number_of_pixels = 3072
detector_distance_um = detector_distance_mm * 1000
detector_size_um = number_of_pixels * pixel_size_um
half_detector_size = detector_size_um / 2.0
detector = Detector(
    pixel_size_z=pixel_size_um,
    pixel_size_y=pixel_size_um,
    det_corner_0=np.array([detector_distance_um, -half_detector_size, -half_detector_size]),
    det_corner_1=np.array([detector_distance_um,  half_detector_size, -half_detector_size]),
    det_corner_2=np.array([detector_distance_um, -half_detector_size,  half_detector_size])
)

# --- 定义几何形状和晶相 (在循环外定义) ---
new_radius = 2000.0
mesh = TetraMesh.generate_mesh_from_levelset(
    level_set=lambda x: np.linalg.norm(x) - new_radius,
    bounding_radius=new_radius + 1.0,
    max_cell_circumradius=150.
)
try:
    nickel_phase = Phase(
        unit_cell=[3.52, 3.52, 3.52, 90., 90., 90.],
        sgname='Fm-3m',
        path_to_cif_file='Ni.cif'
    )
except FileNotFoundError:
    print("错误: 未找到 'Ni.cif' 文件。")
    exit()

# --- 主循环：遍历所有倾斜角度 ---
tilt_angles_degrees = [-5, -3, 0, 3, 5]
for tilt_angle in tilt_angles_degrees:
    print(f"\n--- 开始模拟，倾斜角度: {tilt_angle}° ---")

    # --- 3. 定义特定取向的近单晶 ---
    # A. 计算初始对准旋转 ([110] || beam)
    crystal_direction_to_align = np.array([1, 1, 0])
    beam_direction_in_lab = np.array([1, 0, 0])
    initial_base_rotation, _ = R.align_vectors([beam_direction_in_lab], [crystal_direction_to_align])
    
    # B. 创建绕Y轴的倾斜旋转
    tilt_axis = np.array([0, 0, 1]) # 实验室竖直方向
    tilt_rotation = R.from_rotvec(np.radians(tilt_angle) * tilt_axis)
    
    # C. 复合得到最终的基准取向
    final_base_rotation = tilt_rotation * initial_base_rotation
    print(f"基准取向已设定为: [110]对准光束后，再绕Z轴旋转 {tilt_angle}°")

    # D. 在最终基准取向上添加正态分布的取向扩展
    sigma_degrees = 2
    sigma_radians = np.radians(sigma_degrees)
    orientations_with_spread = []
    for _ in range(mesh.number_of_elements):
        random_axis = np.random.randn(3)
        if np.linalg.norm(random_axis) > 1e-8:
            random_axis /= np.linalg.norm(random_axis)
        random_angle = abs(np.random.randn()) * sigma_radians
        small_random_rotation = R.from_rotvec(random_angle * random_axis)
        final_rotation = small_random_rotation * final_base_rotation
        orientations_with_spread.append(final_rotation.as_matrix())
    single_crystal_orientation = np.array(orientations_with_spread)

    # --- 4. 组装成单晶样品 ---
    polycrystal = Polycrystal(
        mesh=mesh,
        orientation=single_crystal_orientation,
        strain=np.zeros((3, 3)),
        phases=[nickel_phase],
        element_phase_map=None
    )

    # --- 5. 执行模拟和渲染 ---
    # 每次循环都必须重置探测器中的旧数据
    detector.frames = [] 
    motion = RigidBodyMotion(
        rotation_axis=np.array([0, 1, 0]),
        rotation_angle=np.radians(0.5),
        translation=np.array([0., 0., 0.])
    )
    polycrystal.diffract(beam, detector, motion)
    detector.point_spread_kernel_shape = (7, 7)
    diffraction_pattern = detector.render(
        frames_to_render='all',
        lorentz=True,
        polarization=True,
        structure_factor=True,
        method="project"
    )

    # --- 6. 显示与保存结果 (增加对inf值的处理) ---
    fig, ax = plt.subplots(1, 1, figsize=(16, 15))

    # 1. 筛选出所有大于0且为有限数的强度值（即排除0, inf, nan）
    finite_intensities = diffraction_pattern[(diffraction_pattern > 0) & (np.isfinite(diffraction_pattern))]

    # 2. 检查是否存在有效的有限信号
    if finite_intensities.size > 0:
        vmin_val = 0
        vmax_percentile = 99.9
        vmax_val = np.percentile(finite_intensities, vmax_percentile)
    else:
        # 处理没有有效信号或只有inf信号的情况
        vmin_val = 0
        vmax_val = 1
        print(f"警告：在倾角 {tilt_angle}° 时未找到任何有效的有限强度信号。")

    print(f"原始数据最大值: {diffraction_pattern.max():.2e}")
    print(f"为增强对比度，仅对有限信号计算，颜色上限(vmax)设为: {vmax_val:.2e}")

    # 3. 渲染图像
    gamma = 0.3
    im = ax.imshow(
        diffraction_pattern,
        cmap='gnuplot2',
        interpolation='nearest',
        norm=PowerNorm(gamma=gamma, vmin=vmin_val, vmax=vmax_val, clip=True)
    )

    # 4. 添加颜色条
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Diffraction Pattern (Tilt Angle = {tilt_angle} degrees)")

    # 保存图像到文件夹
    output_filename = f"Z-axis_tilt_{tilt_angle}_deg.png"
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"图像已保存到: {output_path}")
    plt.close(fig) # 关闭图形窗口，防止在循环中全部显示出来

print("\n--- 所有模拟已完成！---")