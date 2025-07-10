#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X射线衍射实验几何参数分析脚本

本脚本分析nickel.ipynb中定义的物理建模参数，
并输出关键的几何关系以便在Blender中验证。
"""

import numpy as np
import math

def analyze_geometry():
    """
    分析X射线衍射实验的几何参数
    """
    print("="*60)
    print("X射线衍射实验几何参数分析")
    print("="*60)
    
    # 从代码中提取的原始参数
    print("\n1. 原始参数 (从nickel.ipynb代码中提取):")
    print("-"*40)
    
    # 光束参数
    beam_size_um = 200  # 光束截面 200μm x 200μm
    wavelength_angstrom = 0.1258  # X射线波长
    propagation_direction = np.array([1., 0., 0.])  # 传播方向
    
    print(f"光束截面尺寸: {beam_size_um}μm × {beam_size_um}μm")
    print(f"X射线波长: {wavelength_angstrom}Å")
    print(f"传播方向: {propagation_direction}")
    
    # 探测器参数
    detector_distance_mm = 1324.9  # 探测器距离
    pixel_size_um = 139.0  # 像素尺寸
    number_of_pixels = 3072  # 像素数量
    detector_size_mm = number_of_pixels * pixel_size_um / 1000  # 物理尺寸
    
    print(f"\n探测器距离: {detector_distance_mm}mm")
    print(f"像素尺寸: {pixel_size_um}μm")
    print(f"像素矩阵: {number_of_pixels} × {number_of_pixels}")
    print(f"探测器物理尺寸: {detector_size_mm:.1f}mm × {detector_size_mm:.1f}mm")
    
    # 样品参数
    sample_radius_um = 1000  # 样品半径
    sample_diameter_mm = 2 * sample_radius_um / 1000  # 样品直径
    
    print(f"\n样品半径: {sample_radius_um}μm")
    print(f"样品直径: {sample_diameter_mm}mm")
    print(f"样品材料: 镍 (Nickel) 单晶")
    
    # 2. 单位转换和Blender缩放
    print("\n\n2. Blender可视化中的缩放参数:")
    print("-"*40)
    
    scale_factor = 1000  # 1 Blender单位 = 1000μm = 1mm
    
    # Blender中的尺寸
    blender_sample_radius = sample_radius_um / scale_factor
    blender_detector_distance = detector_distance_mm
    blender_detector_size = detector_size_mm
    blender_beam_size = beam_size_um / scale_factor
    
    print(f"缩放比例: 1 Blender单位 = {scale_factor}μm = 1mm")
    print(f"样品半径: {blender_sample_radius}mm")
    print(f"探测器距离: {blender_detector_distance}mm")
    print(f"探测器尺寸: {blender_detector_size:.1f}mm × {blender_detector_size:.1f}mm")
    print(f"光束截面: {blender_beam_size}mm × {blender_beam_size}mm")
    
    # 3. 关键几何关系分析
    print("\n\n3. 关键几何关系分析:")
    print("-"*40)
    
    # 距离比例
    distance_to_sample_ratio = detector_distance_mm / sample_diameter_mm
    distance_to_detector_ratio = detector_distance_mm / detector_size_mm
    detector_to_sample_ratio = detector_size_mm / sample_diameter_mm
    
    print(f"探测器距离 / 样品直径 = {distance_to_sample_ratio:.1f}")
    print(f"探测器距离 / 探测器尺寸 = {distance_to_detector_ratio:.1f}")
    print(f"探测器尺寸 / 样品直径 = {detector_to_sample_ratio:.1f}")
    
    # 角度分析
    max_scattering_angle_rad = math.atan(detector_size_mm / 2 / detector_distance_mm)
    max_scattering_angle_deg = math.degrees(max_scattering_angle_rad)
    
    print(f"\n最大散射角: {max_scattering_angle_deg:.1f}°")
    print(f"最大散射角 (弧度): {max_scattering_angle_rad:.3f} rad")
    
    # 4. 坐标系统定义
    print("\n\n4. 坐标系统定义:")
    print("-"*40)
    
    print("原点位置: 样品中心 (0, 0, 0)")
    print("X轴: 光束传播方向 (样品 → 探测器)")
    print("Y轴: 探测器水平方向")
    print("Z轴: 探测器垂直方向")
    
    # 探测器角点坐标
    half_detector = detector_size_mm / 2
    detector_corners = {
        "corner_0": (detector_distance_mm, -half_detector, -half_detector),
        "corner_1": (detector_distance_mm, half_detector, -half_detector),
        "corner_2": (detector_distance_mm, -half_detector, half_detector),
        "corner_3": (detector_distance_mm, half_detector, half_detector)
    }
    
    print(f"\n探测器角点坐标 (mm):")
    for corner_name, coords in detector_corners.items():
        print(f"  {corner_name}: ({coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f})")
    
    # 5. 验证要点
    print("\n\n5. Blender中的验证要点:")
    print("-"*40)
    
    checks = [
        "✓ 样品球体位于原点 (0, 0, 0)",
        f"✓ 探测器平面位于 X = {detector_distance_mm}mm",
        "✓ 光束沿X轴正方向传播",
        "✓ 探测器垂直于X轴",
        f"✓ 样品直径 ({sample_diameter_mm}mm) << 探测器尺寸 ({detector_size_mm:.1f}mm)",
        f"✓ 探测器距离 ({detector_distance_mm}mm) >> 样品尺寸",
        f"✓ 光束截面 ({beam_size_um}μm) 适中"
    ]
    
    for check in checks:
        print(f"  {check}")
    
    # 6. 物理合理性评估
    print("\n\n6. 物理合理性评估:")
    print("-"*40)
    
    # 评估各种比例是否合理
    evaluations = []
    
    if distance_to_sample_ratio > 500:
        evaluations.append("✓ 探测器距离相对样品尺寸合理 (远场条件)")
    else:
        evaluations.append("⚠ 探测器距离可能过近")
    
    if 2 < distance_to_detector_ratio < 5:
        evaluations.append("✓ 探测器距离与尺寸比例合理")
    else:
        evaluations.append("⚠ 探测器距离与尺寸比例需要检查")
    
    if detector_to_sample_ratio > 100:
        evaluations.append("✓ 探测器尺寸相对样品足够大")
    else:
        evaluations.append("⚠ 探测器可能过小")
    
    if 10 < max_scattering_angle_deg < 30:
        evaluations.append("✓ 最大散射角范围合理")
    else:
        evaluations.append(f"⚠ 最大散射角 ({max_scattering_angle_deg:.1f}°) 需要检查")
    
    for evaluation in evaluations:
        print(f"  {evaluation}")
    
    # 7. Blender脚本使用说明
    print("\n\n7. 使用Blender验证:")
    print("-"*40)
    print("1. 运行 run_blender_visualization.bat")
    print("2. 或在Blender中执行 xrd_visualization.py")
    print("3. 检查上述验证要点")
    print("4. 调整视角观察空间关系")
    print("5. 验证物理参数的合理性")
    
    print("\n" + "="*60)
    print("分析完成! 请使用Blender脚本进行可视化验证。")
    print("="*60)

if __name__ == "__main__":
    analyze_geometry()