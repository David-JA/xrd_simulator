import bpy
import bmesh
from mathutils import Vector
import math

# 清除默认场景
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# 删除默认材质
for material in bpy.data.materials:
    bpy.data.materials.remove(material)

# 创建材质函数（兼容版本）
def create_material(name, color, alpha=1.0, emission=0.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    # 添加Principled BSDF节点
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (*color, alpha)
    bsdf.inputs['Alpha'].default_value = alpha
    
    # 兼容不同Blender版本的发光设置
    if 'Emission Color' in bsdf.inputs:
        # Blender 4.0+
        bsdf.inputs['Emission Color'].default_value = (*color, 1.0) if emission > 0 else (0, 0, 0, 1)
        bsdf.inputs['Emission Strength'].default_value = emission
    elif 'Emission' in bsdf.inputs:
        # Blender 3.x
        bsdf.inputs['Emission'].default_value = (*color, emission) if emission > 0 else (0, 0, 0, 1)
    
    # 添加输出节点
    output = nodes.new(type='ShaderNodeOutputMaterial')
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    if alpha < 1.0:
        mat.blend_method = 'BLEND'
    
    return mat

# 创建材质
nickel_material = create_material("Nickel_Sample", (0.8, 0.2, 0.2))  # 红色
detector_material = create_material("Detector", (0.2, 0.2, 0.8), alpha=0.7)  # 半透明蓝色
beam_material = create_material("X_Ray_Beam", (0.0, 1.0, 1.0), alpha=0.8, emission=2.0)  # 明亮青色发光
axis_material = create_material("Axis", (0.5, 0.5, 0.5))  # 灰色

# 缩放因子：1 Blender单位 = 1000μm = 1mm
SCALE_FACTOR = 1000  # μm to mm

# 从代码中提取的物理参数（转换为mm）
SAMPLE_RADIUS = 1000 / SCALE_FACTOR  # 1000μm = 1mm
DETECTOR_DISTANCE = 1324900 / SCALE_FACTOR  # 1324.9mm
DETECTOR_SIZE = 427008 / SCALE_FACTOR  # 427mm
BEAM_SIZE = max(200 / SCALE_FACTOR * 20, 10)  # 增大光束尺寸以提高可见性，最小10mm

print(f"样品半径: {SAMPLE_RADIUS}mm")
print(f"探测器距离: {DETECTOR_DISTANCE}mm")
print(f"探测器尺寸: {DETECTOR_SIZE}mm")
print(f"光束尺寸: {BEAM_SIZE}mm")

# 1. 创建镍样品球体
bpy.ops.mesh.primitive_uv_sphere_add(
    radius=SAMPLE_RADIUS,
    location=(0, 0, 0)
)
sample_sphere = bpy.context.active_object
sample_sphere.name = "Nickel_Sample"
sample_sphere.data.materials.append(nickel_material)

# 2. 创建探测器平面
bpy.ops.mesh.primitive_plane_add(
    size=DETECTOR_SIZE,
    location=(DETECTOR_DISTANCE, 0, 0)
)
detector_plane = bpy.context.active_object
detector_plane.name = "Detector"
detector_plane.data.materials.append(detector_material)

# 旋转探测器使其垂直于X轴
detector_plane.rotation_euler = (0, math.radians(90), 0)

# 3. 创建X射线光束可视化
# 根据实际物理模拟，光束应该从样品前方延伸到探测器
# 在nickel.ipynb中，光束长度为2米（2,000,000μm），从x=-1e6到x=1e6
beam_start_distance = 1000000 / SCALE_FACTOR  # 1米，光束起点在样品前方
beam_total_length = 2000000 / SCALE_FACTOR  # 2米，光束总长度
beam_mesh = bpy.data.meshes.new("X_Ray_Beam")
bm = bmesh.new()

# 创建光束的长方体
bmesh.ops.create_cube(
    bm,
    size=1.0
)

# 缩放到正确的尺寸
for vert in bm.verts:
    vert.co.x *= beam_total_length / 2
    vert.co.y *= BEAM_SIZE / 2
    vert.co.z *= BEAM_SIZE / 2

# 移动到正确位置（从样品前方延伸到探测器后方）
# 光束中心位置：从-beam_start_distance开始，延伸beam_total_length
beam_center_x = -beam_start_distance + beam_total_length / 2
for vert in bm.verts:
    vert.co.x += beam_center_x

bm.to_mesh(beam_mesh)
bm.free()

beam_object = bpy.data.objects.new("X_Ray_Beam", beam_mesh)
bpy.context.collection.objects.link(beam_object)
beam_object.data.materials.append(beam_material)

# 4. 创建坐标轴
def create_axis(name, start, end, material, thickness=2.0):
    # 创建圆柱体作为轴线（比线条更容易看见）
    direction = Vector(end) - Vector(start)
    length = direction.length
    
    # 创建圆柱体
    bpy.ops.mesh.primitive_cylinder_add(
        radius=thickness,
        depth=length,
        location=((start[0] + end[0])/2, (start[1] + end[1])/2, (start[2] + end[2])/2)
    )
    
    axis_obj = bpy.context.active_object
    axis_obj.name = name
    
    # 旋转圆柱体以对齐方向
    if name == "X_Axis":
        axis_obj.rotation_euler = (0, math.radians(90), 0)
    elif name == "Y_Axis":
        axis_obj.rotation_euler = (0, 0, 0)
    elif name == "Z_Axis":
        axis_obj.rotation_euler = (math.radians(90), 0, 0)
    
    # 添加材质
    axis_obj.data.materials.append(material)
    
    return axis_obj

# 创建坐标轴（长度适中以便观察）
axis_length = max(DETECTOR_DISTANCE * 0.15, 150)  # 增加轴长度

x_axis = create_axis("X_Axis", (0, 0, 0), (axis_length, 0, 0), 
                    create_material("X_Axis_Mat", (1.0, 0.0, 0.0)))
y_axis = create_axis("Y_Axis", (0, 0, 0), (0, axis_length, 0), 
                    create_material("Y_Axis_Mat", (0.0, 1.0, 0.0)))
z_axis = create_axis("Z_Axis", (0, 0, 0), (0, 0, axis_length), 
                    create_material("Z_Axis_Mat", (0.0, 0.0, 1.0)))

# 5. 添加文本标签
def create_text(text, location, name, size=30):
    bpy.ops.object.text_add(location=location)
    text_obj = bpy.context.active_object
    text_obj.name = name
    text_obj.data.body = text
    text_obj.data.size = size
    return text_obj

# 添加标签
sample_label = create_text("镍样品\n(直径2mm)", (0, 0, SAMPLE_RADIUS + 30), "Sample_Label")
detector_label = create_text(f"探测器\n({DETECTOR_SIZE:.0f}mm x {DETECTOR_SIZE:.0f}mm)\n距离: {DETECTOR_DISTANCE:.1f}mm", 
                           (DETECTOR_DISTANCE, DETECTOR_SIZE/2 + 80, 0), "Detector_Label")
beam_label = create_text(f"X射线光束\n({BEAM_SIZE:.1f}mm x {BEAM_SIZE:.1f}mm)\n长度: {beam_total_length:.0f}mm\n波长: 0.1258Å", 
                        (beam_center_x, BEAM_SIZE + 50, 0), "Beam_Label")

# 添加坐标轴标签
x_label = create_text("X", (axis_length + 20, 0, 0), "X_Label", 25)
y_label = create_text("Y", (0, axis_length + 20, 0), "Y_Label", 25)
z_label = create_text("Z", (0, 0, axis_length + 20), "Z_Label", 25)

# 6. 改进的相机设置
# 删除默认相机（如果存在）
for obj in bpy.context.scene.objects:
    if obj.type == 'CAMERA':
        bpy.data.objects.remove(obj, do_unlink=True)

# 创建主相机 - 全景视角
bpy.ops.object.camera_add(
    location=(DETECTOR_DISTANCE * 0.6, DETECTOR_DISTANCE * 0.8, DETECTOR_DISTANCE * 0.5)
)
main_camera = bpy.context.active_object
main_camera.name = "Main_Camera"

# 设置相机参数以获得更好的视野
main_camera.data.lens = 35  # 广角镜头
main_camera.data.clip_start = 1  # 近裁剪面
main_camera.data.clip_end = DETECTOR_DISTANCE * 3  # 远裁剪面

# 让相机看向场景中心
main_camera.rotation_euler = (math.radians(55), 0, math.radians(40))

# 创建侧视相机
bpy.ops.object.camera_add(
    location=(0, DETECTOR_DISTANCE * 1.2, 0)
)
side_camera = bpy.context.active_object
side_camera.name = "Side_Camera"
side_camera.data.lens = 50
side_camera.rotation_euler = (math.radians(90), 0, math.radians(90))

# 创建顶视相机
bpy.ops.object.camera_add(
    location=(DETECTOR_DISTANCE * 0.5, 0, DETECTOR_DISTANCE * 1.0)
)
top_camera = bpy.context.active_object
top_camera.name = "Top_Camera"
top_camera.data.lens = 50
top_camera.rotation_euler = (0, 0, 0)

# 设置主相机为活动相机
bpy.context.scene.camera = main_camera

# 7. 改进的照明系统
# 删除默认灯光
for obj in bpy.context.scene.objects:
    if obj.type == 'LIGHT':
        bpy.data.objects.remove(obj, do_unlink=True)

# 主光源 - 太阳光（模拟实验室照明）
bpy.ops.object.light_add(
    type='SUN', 
    location=(DETECTOR_DISTANCE * 0.5, DETECTOR_DISTANCE * 0.5, DETECTOR_DISTANCE * 0.8)
)
main_light = bpy.context.active_object
main_light.name = "Main_Sun_Light"
main_light.data.energy = 5  # 增加亮度
main_light.data.angle = math.radians(15)  # 软阴影
main_light.rotation_euler = (math.radians(30), math.radians(30), 0)

# 补光 - 区域光（减少阴影）
bpy.ops.object.light_add(
    type='AREA', 
    location=(-DETECTOR_DISTANCE * 0.3, DETECTOR_DISTANCE * 0.3, DETECTOR_DISTANCE * 0.6)
)
fill_light = bpy.context.active_object
fill_light.name = "Fill_Light"
fill_light.data.energy = 200
fill_light.data.size = 100
fill_light.rotation_euler = (math.radians(45), math.radians(-30), 0)

# 背景光 - 点光源（整体照明）
bpy.ops.object.light_add(
    type='POINT', 
    location=(DETECTOR_DISTANCE * 0.8, -DETECTOR_DISTANCE * 0.4, DETECTOR_DISTANCE * 0.3)
)
back_light = bpy.context.active_object
back_light.name = "Background_Light"
back_light.data.energy = 300
back_light.data.shadow_soft_size = 10

# 环境光 - HDRI世界照明
bpy.context.scene.world.use_nodes = True
world_nodes = bpy.context.scene.world.node_tree.nodes
world_nodes.clear()

# 添加背景着色器
background = world_nodes.new(type='ShaderNodeBackground')
background.inputs['Color'].default_value = (0.1, 0.1, 0.15, 1.0)  # 深蓝色背景
background.inputs['Strength'].default_value = 0.3

# 添加输出节点
world_output = world_nodes.new(type='ShaderNodeOutputWorld')
bpy.context.scene.world.node_tree.links.new(background.outputs['Background'], world_output.inputs['Surface'])

# 8. 设置渲染引擎和视口着色
# 兼容不同版本的Blender渲染引擎
try:
    # 尝试使用新版本的EEVEE引擎
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
except TypeError:
    try:
        # 回退到旧版本的EEVEE引擎
        bpy.context.scene.render.engine = 'EEVEE'
    except TypeError:
        # 如果都不可用，使用Workbench引擎
        bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
        print("警告: 使用BLENDER_WORKBENCH渲染引擎，某些效果可能不可用")

# 设置EEVEE特效（如果可用）
if hasattr(bpy.context.scene, 'eevee'):
    eevee = bpy.context.scene.eevee
    
    # 安全设置EEVEE属性，兼容不同版本
    try:
        if hasattr(eevee, 'use_bloom'):
            eevee.use_bloom = True
            if hasattr(eevee, 'bloom_intensity'):
                eevee.bloom_intensity = 0.1
    except AttributeError:
        pass
    
    try:
        if hasattr(eevee, 'use_ssr'):
            eevee.use_ssr = True  # 屏幕空间反射
        if hasattr(eevee, 'use_ssr_refraction'):
            eevee.use_ssr_refraction = True
        if hasattr(eevee, 'ssr_quality'):
            eevee.ssr_quality = 0.25
    except AttributeError:
        pass
    
    try:
        if hasattr(eevee, 'use_gtao'):
            eevee.use_gtao = True  # 环境光遮蔽
    except AttributeError:
        pass

# 设置视口着色为材质预览
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'MATERIAL'
                space.overlay.show_wireframes = False
                space.overlay.show_floor = True
                space.overlay.show_axis_x = True
                space.overlay.show_axis_y = True
                space.overlay.show_axis_z = True
                break

# 9. 创建集合来组织对象
# 安全移动对象到集合的函数
def safe_move_to_collection(obj, target_collection, source_collection=None):
    """安全地将对象移动到目标集合"""
    if source_collection is None:
        source_collection = bpy.context.scene.collection
    
    # 检查对象是否在源集合中
    if obj.name in source_collection.objects:
        target_collection.objects.link(obj)
        source_collection.objects.unlink(obj)
    else:
        # 如果不在源集合中，只添加到目标集合
        if obj.name not in target_collection.objects:
            target_collection.objects.link(obj)

# 创建样品集合
sample_collection = bpy.data.collections.new("Sample")
bpy.context.scene.collection.children.link(sample_collection)
safe_move_to_collection(sample_sphere, sample_collection)

# 创建探测器集合
detector_collection = bpy.data.collections.new("Detector")
bpy.context.scene.collection.children.link(detector_collection)
safe_move_to_collection(detector_plane, detector_collection)

# 创建光束集合
beam_collection = bpy.data.collections.new("X_Ray_Beam")
bpy.context.scene.collection.children.link(beam_collection)
safe_move_to_collection(beam_object, beam_collection)

# 创建坐标轴集合
axis_collection = bpy.data.collections.new("Coordinate_Axes")
bpy.context.scene.collection.children.link(axis_collection)
for axis in [x_axis, y_axis, z_axis]:
    safe_move_to_collection(axis, axis_collection)

# 创建标签集合
label_collection = bpy.data.collections.new("Labels")
bpy.context.scene.collection.children.link(label_collection)
for label in [sample_label, detector_label, beam_label, x_label, y_label, z_label]:
    safe_move_to_collection(label, label_collection)

# 创建相机集合
camera_collection = bpy.data.collections.new("Cameras")
bpy.context.scene.collection.children.link(camera_collection)
for camera in [main_camera, side_camera, top_camera]:
    safe_move_to_collection(camera, camera_collection)

# 创建照明集合
light_collection = bpy.data.collections.new("Lighting")
bpy.context.scene.collection.children.link(light_collection)
for light in [main_light, fill_light, back_light]:
    safe_move_to_collection(light, light_collection)

# 10. 设置场景单位和视图
bpy.context.scene.unit_settings.system = 'METRIC'
bpy.context.scene.unit_settings.scale_length = 0.001  # 设置为毫米
bpy.context.scene.unit_settings.length_unit = 'MILLIMETERS'

# 设置视图范围
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.clip_start = 1
                space.clip_end = DETECTOR_DISTANCE * 5  # 扩大视图范围
                break

print("\n=== 增强版X射线衍射实验3D可视化完成 ===")
print(f"场景比例: 1 Blender单位 = 1毫米")
print(f"样品: 镍球体，直径 {SAMPLE_RADIUS * 2:.1f}mm，位于原点")
print(f"探测器: {DETECTOR_SIZE:.0f}mm x {DETECTOR_SIZE:.0f}mm，距离样品 {DETECTOR_DISTANCE:.1f}mm")
print(f"光束: {BEAM_SIZE:.1f}mm x {BEAM_SIZE:.1f}mm 截面，长度 {beam_total_length:.0f}mm，从样品前方 {beam_start_distance:.0f}mm 处开始")
print("\n照明系统:")
print("- Main_Sun_Light: 主太阳光源，模拟实验室照明")
print("- Fill_Light: 区域补光，减少阴影")
print("- Background_Light: 点光源背景照明")
print("- 环境光: 深蓝色背景世界照明")
print("\n相机系统:")
print("- Main_Camera: 主视角（当前活动）")
print("- Side_Camera: 侧视角")
print("- Top_Camera: 顶视角")
print("\n可视化元素:")
print("- 红色球体: 镍样品")
print("- 蓝色半透明平面: 探测器")
print("- 青色发光长方体: X射线光束（从样品前方延伸到探测器后方）")
print("- 彩色圆柱体: 坐标轴 (红X, 绿Y, 蓝Z)")
print("\n操作提示:")
print("- 使用数字键盘切换相机视角")
print("- 鼠标中键拖拽旋转视图")
print("- 滚轮缩放")
print("- Shift+鼠标中键平移")
print("- 在Outliner中可以切换不同集合的可见性")