#!/usr/bin/env python3
"""
清空 Blender 场景中的所有物体
"""

import bpy

def clear_all_objects():
    """删除场景中的所有网格物体"""
    # 确保处于对象模式
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # 取消选择所有物体
    bpy.ops.object.select_all(action='DESELECT')
    
    # 选择所有网格物体
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    
    # 删除选中的物体
    bpy.ops.object.delete(use_global=False)
    
    print("已清空场景中的所有网格物体")

def clear_all_objects_including_defaults():
    """删除场景中的所有物体，包括默认的立方体、灯光、相机"""
    # 确保处于对象模式
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # 选择所有物体
    bpy.ops.object.select_all(action='SELECT')
    
    # 删除所有选中的物体
    bpy.ops.object.delete(use_global=False)
    
    print("已清空场景中的所有物体（包括相机、灯光等）")

if __name__ == "__main__":
    # 只清空网格物体（保留相机和灯光）
    clear_all_objects()
    
    # 如果要清空所有物体（包括相机、灯光），取消下面这行的注释
    # clear_all_objects_including_defaults()
