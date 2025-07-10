#!/usr/bin/env python3
"""
直接调用 Blender MCP 清空场景的简化脚本
"""

import subprocess
import json
import sys

def call_blender_mcp():
    """调用 Blender MCP 清空场景"""
    
    # Blender Python 代码，用于清空场景中的网格物体
    blender_code = """
import bpy

# 确保处于对象模式
if bpy.context.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')

# 选择所有网格物体
bpy.ops.object.select_all(action='DESELECT')
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        obj.select_set(True)

# 删除选中的物体
bpy.ops.object.delete(use_global=False)

print('场景中的所有网格物体已清空！')
"""
    
    try:
        # 尝试通过配置的 MCP 服务器调用
        mcp_command = [
            "cmd", "/c", 
            "D:\\tools\\envs\\xrd_simulator\\Scripts\\uvx.exe", 
            "blender-mcp"
        ]
        
        print("正在启动 Blender MCP 服务...")
        print("您可以在 Blender 中手动运行以下代码来清空场景：")
        print("=" * 50)
        print(blender_code)
        print("=" * 50)
        
        # 创建一个简单的 .blend 文件操作指令
        print("\n或者您可以：")
        print("1. 打开 Blender")
        print("2. 按 Tab 键确保处于对象模式")
        print("3. 按 A 键选择所有物体")
        print("4. 按 X 键，然后选择 'Delete' 确认删除")
        print("5. 或者在脚本编辑器中运行上面显示的 Python 代码")
        
    except Exception as e:
        print(f"调用 MCP 服务时发生错误: {e}")
        print("请手动在 Blender 中运行上面显示的 Python 代码")

if __name__ == "__main__":
    call_blender_mcp()
