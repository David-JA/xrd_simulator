#!/usr/bin/env python3
"""
通过 MCP 调用 Blender 清空场景
"""

import subprocess
import sys
import os

def run_blender_script(script_path):
    """运行 Blender 脚本"""
    try:
        # 尝试通过 blender 命令行运行脚本
        cmd = [
            "blender",
            "--background",
            "--python", script_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Blender 脚本执行成功！")
            print(result.stdout)
        else:
            print(f"Blender 脚本执行失败，返回码: {result.returncode}")
            print(f"错误信息: {result.stderr}")
            
    except FileNotFoundError:
        print("未找到 Blender 可执行文件。请确保 Blender 已安装并添加到 PATH 环境变量中。")
        print("或者您可以手动在 Blender 中运行以下 Python 代码：")
        print_blender_code()
    except Exception as e:
        print(f"执行过程中发生错误: {e}")

def print_blender_code():
    """打印可在 Blender 中直接运行的代码"""
    print("""
在 Blender 中，您可以打开脚本编辑器并运行以下代码来清空场景：

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

print("场景已清空！")
""")

if __name__ == "__main__":
    script_path = "clear_blender_scene.py"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_script_path = os.path.join(current_dir, script_path)
    
    if os.path.exists(full_script_path):
        run_blender_script(full_script_path)
    else:
        print(f"脚本文件不存在: {full_script_path}")
        print_blender_code()
