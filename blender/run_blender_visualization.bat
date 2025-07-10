@echo off
echo ================================================
echo X射线衍射实验3D可视化启动脚本
echo ================================================
echo.

:: 检查Blender是否在PATH中
blender --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Blender可执行文件
    echo.
    echo 请确保:
    echo 1. 已安装Blender (推荐版本3.0或更高)
    echo 2. Blender已添加到系统PATH环境变量中
    echo.
    echo 或者手动运行:
    echo "C:\Program Files\Blender Foundation\Blender 3.x\blender.exe" --python "%~dp0xrd_visualization_enhanced.py"
    echo.
    pause
    exit /b 1
)

echo 正在启动Blender并加载X射线衍射可视化场景...
echo.
echo 场景包含:
echo - 红色球体: 镍样品 (直径2mm)
echo - 蓝色平面: 探测器 (427mm x 427mm, 距离1324.9mm)
echo - 黄色光束: X射线传播路径 (0.2mm x 0.2mm截面)
echo - 坐标轴和标签
echo.

:: 运行Blender脚本
blender --python "%~dp0xrd_visualization_enhanced.py"

if %errorlevel% neq 0 (
    echo.
    echo 错误: Blender脚本执行失败
    echo 请检查脚本文件是否存在: %~dp0xrd_visualization_enhanced.py
    pause
    exit /b 1
)

echo.
echo 可视化场景已创建完成!
echo 请在Blender中验证空间关系是否与代码中的物理建模一致。
echo.
pause