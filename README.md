# Cultural Heritage 3D Reconstruction

这是一个用于文物建筑 3D 重建的代码骨架，包含数据组织、重建流程（占位）、网格生成与可视化示例。

快速开始（演示模式）:

```powershell
cd d:/wenwu/cultural_heritage_3d_reconstruction
python -m src.main demo
```

说明:
- 演示模式会在没有输入图像时生成一个简单的立方体网格并保存到 `data/outputs/`。
- 真正的 SfM/MVS 流程应集成 COLMAP/OpenMVG/OpenMVS 或其它工具；当前实现为可运行的结构化骨架。
