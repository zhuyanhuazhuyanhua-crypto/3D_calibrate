
# Cultural Heritage 3D Reconstruction

这是一个面向文物建筑的 3D 重建项目骨架，提供从多视角图像到稠密点云、网格、纹理与语义标注的模块化流水线。项目设计以可替换外部工具（如 COLMAP、OpenMVS）和可选深度学习语义增强（Detectron2 / SAM）为目标，同时在缺少外部依赖时提供纯 Python 的回退实现以便测试与演示。

**概览**
- **项目结构**: 仓库包含 `config/`、`data/`、`src/`、`scripts/` 等目录，按职责划分模块化代码。
- **目标**: 提供可扩展的 SfM/MVS → Meshing → Texturing → Semantic pipeline，易于集成第三方二进制工具与深度学习模型。

**快速开始**
- 克隆并进入项目目录:

```powershell
cd d:/wenwu/cultural_heritage_3d_reconstruction
```
- 安装 Python 依赖（建议在虚拟环境中）:

```powershell
python -m pip install -r requirements.txt
```
- 运行演示（演示模式会在未提供图片时生成示例立方体网格）:

```powershell
python -m src.main demo
```

**主要命令与测试**
- 运行内置单元测试:

```powershell
python -m unittest discover -s tests -v
```

**模块说明**
- `config/`: YAML 配置文件（示例: `dataset.yaml`, `reconstruction.yaml`, `visualization.yaml`）。
- `data/`: 数据目录，包含 `raw/`, `processed/`, `outputs/`。
- `src/acquisition/`: 图像与点云读取、EXIF/POS 解析模块。
- `src/reconstruction/`: 核心重建模块，包括 `sfm_pipeline.py`, `mvs_pipeline.py`, `meshing.py`, `texturing.py`, `semantic_enrichment.py`。
- `src/digital_twin/`: 数字孪生抽象与元数据管理（版本管理等）。
- `src/visualization/`: 本地渲染与 Web 导出工具（`scripts/export_to_web.py` 用于生成 Potree/glTF 导出）。
- `src/utils/`: 常用辅助函数（文件 IO、坐标转换、日志等）。

**语义增强（可选）**
- 文件: `src/reconstruction/semantic_enrichment.py`。
- 支持: Detectron2（目标/实例分割）与 Segment-Anything（SAM）输出的接入与处理。模块包含：
	- 将检测/分割结果转换为 per-pixel class-confidence maps 的转换函数（便于按相机视角投票回顶点）。
	- 顶点投票融合（含置信度加权）和基于网格邻接的空间平滑（`_build_vertex_adjacency` / `_smooth_vertex_labels`）。
	- 输出语义网格（支持 Open3D 写入或简单 PLY 回退）。

**可选系统依赖（用于更高质量重建）**
- COLMAP: 用于可靠的 SfM（相机位姿 + 稀疏点云）。若安装，配置 `colmap_path` 可启用完整 SfM 流程。
- OpenMVS / OpenMVS tools: 可用于稠密重建与高质量网格化。
- Open3D / trimesh: 提供高级点云/网格处理、Poisson 重建与 glTF 导出支持。
- PotreeConverter: 将网格/点云导出为 Potree Web 浏览格式。

若缺少上述工具，项目会使用轻量级回退实现以保证演示与单元测试可运行。

**示例工作流（简要）**
- 演示快速运行（不依赖外部二进制）:

```powershell
python -m src.main demo
```
- 若已配置 COLMAP（具有图像数据）:

```powershell
# 修改 config/reconstruction.yaml 指定 colmap_path 与输入/输出 路径
python -m src.main demo
```

**调试与常见问题**
- 若 SfM 阶段没有运行，请确认 `colmap` 二进制在 `PATH` 中或在 `config/reconstruction.yaml` 中设置了 `colmap_path`。
- 若纹理或 meshing 行为不如预期，安装 `open3d` 与 `trimesh` 可以启用更完整的实现。

**贡献与许可证**
- 欢迎提交 issue 与 PR。当前仓库未包含特定许可证，请在需要时添加 `LICENSE` 文件。

---

如果你希望我：
- 将语义平滑直接集成到 `project_masks_to_mesh_vertices` 并添加相应单元测试；或
- 提供一个端到端演示脚本（从合成分割结果到语义网格）——告诉我你想先做哪个，我会继续实现并运行测试。
