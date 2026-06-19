# 相机点云语义 2.5D 建图：数据要求

本项目使用的是你的 `相机 + LiDAR 点云` 数据，不依赖外部数据集。

## 必须提供

```text
your_dataset/
|-- camera/
|   |-- 000000.png
|   |-- 000001.png
|-- lidar/
|   |-- 000000.ply  # 或 .bin / .npy，文件名必须和图像一一对应
|   |-- 000001.ply
|-- poses/          # 可选；提供后启用多帧滑窗融合
|   |-- 000000.txt  # 4x4 world_from_lidar 矩阵，共 16 个数
`-- calib.json
```

`calib.json` 示例：

```json
{
  "camera_intrinsics": [
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
  ],
  "lidar_to_camera": [
    [r11, r12, r13, tx],
    [r21, r22, r23, ty],
    [r31, r32, r33, tz],
    [0.0, 0.0, 0.0, 1.0]
  ]
}
```

还需要保证图像和点云时间上对应。最简单的办法是同名，如 `camera/000023.png` 对应 `lidar/000023.ply`。

## 位姿：局部图与多帧融合

单帧避障图只需要相机、点云和标定。若要将连续帧融合成稳定局部子图，还需要每帧 `world_from_lidar` 位姿；它可以来自 LiDAR SLAM、视觉-LiDAR SLAM、RTK，或外部定位，不要求 IMU 必须存在。

## 是否需要人工标注

不需要，第一版可以直接运行：

```text
SegFormer：道路、建筑、树/植被等像素级场景语义
YOLO：person、car、bus、truck、motorcycle、bicycle
LiDAR：深度、位置、高度、障碍几何
```

但若目标是项目验收级精度，推荐后续标注一小批自己的数据：

```text
语义分割：100-300 张图像做像素级 mask
YOLO：100-300 张图像做车、人框标注
评估集：至少 50 张从未参与训练的图像
```

无人机高视角下，车和人通常很小，COCO 预训练 YOLO 可以先用，但最终建议用自己的航拍样本微调。

## 运行

```bash
python scripts\fuse_camera_lidar_semantic_25d.py --dataset path\to\your_dataset --window-frames 5 --output-dir outputs\your_semantic_25d
```

首次运行会下载 SegFormer 和 YOLO 的预训练权重。输出包括每帧语义 2.5D 栅格 PNG、可传输的 `.npz` 多层数据和 YOLO 检测报告 `report.json`。
