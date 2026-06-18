# Pier Detect — Boat Bottom Coverage Path Planning

从桥梁扫描 PCD 点云中移除船底弧面，提取桥墩 XY 投影边界，并生成喷涂机器人覆盖路径。

## Pipeline

```
PCD 点云
  │
  ├─► 地面剔除 (Ground Removal)
  │     法线方向预筛选 + 迭代 RANSAC 平面拟合
  │     → non_ground.pcd / ground.pcd
  │
  ├─► 边界提取 (Boundary Extraction)
  │     XY 投影 → 栅格化 → 形态学 → 边界追踪
  │     → boundaries.png / work_area.png
  │
  ├─► 直线拟合 (Line Fitting)
  │     RANSAC 全局拟合桥墩排列线 + OBB 边界
  │     → fitted_lines 可视化
  │
  └─► 覆盖路径规划 (Coverage Planning)
        滑动窗口 + 残区补全贪心放置
        → coverage_plan.json + 可视化图像
```

## 几何模型

```
baselink (1.3m×0.85m)  ←0.616m→  base (max 1.17m×1.17m)
   小车底盘                          喷漆区域
```

- `base` 在工作区域内、不碰红色边界、不与已放置 base 重叠
- `baselink` 可在工作区外，须满足"最近边界在行进方向左侧"约束
- 两者轴对齐，不旋转

## 覆盖路径规划策略

### Phase 1: 滑动窗口

| 参数 | 值 |
|------|--------|
| 窗口初始尺寸 | 1.17m × 1.17m |
| X 步长 | 1.4m |
| Y 步长 | 1.17m |
| 扫描方向 | 左→右，上→下 |

每个网格点尝试：
1. 原始中心 + 全尺寸 → 检测灰色区域 + 不与已覆盖重叠
2. 受阻 → 移动中心 ±0.3m (5cm 步长)
3. 仍受阻 → 逐步缩小窗口 (0.95→0.15)
4. 无法放置 → 跳过

### Phase 2: 残区补全

自适应步长扫描剩余未覆盖区域 (~5000 候选/轮)，每轮选覆盖最多者放置。

### 后处理

蛇形排序 (奇数行左→右，偶数行右→左) + 路径插值 (最大间距 1.4m)。

## 构建

### 依赖

- **PCL** 1.13+ (io, features, filters, segmentation, search)
- **OpenCV** 4+ (core, imgproc, highgui, imgcodecs)
- **CMake** 3.16+
- **vcpkg** (包管理)
- **MSVC** 2022+ (Windows)

### 编译

```powershell
# 安装依赖
vcpkg install pcl[visualization] opencv4

# 配置 & 构建
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

或使用 Visual Studio 打开 `build/pier-detect.sln`。

## 使用

```powershell
.\build\Release\pier_detect.exe <input.pcd> [options]
```

### 选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--normal-radius N` | 0.10 | 法线搜索半径 (m) |
| `--normal-z-thresh N` | 0.70 | 法线 Z 分量阈值，\|nz\| > 此值视为地面候选 |
| `--plane-threshold N` | 0.10 | RANSAC 平面拟合公差 (m) |
| `--plane-iters N` | 500 | 每次 RANSAC 最大迭代次数 |
| `--plane-rounds N` | 5 | 迭代 RANSAC 轮数 |
| `--min-inliers N` | 1000 | 最小内点数，低于此值停止 |
| `--output DIR` | output | 输出目录 |
| `--skip-ground` | false | 跳过地面剔除 |
| `--ground-pcd PATH` | — | 单独指定地面 PCD |
| `--help` | — | 显示帮助 |

### 示例

```powershell
# 完整流程
.\build\Release\pier_detect.exe data\scan.pcd

# 跳过地面剔除 (已处理过的点云)
.\build\Release\pier_detect.exe output\non_ground.pcd --skip-ground

# 指定地面 PCD 作为工作区域
.\build\Release\pier_detect.exe output\non_ground_non_person_noise.pcd --skip-ground --ground-pcd output\ground_non_person_noise.pcd
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `output/non_ground.pcd` | 移除地面后的点云 |
| `output/ground.pcd` | 被移除的地面点 |
| `output/boundaries.png` | 边界可视化 |
| `output/work_area.png` | 工作区（灰=可喷涂，红=边界，黑=不可达） |
| `output/01_base_coverage.png` | base 覆盖区域 + 编号 |
| `output/02_baselink_positions.png` | baselink 位置 + 行进方向 |
| `output/03_path.png` | 连续路径 + 跳变段标注 |
| `output/04_baselink_sequence.png` | 路径序号 |
| `output/05_overview.png` | 综合图 |
| `output/coverage_plan.json` | 路径规划 JSON |

## 项目结构

```
src/
├── main.cpp                  # 入口，CLI 解析
├── ground_remover.h/cpp      # 地面剔除
├── boundary_extractor.h/cpp  # XY 投影边界提取
├── line_fitter.h/cpp         # 桥墩边界直线拟合
└── coverage_planner.h/cpp    # 覆盖路径规划
```

## 更新日志

### v1.0.0 (2026-06-17) — 初始版本

- 地面剔除：法线方向预筛选 + 迭代 RANSAC 平面拟合
- 边界提取：XY 投影栅格化 + OpenCV findContours / Alpha Shape / Marching Squares
- 直线拟合：RANSAC 全局拟合桥墩排列线
- 覆盖规划：自适应行放置 + 残区补全 (Pass 1 + Pass 2)

### v1.1.0 (2026-06-17) — 覆盖规划 V2

- Pass 1 base 尺寸修复：从灰色像素边界扩展 (`expand_base`) 到最大合法矩形
- Pass 1 X 推进修复：基于实际覆盖宽度推进，消除行内空隙
- Pass 2 性能优化：自适应扫描步长 (~5000 候选/轮) + 100 轮上限
- Pass 2 Y-snap 后增加 base 合法性验证
- 增加 base 重叠检测：`is_base_valid` / `expand_base` 支持 `uncovered` mask

### v1.2.0 (2026-06-17) — 覆盖规划 V3 全局贪心

- 去掉行网格机制，改为全局贪心扫描循环
- 自适应步长扫描 (~8000 候选/轮)，每轮取覆盖最多者放置
- baselink 验证失败时标记该位置避免死循环
- 问题：速度太慢，迭代后期生成碎片化小 base

### v1.3.0 (2026-06-17) — 覆盖规划 V4 两阶段

- Phase 1：条带行放置，沿自适应行逐行快速填充
- Phase 2：残区补全扫描 (最多 100 轮)
- 行间距改为 ≥ 1.17m (`base_max`) 保证行间不重叠

### v1.4.0 (2026-06-18) — 覆盖规划 V5 滑动窗口

- Phase 1：固定 1.17m 条带 + 滑动窗口
  - X 步长 1.4m，Y 步长 1.17m，左→右 上→下扫描
  - 窗口初始 1.17×1.17m
  - 受阻时优先移动中心 (±0.3m)，其次缩小窗口 (0.95→0.15)
  - 不重叠、不碰红色边界
- Phase 2：残区补全 (不变)
- 后处理：蛇形排序 + 路径插值 (最大间距 1.4m)

## License

MIT
