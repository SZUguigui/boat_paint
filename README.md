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
        两阶段规划 (边界环 + 滑动窗口) + 连通区域聚类后处理
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

两阶段规划 + 连通区域聚类后处理。

### Phase 1 — 边界环 (蓝色)

沿侧墩 AABB 四条边平铺一圈 base：
- 步长 = `base_max` (1.17m)，中心偏移 0.75m (保证 baselink 不碰红色边界)
- 每条边固定行进方向，使侧墩边界始终在 base 左侧
- `expand_base` 从 0.2m 起始四向扩展，面积 < 0.16m² 跳过
- 跳过宽度 > 3m 的中轴桥墩

### Phase 2 — 内部滑动窗口 (绿色)

- X 步长 1.4m，Y 步长 1.17m，左→右 上→下扫描
- 窗口初始 1.17×1.17m
- 受阻时优先移动中心 (±0.3m)，其次缩小窗口 (0.95→0.35)
- 放置后 `expand_base` 最大化，边长 < 0.4m 跳过
- 近边界时强制 `boundary_on_left` 检查

### 后处理 V7

1. **连通区域聚类**: `connectedComponents` 将灰色区域分组
2. **区域内排序**: ring (P1) 按极角排序成环，fill (P2) 按 1.17m 带宽蛇形排列
3. **区域间链接**: 贪心 NN + 2-opt 优化 (≤4 趟)，最小化区域间跳转距离
4. **路径插值**: 相邻 baselink 间距 > 1.4m 时线性插值

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

### v1.5.0 (2026-06-18) — 覆盖规划 V6 三阶段

- **方向系统扩展**: `direction` 从 2 方向 (±X) 扩到 4 方向 (0=+X, 1=-X, 2=+Y, 3=-Y)
- **Phase 1 边界优先**: 沿红色边界带放置 base，方向保证边界在左侧
  - 对边界做 dilate 得到"边界带"，在带内扫描未覆盖像素
  - 4 方向候选，`expand_base` 从 0.2m 起始最大化
- **Phase 2 内部滑动窗口**: X 轴方向规则填充 (同 V5 Phase 1)
- **Phase 3 MER 填缝**: 最大空矩形贪心，自适应步长 ~5000 候选/轮
- **可视化**: 三色区分 — 蓝色 (P1) / 绿色 (P2) / 黄色 (P3)，箭头标注 base 方向

### v1.6.0 (2026-06-18) — 侧墩 AABB + V6 修复

- **侧墩 AABB 简化**: 红色侧墩轮廓替换为外接矩形，简化后续 base 放置
- **P1 固定位置**: 每侧墩 6 个固定候选位 (上1/下1/左2/右2)
- **边界在左检查修复**: `boundary_on_left` 的 `check_dist` 从 0.6m 扩展到 `base_max + 0.6f` (1.77m)，覆盖膨胀后的 base
- **offset 修复**: P1 偏移从 0.6m 调整到 0.75m (>baselink_x/2 = 0.65m)

### v1.7.0 (2026-06-18) — 覆盖规划 V7 优化

- **P1 环重新设计**: 从 6 固定候选位改为沿 AABB 四边均匀步长平铺，每条边固定行进方向 (CCW 环)
- **最小 base 阈值**: `MIN_BASE_EDGE = 0.4m`, `MIN_BASE_AREA = 0.16m²`，应用于 P1/P2/P3，消除碎片化小方块
- **P2 expand_base**: 滑动窗口放置后调用 `expand_base` 最大化 base 尺寸
- **连通区域聚类后处理**:
  - `connectedComponents` 将灰色区域分组为独立 region
  - 区域内: ring (P1) 极角排序成环, fill (P2+P3) 按 1.17m 带宽蛇形排列
  - 区域间: 贪心 NN + 2-opt 优化 (≤4 趟)，大幅减少跨区域跳转距离
- **效果**: 路径长度从 ~12,300m 降到 ~2,500–3,500m (3–5×)，消除所有 <0.16m² 碎片

### v1.8.0 (2026-06-18) — 移除 Phase 3 MER

- **移除 P3 MER 填缝**: P1+P2 已经覆盖 ~74%，P3 贡献有限但增加大量航点和路径长度
- **数据对比** (同一数据):
  - 有 P3: 2132 waypoints, 72.8% coverage
  - 无 P3: 2119 waypoints, 74.1% coverage, ~8,944m
- 可视化从三色简化为双色 (蓝=环, 绿=填充)

## License

MIT
