//=============================================================================
// ground_remover.cpp — 船底地面点剔除模块实现
//
// 算法流程:
//   [输入点云]
//       │
//       ├─ 步骤1: 估计法线 (KD-tree, radius=0.10m)
//       │
//       ├─ 步骤2: 法线预筛选
//       │    │  ground_candidates = {点 | |normal_z| > 0.7}  — 法线近似竖直
//       │    │  non_ground = {点 | |normal_z| <= 0.7}         — 法线水平 (桥墩侧面)
//       │    │
//       │    └─ 注: non_ground 直接保留, 不做后续处理
//       │
//       ├─ 步骤3: 迭代 RANSAC 平面拟合 (仅对 ground_candidates)
//       │    │  for i in 1..N:
//       │    │    RANSAC 拟合最大平面
//       │    │    若内点数 < min_inliers → 停止迭代
//       │    │    将内点标记为地面, 从候选集中移除
//       │    │
//       │    └─ 剩余候选点 → 放回 non_ground (可能是桥墩顶面等)
//       │
//       └─ 输出: non_ground (保留下来的所有点)
//=============================================================================
#include "ground_remover.h"
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <iostream>

GroundRemovalResult remove_boat_bottom(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const GroundRemovalParams& params)
{
    GroundRemovalResult result;
    if (!cloud || cloud->empty()) return result;

    const size_t N = cloud->size();
    std::cout << "\n========== Ground Removal Start ==========\n";
    std::cout << "Input points: " << N << "\n";

    // ── 步骤1: 估计法线 ────────────────────────────────
    std::cout << "[1/3] Estimating normals (radius=" << params.normal_radius << "m)...\n";
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(cloud);

    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(params.normal_radius);
    ne.compute(*normals);

    // ── 步骤2: 法线预筛选 ──────────────────────────────
    std::cout << "[2/3] Filtering by normal direction (|nz| > "
              << params.normal_z_threshold << " → ground candidate)...\n";

    pcl::PointIndices::Ptr ground_candidates(new pcl::PointIndices);
    pcl::PointIndices::Ptr direct_keep(new pcl::PointIndices);  // 法线水平, 直接保留

    for (size_t i = 0; i < N; ++i) {
        // 跳过无效法线
        if (!std::isfinite(normals->points[i].normal_z)) {
            continue;  // 无效法线的点不保留 (边界/噪声)
        }
        if (std::fabs(normals->points[i].normal_z) > params.normal_z_threshold) {
            ground_candidates->indices.push_back(static_cast<int>(i));
        } else {
            direct_keep->indices.push_back(static_cast<int>(i));
        }
    }

    std::cout << "  Ground candidates (|nz| > threshold): "
              << ground_candidates->indices.size() << " pts\n";
    std::cout << "  Direct keep (|nz| <= threshold):      "
              << direct_keep->indices.size() << " pts\n";

    // ── 步骤3: 迭代 RANSAC 平面拟合 ────────────────────
    std::cout << "[3/3] Iterative RANSAC plane fitting...\n";
    std::cout << "  Plane threshold: " << params.plane_threshold
              << "m | Max iterations per plane: " << params.plane_max_iters
              << " | Max planes: " << params.plane_max_iterations << "\n";

    // 提取 ground_candidates 对应的子点云用于 RANSAC
    pcl::PointCloud<pcl::PointXYZI>::Ptr candidates_cloud(
        new pcl::PointCloud<pcl::PointXYZI>);
    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(ground_candidates);
    extract.filter(*candidates_cloud);

    // 维护 "已被标记为地面" 的索引集合 (在 candidates_cloud 中的索引)
    std::vector<bool> is_ground(candidates_cloud->size(), false);
    int total_ground_removed = 0;

    pcl::SACSegmentation<pcl::PointXYZI> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(params.plane_threshold);
    seg.setMaxIterations(params.plane_max_iters);

    for (int iter = 1; iter <= params.plane_max_iterations; ++iter)
    {
        // 构建"尚未被标记"的点的索引列表
        pcl::PointIndices::Ptr remaining_indices(new pcl::PointIndices);
        for (size_t i = 0; i < candidates_cloud->size(); ++i) {
            if (!is_ground[i])
                remaining_indices->indices.push_back(static_cast<int>(i));
        }

        if (remaining_indices->indices.size() < static_cast<size_t>(params.plane_min_inliers)) {
            std::cout << "  Iter " << iter << ": only "
                      << remaining_indices->indices.size()
                      << " pts remain, stopping.\n";
            break;
        }

        // RANSAC 找最大平面
        pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);

        seg.setInputCloud(candidates_cloud);
        seg.setIndices(remaining_indices);
        seg.segment(*plane_inliers, *coeff);

        if (plane_inliers->indices.size() < static_cast<size_t>(params.plane_min_inliers)) {
            std::cout << "  Iter " << iter << ": largest plane has only "
                      << plane_inliers->indices.size()
                      << " inliers < " << params.plane_min_inliers
                      << " min, stopping.\n";
            break;
        }

        // 标记这些点为地面
        for (int idx : plane_inliers->indices) {
            is_ground[idx] = true;
        }
        total_ground_removed += static_cast<int>(plane_inliers->indices.size());

        // 打印拟合的平面参数
        std::cout << "  Iter " << iter << ": plane inliers="
                  << plane_inliers->indices.size()
                  << " | model: a=" << coeff->values[0]
                  << " b=" << coeff->values[1]
                  << " c=" << coeff->values[2]
                  << " d=" << coeff->values[3] << "\n";
    }

    std::cout << "  Total ground points removed via RANSAC: "
              << total_ground_removed << "\n";

    // ── 步骤4: 组装输出 ────────────────────────────────
    // non_ground = direct_keep + candidates中未被标记为地面的点
    result.non_ground.reset(new pcl::PointCloud<pcl::PointXYZI>);
    result.ground.reset(new pcl::PointCloud<pcl::PointXYZI>);

    // 先加入直接保留的点 (法线水平方向)
    for (int idx : direct_keep->indices) {
        result.non_ground->push_back(cloud->points[idx]);
    }

    // 从 candidates 中: 地面 → ground, 非地面 → non_ground
    for (size_t i = 0; i < candidates_cloud->size(); ++i) {
        if (is_ground[i]) {
            result.ground->push_back(candidates_cloud->points[i]);
        } else {
            result.non_ground->push_back(candidates_cloud->points[i]);
        }
    }

    result.ground_points_removed = static_cast<int>(result.ground->size());

    std::cout << "\n--- Ground Removal Results ---\n";
    std::cout << "Kept (non-ground):    " << result.non_ground->size() << " pts\n";
    std::cout << "Removed (ground):     " << result.ground->size() << " pts\n";
    std::cout << "==========================================\n\n";

    return result;
}
