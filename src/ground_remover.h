//=============================================================================
// ground_remover.h — 船底地面点剔除模块接口
//
// 核心思路:
//   船底是弧形的大面积曲面, 法线始终近似指向 ±Z
//   桥墩是孤立的竖直柱体, 侧面法线指向水平方向
//   利用法线方向 + 迭代RANSAC平面拟合分离二者
//
// 三步策略:
//   1. 法线预筛选 — |normal_z| > 0.7 → 地面候选
//   2. 迭代RANSAC — 多次平面拟合剥除弧形船底
//   3. 聚类保护   — 防止桥墩顶面被误删
//=============================================================================
#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

/// 地面剔除配置参数
struct GroundRemovalParams
{
    // 法线估计
    float normal_radius       = 0.10f;   // 法线搜索半径 (m)

    // 法线预筛选: |normal_z| > 此值 → 地面候选
    float normal_z_threshold  = 0.70f;

    // RANSAC平面拟合
    float plane_threshold     = 0.15f;   // 点到平面距离容忍度 (m) — 较大以覆盖船底弧度
    int   plane_max_iters     = 500;     // 每次RANSAC最大迭代次数
    int   plane_max_iterations = 10;      // 迭代RANSAC次数 (剥多层)
    int   plane_min_inliers    = 500;   // 最小内点数, 低于此值停止迭代

    // 聚类保护: 每个桥墩簇最小点数
    int   cluster_min_size    = 50;
    float cluster_tolerance   = 0.3f;    // 聚类间距 (m)
};

/// 地面剔除结果
struct GroundRemovalResult
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr non_ground;  // 移除地面后的点云 (桥墩+结构)
    pcl::PointCloud<pcl::PointXYZI>::Ptr ground;      // 被剔除的地面点 (船底)
    int   ground_points_removed = 0;   // 被移除的地面点数
};

/// 执行船底地面剔除
/// @param cloud    输入点云 (PointXYZI)
/// @param params   配置参数
/// @return         剔除结果: non_ground = 保留下来的点, ground = 被移除的点
GroundRemovalResult remove_boat_bottom(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const GroundRemovalParams& params = GroundRemovalParams{});
