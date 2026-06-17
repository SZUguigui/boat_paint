//=============================================================================
// line_fitter.h — 桥墩边界直线拟合模块
//
// 目标:
//   将边界提取得到的多个桥墩轮廓, 按其空间排列关系,
//   拟合出统一的边缘直线 (桥墩排列线).
//
// 策略 (V2):
//   中间主排: 收集所有同类型边 → RANSAC 全局拟合上/下边界线
//   两侧桥墩: 保留各自的 OBB 边 → 找 Y 最近邻 → 左右边中点相连
//=============================================================================
#pragma once
#include "boundary_extractor.h"
#include <vector>
#include <string>
#include <opencv2/core.hpp>

// ── 拟合后的直线 ────────────────────────────────────────
struct FittedLine {
    // 直线: ax + by + c = 0 (归一化, a²+b²=1)
    float a, b, c;

    // 线段端点 (世界坐标)
    cv::Point2f start, end;

    // 属于这条直线的边界 ID 列表
    std::vector<int> boundary_ids;

    // 边类型: "upper" / "lower" / "left" / "right"
    std::string edge_type;

    // 点数 / 内点率
    int total_points;
    float inlier_ratio;
};

// ── 拟合配置 ────────────────────────────────────────────
struct LineFitterParams {
    float group_distance  = 3.0f;    // 墩组间距 (m), 沿主方向
    float line_tolerance  = 0.15f;   // 直线拟合容忍度 (m)
    int   min_piers_per_line = 2;    // 少于 2 个墩参与的不出线
    float row_tolerance     = 1.0f;   // 分行容忍度 (m), 垂直桥轴方向
    float connect_min_dist = 4.0f;   // 侧墩连接最小距离 (m)
    float connect_max_dist = 20.0f;  // 侧墩连接最大距离 (m)
};

// ── 拟合器 ──────────────────────────────────────────────
class LineFitter {
public:
    explicit LineFitter(const LineFitterParams& params = LineFitterParams{});

    // ── OBB 边结构 (公开, 供 standalone 函数使用) ──────
    struct Edge2D {
        cv::Point2f p1, p2;          // 端点
        cv::Point2f midpoint;
        cv::Point2f direction;       // 单位方向
        cv::Point2f normal;          // 单位法线 (指向外)
        int boundary_id;
        std::string type;            // upper/lower/left/right
    };

    /// 从边界集合中拟合排列线 (中间主排: RANSAC 全局拟合)
    std::vector<FittedLine> fit(const std::vector<Boundary>& boundaries);

    /// 两侧桥墩: 保留 OBB 边 + 连接最近邻桥墩的左右边中点
    /// @return 连接线段列表
    std::vector<FittedLine> connect_neighbors(
        const std::vector<Boundary>& boundaries,
        const std::vector<FittedLine>& middle_lines);

    /// 保存可视化 (在二值图上叠加边界 + 拟合直线 + 邻居连线)
    /// base_image: 边界提取的二值图 (像素坐标)
    /// boundaries: 边界列表 (像素坐标)
    /// lines:      拟合直线 (世界坐标, 需要 u_min/v_min/pixel_size 转换)
    void save_visualization(const std::string& path,
                            const cv::Mat& base_image,
                            const std::vector<Boundary>& boundaries,
                            const std::vector<FittedLine>& lines,
                            float u_min, float v_min, float pixel_size) const;

private:
    LineFitterParams params_;

    // 为每个边界提取 4 条边
    std::vector<Edge2D> extract_edges(const std::vector<Boundary>& boundaries);

    // 计算单个边界的 OBB 并返回 4 条边
    std::vector<Edge2D> obb_edges(const Boundary& b);

    // 将边按位置和方向分组
    std::vector<std::vector<Edge2D>> group_edges(
        const std::vector<Edge2D>& edges);

    // 从一组边中拟合一条共识直线
    FittedLine fit_consensus_line(const std::vector<Edge2D>& group);
};
