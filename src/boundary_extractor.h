//=============================================================================
// boundary_extractor.h — XY投影边界提取模块
//
// 输入: non_ground.pcd (地面剔除后的点云)
// 流程: XY投影 → 栅格化二值图 → 形态学 → 边界提取 → 输出
//
// 三种边界提取方法:
//   FINDCONTOURS   - OpenCV 连通域边界追踪 (最快, 推荐)
//   ALPHA_SHAPE    - Delaunay三角剖分 + 半径阈值 (点级精度)
//   MARCHING_SQUARES - 网格水平集提取 (精确到像素)
//=============================================================================
#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <memory>

// ── 边界提取方法 ────────────────────────────────────────
enum class BoundaryMethod {
    FINDCONTOURS,       // OpenCV findContours
    ALPHA_SHAPE,        // Delaunay triangulation + radius
    MARCHING_SQUARES    // Grid marching squares
};

// ── 单条边界 (闭合/非闭合轮廓) ──────────────────────────
struct Boundary {
    int id;
    std::vector<cv::Point2f> points_m;   // 世界坐标 (米)
    std::vector<cv::Point>   points_px;  // 像素坐标
    bool is_closed;
    double area_m2;
    double perimeter_m;
    std::string source;                   // "findContours" / "alpha_shape" / "marching_squares"
};

// ── 边界提取参数 ────────────────────────────────────────
struct BoundaryParams {
    // 栅格化
    float pixel_size       = 0.01f;  // 固定分辨率 (m)

    // 形态学 (0 = 跳过)
    int morph_close_k      = 1;      // 闭运算核大小 (补洞)
    int morph_open_k       = 1;      // 开运算核大小 (去毛刺)

    // 边界提取
    BoundaryMethod method  = BoundaryMethod::FINDCONTOURS;

    // 过滤
    double min_contour_area = 0.3;   // 最小轮廓面积 (m²)

    // Alpha Shape 专用
    double alpha           = 0.15;   // 半径阈值 (m), 越小越紧贴

    // 输出选项
    bool save_visualization = true;
};

// ── 边界提取器 ──────────────────────────────────────────
class BoundaryExtractor {
public:
    explicit BoundaryExtractor(const BoundaryParams& params = BoundaryParams{});

    // ── 栅格化结果 (extract() 执行后可用, 用于叠加可视化) ──
    struct RasterData {
        cv::Mat occupancy;        // 清理后的二值图 (形态学处理过)
        cv::Mat raw_occupancy;    // 原始二值图 (形态学前)
        float u_min = 0.0f;
        float v_min = 0.0f;
    };
    const RasterData& get_raster_data() const { return raster_data_; }

    /// 执行边界提取
    std::vector<Boundary> extract(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);

    /// 保存可视化图像
    void save_visualization(const std::string& path,
                            const cv::Mat& occupancy,
                            const std::vector<Boundary>& boundaries) const;

private:
    BoundaryParams params_;

    // 步骤1: XY投影 → 2D点集
    std::vector<cv::Point2f> project_to_xy(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) const;

    // 步骤2: 栅格化 → 二值图
    cv::Mat rasterize(const std::vector<cv::Point2f>& pts,
                      float& u_min, float& v_min) const;

    // 步骤3: 形态学
    cv::Mat morphology(const cv::Mat& binary) const;

    // 步骤4a: findContours
    std::vector<Boundary> extract_findcontours(
        const cv::Mat& binary, float u_min, float v_min) const;

    // 步骤4b: Alpha Shape
    std::vector<Boundary> extract_alpha_shape(
        const std::vector<cv::Point2f>& pts) const;

    // 步骤4c: Marching Squares
    std::vector<Boundary> extract_marching_squares(
        const cv::Mat& binary, float u_min, float v_min) const;

    // 坐标反算: 像素 → 世界 (米)
    cv::Point2f pixel_to_world(const cv::Point& px,
                               float u_min, float v_min) const;
    cv::Point world_to_pixel(const cv::Point2f& w,
                             float u_min, float v_min) const;

    RasterData raster_data_;
};
