//=============================================================================
// coverage_planner.h — 船底喷漆作业 Coverage Path Planning
//
// 几何模型:
//   baselink (1.3m X × 0.85m Y)  ←0.616m→  base (max 1.17m × 1.17m)
//   中心点刚性连接: base_center = baselink_center + (0.616, 0)
//   baselink 可在工作区外, base 必须在工作区内
//   小车长边始终与行进方向平行 (随 direction 旋转)
//
// 坐标系: 右手系 (X→右, Y↑上), 俯视点云
//   入口处 cv::flip(wa, wa, 0) 翻转图像, 使图像 Y↑=世界 Y↑
//=============================================================================
#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <string>

// ── 单个作业点 ──────────────────────────────────────────
struct Waypoint {
    int id;
    cv::Point2f baselink_center;  // 小车中心 (世界坐标, m)
    cv::Point2f base_center;      // 喷漆中心 (世界坐标, m)
    float base_x_size;            // base X 方向实际覆盖 (m)
    float base_y_size;            // base Y 方向实际覆盖 (m)
    int stripe;                   // 所属条带 / phase (0=P1边界, 1=P2内部, 2=P3填缝)
    int direction;                // base 行进方向: 0=+X, 1=-X, 2=+Y, 3=-Y
    int region = -1;              // 连通灰色区域标号 (后处理排序时赋值)
};

// ── 规划结果 ────────────────────────────────────────────
struct CoverageResult {
    std::vector<Waypoint> waypoints;
    std::vector<cv::Point2f> path;  // baselink 中心连续路径 (世界坐标)
    float coverage_ratio;           // 覆盖率 [0,1]
    float overlap_ratio;            // 重叠率 [0,1]
    int total_waypoints;
    float total_path_length_m;
};

// ── 规划参数 ────────────────────────────────────────────
struct CoverageParams {
    // 几何参数
    float baselink_x = 1.3f;        // baselink X 方向 (沿桥轴)
    float baselink_y = 0.85f;       // baselink Y 方向 (垂直)
    float base_max   = 1.17f;       // base 最大边长
    float center_offset = 0.616f;   // base_center - baselink_center 沿 X

    // 条带参数
    float stripe_width = 1.17f;     // 条带 Y 宽度 = base_max
    float min_overlap  = 0.05f;     // 相邻 base X 方向最小重叠 (m)

    // 碰撞
    float collision_margin = 0.02f; // 碰撞安全边距 (m)

    // 分辨率
    float pixel_size = 0.01f;       // 与栅格图一致

    // Dubins 参数
    float dubins_turning_radius = 1.5f;  // 最小转弯半径 (m)
    float dubins_sample_dist = 0.1f;     // 路径采样分辨率 (m)

    // DP / 2-opt
    float turn_penalty_weight = 0.4f;    // 转角惩罚权重 (m/rad)
};

// ── 规划器 ──────────────────────────────────────────────
class CoveragePlanner {
public:
    explicit CoveragePlanner(const CoverageParams& params = CoverageParams{});

    /// 执行覆盖规划
    /// @param work_area  工作区域图 (CV_8UC3, 白色=可喷漆, 红色=边界)
    /// @param u_min, v_min  栅格图原点 (世界坐标)
    CoverageResult plan(const cv::Mat& work_area,
                        float u_min, float v_min);

    /// 保存结果可视化 (多图)
    void save_visualization(const std::string& out_dir,
                            const cv::Mat& work_area,
                            const CoverageResult& result,
                            float u_min, float v_min) const;

    /// 导出 JSON
    void save_json(const std::string& path,
                   const CoverageResult& result) const;

private:
    CoverageParams params_;
    int img_rows_ = 0;  // 运行时: work_area.rows

    // 像素 ↔ 世界 (Y 翻转: pixel_y=0 ↔ 世界 Y_max)
    cv::Point world_to_px(const cv::Point2f& w, float u_min, float v_min) const;
    cv::Point2f px_to_world(const cv::Point& px, float u_min, float v_min) const;

    // 碰撞检测: base 矩形是否完全在灰色可喷漆区域内
    // 若 uncovered 非空, 还需检查矩形内像素均未被覆盖
    bool is_base_valid(const cv::Mat& work_area,
                       const cv::Point2f& base_center,
                       float bx, float by,
                       float u_min, float v_min,
                       const cv::Mat* uncovered = nullptr) const;

    // base 自适应扩展: 从初始尺寸向四方向扩展
    // 若 uncovered 非空, 扩展时只允许进入未覆盖的灰色区域
    void expand_base(const cv::Mat& work_area,
                     const cv::Point2f& base_center,
                     float& out_x, float& out_y,
                     float u_min, float v_min,
                     const cv::Mat* uncovered = nullptr) const;

    // 检查某像素是否是桥墩侧障碍 (行进左侧不能有障碍)
    bool check_pier_left(const cv::Mat& work_area,
                         const cv::Point2f& baselink_center,
                         int direction,
                         float u_min, float v_min) const;

    // 检查边界是否在 base 方向的左侧 (check_right=true 时检查右侧)
    bool boundary_on_left(const cv::Mat& work_area,
                          const cv::Point2f& center,
                          float bx, float by, int dir,
                          float u_min, float v_min,
                          bool check_right = false) const;

    // baselink 方向从 base 方向推导
    bool base_dir_to_baselink(const cv::Mat& work_area,
                               const cv::Point2f& base_center,
                               float bx, float by, int base_dir,
                               float u_min, float v_min,
                               cv::Point2f& out_bl) const;

    // 纯几何偏移: base_center + direction → baselink 中心 (不做碰撞检测)
    cv::Point2f baselink_for_direction(const cv::Point2f& base_center,
                                       int dir) const;

    // Dubins 最短路径: 返回采样后的离散路径点
    static std::vector<cv::Point2f> dubins_shortest_path(
        cv::Point2f start_pos, float start_heading,
        cv::Point2f end_pos,   float end_heading,
        float turning_radius, float sample_dist);
};
