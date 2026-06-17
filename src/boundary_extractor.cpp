//=============================================================================
// boundary_extractor.cpp — XY投影边界提取模块实现
//
// 三种边界提取方法:
//   1. findContours      — 对二值图做 OpenCV 连通域边界追踪
//   2. Alpha Shape       — 对投影点做 Delaunay 三角剖分, 剔除大三角形得边界
//   3. Marching Squares  — 遍历二值图 2×2 像素块, 查表插值出等值线
//=============================================================================
#include "boundary_extractor.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <map>
#include <iostream>

// ═══════════════════════════════════════════════════════════
// 构造函数
// ═══════════════════════════════════════════════════════════
BoundaryExtractor::BoundaryExtractor(const BoundaryParams& params)
    : params_(params) {}

// ═══════════════════════════════════════════════════════════
// 坐标转换
// ═══════════════════════════════════════════════════════════
cv::Point2f BoundaryExtractor::pixel_to_world(
    const cv::Point& px, float u_min, float v_min) const
{
    return {
        u_min + px.x * params_.pixel_size,
        v_min + px.y * params_.pixel_size
    };
}

cv::Point BoundaryExtractor::world_to_pixel(
    const cv::Point2f& w, float u_min, float v_min) const
{
    return {
        static_cast<int>((w.x - u_min) / params_.pixel_size),
        static_cast<int>((w.y - v_min) / params_.pixel_size)
    };
}

// ═══════════════════════════════════════════════════════════
// 步骤1: XY 投影
// ═══════════════════════════════════════════════════════════
std::vector<cv::Point2f> BoundaryExtractor::project_to_xy(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) const
{
    std::vector<cv::Point2f> pts;
    pts.reserve(cloud->size());
    for (const auto& p : cloud->points)
        pts.emplace_back(p.x, p.y);
    return pts;
}

// ═══════════════════════════════════════════════════════════
// 步骤2: 栅格化 → 二值 Occupancy Grid
// ═══════════════════════════════════════════════════════════
cv::Mat BoundaryExtractor::rasterize(
    const std::vector<cv::Point2f>& pts,
    float& u_min, float& v_min) const
{
    if (pts.empty()) return {};

    // 计算边界
    float u_max = pts[0].x, v_max = pts[0].y;
    u_min = pts[0].x; v_min = pts[0].y;
    for (const auto& p : pts) {
        u_min = std::min(u_min, p.x); u_max = std::max(u_max, p.x);
        v_min = std::min(v_min, p.y); v_max = std::max(v_max, p.y);
    }

    int cols = static_cast<int>((u_max - u_min) / params_.pixel_size) + 1;
    int rows = static_cast<int>((v_max - v_min) / params_.pixel_size) + 1;

    cols = std::max(1, std::min(cols, 20000));
    rows = std::max(1, std::min(rows, 20000));

    cv::Mat binary = cv::Mat::zeros(rows, cols, CV_8UC1);

    for (const auto& p : pts) {
        int c = static_cast<int>((p.x - u_min) / params_.pixel_size);
        int r = static_cast<int>((p.y - v_min) / params_.pixel_size);
        if (c >= 0 && c < cols && r >= 0 && r < rows)
            binary.at<uint8_t>(r, c) = 255;
    }

    std::cout << "  Rasterized: " << cols << "x" << rows
              << " grid, pixel=" << params_.pixel_size << "m\n";
    return binary;
}

// ═══════════════════════════════════════════════════════════
// 步骤3: 形态学 (闭运算补洞 + 开运算去毛刺)
// ═══════════════════════════════════════════════════════════
cv::Mat BoundaryExtractor::morphology(const cv::Mat& binary) const
{
    cv::Mat result = binary.clone();

    // 闭运算: 膨胀 → 腐蚀 (补小洞)
    if (params_.morph_close_k > 0) {
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(2 * params_.morph_close_k + 1,
                     2 * params_.morph_close_k + 1));
        cv::morphologyEx(result, result, cv::MORPH_CLOSE, kernel);
    }

    // 开运算: 腐蚀 → 膨胀 (去毛刺)
    if (params_.morph_open_k > 0) {
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(2 * params_.morph_open_k + 1,
                     2 * params_.morph_open_k + 1));
        cv::morphologyEx(result, result, cv::MORPH_OPEN, kernel);
    }

    return result;
}

// ═══════════════════════════════════════════════════════════
// 步骤4a: findContours — OpenCV 连通域边界追踪
// ═══════════════════════════════════════════════════════════
std::vector<Boundary> BoundaryExtractor::extract_findcontours(
    const cv::Mat& binary, float u_min, float v_min) const
{
    std::vector<Boundary> results;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(binary.clone(), contours, hierarchy,
                     cv::RETR_EXTERNAL,            // 只取外轮廓
                     cv::CHAIN_APPROX_TC89_L1);    // Teh-Chin 近似

    int id = 0;
    for (const auto& contour : contours) {
        double area_px = cv::contourArea(contour);
        double area_m2 = area_px * params_.pixel_size * params_.pixel_size;

        if (area_m2 < params_.min_contour_area)
            continue;

        double perimeter_px = cv::arcLength(contour, true);
        double perimeter_m = perimeter_px * params_.pixel_size;

        Boundary b;
        b.id = id++;
        b.is_closed = true;
        b.area_m2 = area_m2;
        b.perimeter_m = perimeter_m;
        b.source = "findContours";
        b.points_px = contour;

        for (const auto& px : contour)
            b.points_m.push_back(pixel_to_world(px, u_min, v_min));

        results.push_back(b);
    }

    std::cout << "  findContours: " << results.size() << " boundaries\n";
    return results;
}

// ═══════════════════════════════════════════════════════════
// 步骤4b: Alpha Shape — Delaunay三角剖分 + 半径阈值
//
// 原理:
//   1. 对投影点做 Delaunay 三角剖分
//   2. 计算每个三角形的外接圆半径
//   3. 半径 > alpha → 三角形被"挖掉"
//   4. 仅属于 1 个保留三角形的边 → 边界
// ═══════════════════════════════════════════════════════════

// 三角形外接圆半径
static double circumradius(const cv::Point2f& a,
                           const cv::Point2f& b,
                           const cv::Point2f& c)
{
    double ab = cv::norm(a - b);
    double bc = cv::norm(b - c);
    double ca = cv::norm(c - a);
    if (ab < 1e-9 || bc < 1e-9 || ca < 1e-9) return 1e9;
    // R = abc / (4 * area)
    double s = (ab + bc + ca) / 2.0;
    double area = std::sqrt(std::max(0.0, s * (s - ab) * (s - bc) * (s - ca)));
    if (area < 1e-9) return 1e9;
    return (ab * bc * ca) / (4.0 * area);
}

// 边哈希 (用于统计每条边被几个三角形共享)
struct Edge {
    int a, b;
    Edge(int _a, int _b) : a(std::min(_a, _b)), b(std::max(_a, _b)) {}
    bool operator==(const Edge& o) const { return a == o.a && b == o.b; }
};
struct EdgeHash {
    size_t operator()(const Edge& e) const {
        return std::hash<int>()(e.a) ^ (std::hash<int>()(e.b) << 1);
    }
};

std::vector<Boundary> BoundaryExtractor::extract_alpha_shape(
    const std::vector<cv::Point2f>& pts) const
{
    std::vector<Boundary> results;
    if (pts.size() < 3) return results;

    // 构建 Delaunay 三角剖分
    // 先用全部点构建 Subdiv2D (需要 bbox)

    // === 简化方案: 点太多, Delaunay O(n log n) 但对 800k 点依然很慢 ===
    // 改用网格采样: 每个像素取一个代表点, 大幅减少三角剖分规模
    std::cout << "  Alpha Shape: subsampling points for triangulation...\n";

    // 用栅格化后每个非零像素的中心点作为输入
    // 这样点数量 = 非零像素数, 远小于原始点数
    float u_min, v_min;
    cv::Mat binary = rasterize(pts, u_min, v_min);

    std::vector<cv::Point2f> sampled;
    std::vector<int> original_indices;
    for (int r = 0; r < binary.rows; ++r) {
        for (int c = 0; c < binary.cols; ++c) {
            if (binary.at<uint8_t>(r, c) == 0) continue;
            // 像素中心的世界坐标
            sampled.emplace_back(
                u_min + (c + 0.5f) * params_.pixel_size,
                v_min + (r + 0.5f) * params_.pixel_size
            );
        }
    }

    if (sampled.size() < 3) {
        std::cout << "  Alpha Shape: too few points for triangulation\n";
        return results;
    }

    // 构建 Subdiv2D (需要外围框)
    float bbox_margin = 10.0f;
    cv::Rect2f bbox(u_min - bbox_margin, v_min - bbox_margin,
                    binary.cols * params_.pixel_size + 2 * bbox_margin,
                    binary.rows * params_.pixel_size + 2 * bbox_margin);

    cv::Subdiv2D subdiv(bbox);
    for (const auto& p : sampled)
        subdiv.insert(p);

    // 获取三角形
    std::vector<cv::Vec6f> triangles;
    subdiv.getTriangleList(triangles);

    double alpha_thresh = params_.alpha;

    // 构建点 → 索引映射 (浮点比较, 用网格近似)
    // 用像素坐标做 key
    auto point_key = [this, u_min, v_min](const cv::Point2f& p) -> std::pair<int,int> {
        return {
            static_cast<int>((p.x - u_min) / params_.pixel_size + 0.5f),
            static_cast<int>((p.y - v_min) / params_.pixel_size + 0.5f)
        };
    };

    // 对每个三角形三个顶点, 找到在 sampled 中的索引
    // 用 grid hash map 加速匹配
    std::map<std::pair<int,int>, int> key_to_idx;
    for (size_t i = 0; i < sampled.size(); ++i)
        key_to_idx[point_key(sampled[i])] = static_cast<int>(i);

    std::unordered_map<Edge, int, EdgeHash> edge_count;

    int kept_tris = 0;
    for (const auto& tri : triangles) {
        cv::Point2f a(tri[0], tri[1]);
        cv::Point2f b(tri[2], tri[3]);
        cv::Point2f c(tri[4], tri[5]);

        // 跳过 bbox 外围的三角形
        auto it_a = key_to_idx.find(point_key(a));
        auto it_b = key_to_idx.find(point_key(b));
        auto it_c = key_to_idx.find(point_key(c));
        if (it_a == key_to_idx.end() || it_b == key_to_idx.end() ||
            it_c == key_to_idx.end())
            continue;

        double r = circumradius(a, b, c);
        if (r > alpha_thresh) continue; // 三角形太大 → 丢弃

        kept_tris++;
        int ia = it_a->second, ib = it_b->second, ic = it_c->second;
        edge_count[Edge(ia, ib)]++;
        edge_count[Edge(ib, ic)]++;
        edge_count[Edge(ic, ia)]++;
    }

    std::cout << "  Alpha Shape: " << sampled.size() << " sampled pts, "
              << triangles.size() << " Delaunay tris, "
              << kept_tris << " kept (r <= " << alpha_thresh << "m)\n";

    // 提取 count == 1 的边 (仅属于 1 个保留三角形 → 边界)
    std::vector<Edge> boundary_edges;
    for (const auto& [edge, count] : edge_count)
        if (count == 1)
            boundary_edges.push_back(edge);

    std::cout << "  Alpha Shape: " << boundary_edges.size() << " boundary edges\n";

    if (boundary_edges.empty()) return results;

    // 把边串成闭合轮廓 (贪心连接)
    std::unordered_map<int, std::vector<int>> adjacency;
    for (const auto& e : boundary_edges) {
        adjacency[e.a].push_back(e.b);
        adjacency[e.b].push_back(e.a);
    }

    std::unordered_set<int> visited;
    int id = 0;
    for (const auto& [start, _] : adjacency) {
        if (visited.count(start)) continue;

        std::vector<cv::Point2f> chain;
        int cur = start;
        int prev = -1;

        do {
            visited.insert(cur);
            chain.push_back(sampled[cur]);

            const auto& neighbors = adjacency[cur];
            int next = -1;
            for (int n : neighbors) {
                if (n != prev && !visited.count(n)) {
                    next = n; break;
                }
            }
            // 没有未访问邻居但 chain 可以闭合回起点
            if (next == -1 && prev != -1) {
                for (int n : neighbors) {
                    if (n == start && chain.size() > 2) { next = n; break; }
                }
            }
            prev = cur;
            cur = next;
        } while (cur != -1 && cur != start);

        // 闭合
        if (cur == start && chain.size() > 2) {
            chain.push_back(sampled[start]); // 闭合

            // 计算面积
            double area = 0.0;
            for (size_t i = 0; i < chain.size() - 1; ++i)
                area += chain[i].x * chain[i+1].y - chain[i+1].x * chain[i].y;
            area = std::abs(area) / 2.0;

            if (area >= params_.min_contour_area) {
                Boundary b;
                b.id = id++;
                b.is_closed = true;
                b.area_m2 = area;
                b.perimeter_m = 0; // 粗略估算
                b.source = "alpha_shape";
                b.points_m = chain;
                // 反算像素坐标
                for (const auto& p : chain)
                    b.points_px.push_back(world_to_pixel(p, u_min, v_min));
                results.push_back(b);
            }
        }
    }

    std::cout << "  Alpha Shape: " << results.size() << " boundaries\n";
    return results;
}

// ═══════════════════════════════════════════════════════════
// 步骤4c: Marching Squares — 2×2 像素块等值线提取
//
// 每个 2×2 像素块有 16 种状态 (4 个角, 每个角 ∈ {0,1})
// 查表输出对应线段
// ═══════════════════════════════════════════════════════════

// 每条线段: 从 (x1,y1) 到 (x2,y2) 的像素坐标
struct SegmentMS {
    cv::Point2f a, b;
};

// 16 种状态的查表 (每条线段的中点插值)
// 四个角编号: 0=左上, 1=右上, 2=右下, 3=左下
// 值为 1 → 255 (物体内部)
static const SegmentMS MS_TABLE[16][2] = {
    /* 0000 */ {{}, {}},                                                          // 全空
    /* 0001 */ {{{0.5f,0.0f}, {0.0f,0.5f}}, {}},                                 // 左下
    /* 0010 */ {{{1.0f,0.5f}, {0.5f,0.0f}}, {}},                                 // 右下
    /* 0011 */ {{{1.0f,0.5f}, {0.0f,0.5f}}, {}},                                 // 底部
    /* 0100 */ {{{0.5f,1.0f}, {1.0f,0.5f}}, {}},                                 // 右上
    /* 0101 */ {{{0.5f,1.0f}, {1.0f,0.5f}}, {{0.5f,0.0f}, {0.0f,0.5f}}},        // 对顶(左上+右下)
    /* 0110 */ {{{0.5f,1.0f}, {0.5f,0.0f}}, {}},                                 // 右侧
    /* 0111 */ {{{0.5f,1.0f}, {0.0f,0.5f}}, {}},                                 // 右上+右下+左下
    /* 1000 */ {{{0.0f,0.5f}, {0.5f,1.0f}}, {}},                                 // 左上
    /* 1001 */ {{{0.0f,0.5f}, {0.5f,0.0f}}, {}},                                 // 左侧
    /* 1010 */ {{{0.0f,0.5f}, {0.5f,1.0f}}, {{1.0f,0.5f}, {0.5f,0.0f}}},        // 对顶(右上+左下)
    /* 1011 */ {{{1.0f,0.5f}, {0.5f,1.0f}}, {}},                                 // 左上+左下+右下
    /* 1100 */ {{{0.0f,0.5f}, {1.0f,0.5f}}, {}},                                 // 顶部
    /* 1101 */ {{{0.5f,0.0f}, {1.0f,0.5f}}, {}},                                 // 左上+右上+左下
    /* 1110 */ {{{0.5f,1.0f}, {0.5f,0.0f}}, {}},                                 // 左上+右上+右下
    /* 1111 */ {{}, {}},                                                          // 全满
};

std::vector<Boundary> BoundaryExtractor::extract_marching_squares(
    const cv::Mat& binary, float u_min, float v_min) const
{
    std::vector<Boundary> results;
    if (binary.rows < 2 || binary.cols < 2) return results;

    std::cout << "  Marching Squares: scanning " << binary.cols << "x"
              << binary.rows << " grid...\n";

    // 收集所有线段
    struct Segment {
        cv::Point2f a, b;
        bool used = false;
    };
    std::vector<Segment> segments;

    for (int r = 0; r < binary.rows - 1; ++r) {
        for (int c = 0; c < binary.cols - 1; ++c) {
            // 四个角的二值状态
            int tl = (binary.at<uint8_t>(r,   c  ) > 127) ? 1 : 0;
            int tr = (binary.at<uint8_t>(r,   c+1) > 127) ? 1 : 0;
            int br = (binary.at<uint8_t>(r+1, c+1) > 127) ? 1 : 0;
            int bl = (binary.at<uint8_t>(r+1, c  ) > 127) ? 1 : 0;
            int idx = (tl << 3) | (tr << 2) | (br << 1) | bl;

            for (int k = 0; k < 2; ++k) {
                const auto& seg = MS_TABLE[idx][k];
                if (seg.a == seg.b) continue; // 空槽
                // 转为该 cell 内的像素坐标
                Segment s;
                s.a = cv::Point2f(c + seg.a.x, r + seg.a.y);
                s.b = cv::Point2f(c + seg.b.x, r + seg.b.y);
                segments.push_back(s);
            }
        }
    }

    std::cout << "  Marching Squares: " << segments.size() << " segments\n";
    if (segments.empty()) return results;

    // 把线段串成闭合轮廓
    std::unordered_map<int, std::vector<int>> graph;
    auto pkey = [](const cv::Point2f& p) -> int64_t {
        // 像素坐标量化到 0.5 精度 → hash
        int ix = static_cast<int>(p.x * 2.0f + 0.5f);
        int iy = static_cast<int>(p.y * 2.0f + 0.5f);
        return (static_cast<int64_t>(ix) << 32) | static_cast<int64_t>(iy & 0xFFFFFFFF);
    };

    std::unordered_map<int64_t, int> vertex_to_id;
    std::vector<cv::Point2f> vertices;

    auto get_or_create_vid = [&](const cv::Point2f& v) -> int {
        int64_t key = pkey(v);
        auto it = vertex_to_id.find(key);
        if (it != vertex_to_id.end()) return it->second;
        int id = static_cast<int>(vertices.size());
        vertices.push_back(v);
        vertex_to_id[key] = id;
        return id;
    };

    for (const auto& seg : segments) {
        int a_id = get_or_create_vid(seg.a);
        int b_id = get_or_create_vid(seg.b);
        graph[a_id].push_back(b_id);
        graph[b_id].push_back(a_id);
    }

    // 贪心追踪轮廓
    std::unordered_set<int> visited;
    int id = 0;
    for (const auto& [vid, _] : graph) {
        if (visited.count(vid)) continue;

        std::vector<cv::Point2f> chain;
        int cur = vid, prev = -1;

        do {
            visited.insert(cur);
            chain.push_back(vertices[cur]);

            const auto& neighbors = graph[cur];
            int next = -1;
            // 优先找未访问的邻居
            for (int n : neighbors) {
                if (n != prev) {
                    if (!visited.count(n)) { next = n; break; }
                    if (n == vid && chain.size() > 2) { next = n; break; }
                }
            }
            if (next == -1) break;
            prev = cur;
            cur = next;
        } while (cur != -1 && cur != vid);

        if (cur == vid && chain.size() > 2) {
            chain.push_back(vertices[vid]); // 闭合

            // 计算面积
            double area = 0.0;
            for (size_t i = 0; i < chain.size() - 1; ++i)
                area += chain[i].x * chain[i+1].y - chain[i+1].x * chain[i].y;
            area = std::abs(area) / 2.0;

            double area_m2 = area * params_.pixel_size * params_.pixel_size;
            if (area_m2 >= params_.min_contour_area) {
                Boundary b;
                b.id = id++;
                b.is_closed = true;
                b.area_m2 = area_m2;
                b.source = "marching_squares";
                // 转为世界坐标
                for (const auto& pt : chain) {
                    b.points_m.push_back(pixel_to_world(
                        cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)),
                        u_min, v_min));
                    b.points_px.push_back(cv::Point(
                        static_cast<int>(pt.x), static_cast<int>(pt.y)));
                }
                results.push_back(b);
            }
        }
    }

    std::cout << "  Marching Squares: " << results.size() << " boundaries\n";
    return results;
}

// ═══════════════════════════════════════════════════════════
// 主入口: 执行完整边界提取流水线
// ═══════════════════════════════════════════════════════════
std::vector<Boundary> BoundaryExtractor::extract(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud)
{
    std::vector<Boundary> results;
    if (!cloud || cloud->empty()) return results;

    std::cout << "\n========== Boundary Extraction ==========\n";
    std::cout << "Input points: " << cloud->size() << "\n";
    std::cout << "Method: "
              << (params_.method == BoundaryMethod::FINDCONTOURS    ? "findContours" :
                  params_.method == BoundaryMethod::ALPHA_SHAPE     ? "Alpha Shape" :
                  params_.method == BoundaryMethod::MARCHING_SQUARES ? "Marching Squares" :
                  "Unknown")
              << "\n";

    // 步骤1: XY投影
    auto pts_2d = project_to_xy(cloud);

    // 步骤2: 栅格化
    float u_min, v_min;
    cv::Mat binary = rasterize(pts_2d, u_min, v_min);

    // 步骤3: 形态学
    cv::Mat cleaned = morphology(binary);

    // 存储栅格化结果 (供 line_fitter 叠加可视化)
    raster_data_.raw_occupancy = binary;
    raster_data_.occupancy     = cleaned;
    raster_data_.u_min         = u_min;
    raster_data_.v_min         = v_min;

    // 步骤4: 边界提取 (按 method 分发)
    switch (params_.method) {
    case BoundaryMethod::FINDCONTOURS:
        results = extract_findcontours(cleaned, u_min, v_min);
        break;
    case BoundaryMethod::ALPHA_SHAPE:
        results = extract_alpha_shape(pts_2d);
        break;
    case BoundaryMethod::MARCHING_SQUARES:
        results = extract_marching_squares(cleaned, u_min, v_min);
        break;
    }

    // 输出统计
    std::cout << "\n--- Boundary Results ---\n";
    std::cout << "Total boundaries found: " << results.size() << "\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& b = results[i];
        std::cout << "  #" << b.id
                  << " area=" << b.area_m2 << "m²"
                  << " perim=" << b.perimeter_m << "m"
                  << " pts=" << b.points_m.size()
                  << " src=" << b.source << "\n";
    }

    // 保存可视化
    if (params_.save_visualization)
        save_visualization("output/boundaries.png", cleaned, results);

    return results;
}

// ═══════════════════════════════════════════════════════════
// 可视化: 在二值图上绘制边界
// ═══════════════════════════════════════════════════════════
void BoundaryExtractor::save_visualization(
    const std::string& path,
    const cv::Mat& occupancy,
    const std::vector<Boundary>& boundaries) const
{
    cv::Mat vis;
    cv::cvtColor(occupancy, vis, cv::COLOR_GRAY2BGR);

    // 每种方法不同颜色
    std::map<std::string, cv::Scalar> colors = {
        {"findContours",      cv::Scalar(0, 255, 0)},   // 绿
        {"alpha_shape",       cv::Scalar(255, 0, 0)},   // 蓝
        {"marching_squares",  cv::Scalar(0, 0, 255)}    // 红
    };

    for (const auto& b : boundaries) {
        cv::Scalar color = colors.count(b.source) ? colors[b.source]
                                                   : cv::Scalar(255, 255, 0);
        if (b.points_px.size() >= 2) {
            for (size_t i = 0; i < b.points_px.size() - 1; ++i)
                cv::line(vis, b.points_px[i], b.points_px[i+1], color, 2, cv::LINE_AA);
            if (b.is_closed)
                cv::line(vis, b.points_px.back(), b.points_px.front(), color, 2, cv::LINE_AA);
        }
    }

    cv::imwrite(path, vis);
    std::cout << "  Saved visualization: " << path << "\n";
}
