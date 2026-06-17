//=============================================================================
// line_fitter.cpp — 桥墩边界直线拟合模块实现
//
// V2 流程:
//   1. 每个边界求 OBB → 提取 4 条边 (上/下/左/右)
//   2. 按类型收集, Y 坐标聚类分行 (区分中间排/两侧排)
//   3. 每行 RANSAC 全局拟合一条直线 → 输出 FittedLine
//=============================================================================
#include "line_fitter.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <map>
#include <set>
#include <iostream>

LineFitter::LineFitter(const LineFitterParams& params) : params_(params) {}

// ═══════════════════════════════════════════════════════════
// 计算单个边界的 OBB → 4 条边
// ═══════════════════════════════════════════════════════════
std::vector<LineFitter::Edge2D> LineFitter::obb_edges(const Boundary& b)
{
    std::vector<Edge2D> edges;
    if (b.points_m.size() < 3) return edges;

    // PCA 求主方向 (与 rect_detector.cpp 的 compute_obb 同样逻辑)
    float cx = 0, cy = 0;
    for (const auto& p : b.points_m) { cx += p.x; cy += p.y; }
    cx /= b.points_m.size();
    cy /= b.points_m.size();

    Eigen::Matrix2f cov = Eigen::Matrix2f::Zero();
    for (const auto& p : b.points_m) {
        float dx = p.x - cx, dy = p.y - cy;
        cov(0,0) += dx*dx; cov(0,1) += dx*dy;
        cov(1,0) += dy*dx; cov(1,1) += dy*dy;
    }
    cov /= b.points_m.size();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eig(cov);
    Eigen::Vector2f major(eig.eigenvectors().col(1).x(),
                           eig.eigenvectors().col(1).y());
    Eigen::Vector2f minor(eig.eigenvectors().col(0).x(),
                           eig.eigenvectors().col(0).y());

    // 投影求半长/半宽
    float min_maj = 1e9, max_maj = -1e9, min_min = 1e9, max_min = -1e9;
    for (const auto& p : b.points_m) {
        float dx = p.x - cx, dy = p.y - cy;
        float pm = dx * major.x() + dy * major.y();
        float pn = dx * minor.x() + dy * minor.y();
        min_maj = std::min(min_maj, pm); max_maj = std::max(max_maj, pm);
        min_min = std::min(min_min, pn); max_min = std::max(max_min, pn);
    }
    float half_l = (max_maj - min_maj) / 2.0f;
    float half_w = (max_min - min_min) / 2.0f;

    // 4 个角点 (顺序: right-top, right-bot, left-bot, left-top 在 major 方向)
    float signs[4][2] = {{1,1},{1,-1},{-1,-1},{-1,1}};
    cv::Point2f corners[4];
    for (int i = 0; i < 4; ++i) {
        corners[i].x = cx + signs[i][0] * half_l * major.x()
                         + signs[i][1] * half_w * minor.x();
        corners[i].y = cy + signs[i][0] * half_l * major.y()
                         + signs[i][1] * half_w * minor.y();
    }

    // 确定边的类型: 比较法线方向
    // edge 0: corners[0]→[1], edge 1: [1]→[2], edge 2: [2]→[3], edge 3: [3]→[0]
    auto classify_edge = [&](const cv::Point2f& p1, const cv::Point2f& p2,
                              const cv::Point2f& center) -> std::string {
        cv::Point2f mid((p1.x + p2.x) * 0.5f, (p1.y + p2.y) * 0.5f);
        cv::Point2f to_mid(mid.x - center.x, mid.y - center.y);
        cv::Point2f dir(p2.x - p1.x, p2.y - p1.y);
        float len = std::sqrt(dir.x*dir.x + dir.y*dir.y);
        if (len < 1e-6) return "unknown";
        cv::Point2f norm(-dir.y / len, dir.x / len); // 顺时针旋转 90°
        // 法线应指向外 (和 to_mid 同向)
        if (norm.x * to_mid.x + norm.y * to_mid.y < 0) {
            norm.x = -norm.x; norm.y = -norm.y;
        }
        // 按法线方向分类
        float abs_nx = std::fabs(norm.x);
        float abs_ny = std::fabs(norm.y);
        if (abs_ny > abs_nx)
            return (norm.y > 0) ? "upper" : "lower";
        else
            return (norm.x > 0) ? "right" : "left";
    };

    cv::Point2f cent(cx, cy);
    std::string types[4] = {
        classify_edge(corners[0], corners[1], cent),
        classify_edge(corners[1], corners[2], cent),
        classify_edge(corners[2], corners[3], cent),
        classify_edge(corners[3], corners[0], cent)
    };

    for (int i = 0; i < 4; ++i) {
        int j = (i + 1) % 4;
        Edge2D e;
        e.p1 = corners[i];
        e.p2 = corners[j];
        e.midpoint = cv::Point2f((e.p1.x + e.p2.x) * 0.5f,
                                  (e.p1.y + e.p2.y) * 0.5f);
        float len = std::sqrt((e.p2.x - e.p1.x) * (e.p2.x - e.p1.x) +
                               (e.p2.y - e.p1.y) * (e.p2.y - e.p1.y));
        if (len > 1e-6) {
            e.direction = cv::Point2f((e.p2.x - e.p1.x) / len,
                                       (e.p2.y - e.p1.y) / len);
            e.normal = cv::Point2f(-e.direction.y, e.direction.x);
        }
        e.boundary_id = b.id;
        e.type = types[i];
        edges.push_back(e);
    }
    return edges;
}

// ═══════════════════════════════════════════════════════════
// 提取所有边界的边
// ═══════════════════════════════════════════════════════════
std::vector<LineFitter::Edge2D> LineFitter::extract_edges(
    const std::vector<Boundary>& boundaries)
{
    std::vector<Edge2D> all_edges;
    for (const auto& b : boundaries) {
        auto edges = obb_edges(b);
        all_edges.insert(all_edges.end(), edges.begin(), edges.end());
    }
    std::cout << "  Extracted " << all_edges.size()
              << " edges from " << boundaries.size() << " boundaries\n";
    return all_edges;
}

// ═══════════════════════════════════════════════════════════
// 按 Y 坐标聚类: 把上下边分成不同的"行"
// ═══════════════════════════════════════════════════════════
static std::vector<std::vector<LineFitter::Edge2D>> cluster_by_y(
    const std::vector<LineFitter::Edge2D>& edges, float y_tolerance)
{
    if (edges.empty()) return {};

    // 按 Y 排序
    auto sorted = edges;
    std::sort(sorted.begin(), sorted.end(),
        [](const auto& a, const auto& b) { return a.midpoint.y < b.midpoint.y; });

    std::vector<std::vector<LineFitter::Edge2D>> clusters;
    std::vector<LineFitter::Edge2D> current;
    float last_y = sorted[0].midpoint.y - y_tolerance - 1;

    for (const auto& e : sorted) {
        if (std::fabs(e.midpoint.y - last_y) > y_tolerance) {
            if (!current.empty()) clusters.push_back(current);
            current.clear();
            last_y = e.midpoint.y;
        }
        current.push_back(e);
    }
    if (!current.empty()) clusters.push_back(current);

    return clusters;
}

// ═══════════════════════════════════════════════════════════
// 分组: 按类型, 然后按 Y 坐标聚类分行
// ═══════════════════════════════════════════════════════════
std::vector<std::vector<LineFitter::Edge2D>> LineFitter::group_edges(
    const std::vector<Edge2D>& edges)
{
    // 按 type 分开
    std::map<std::string, std::vector<Edge2D>> by_type;
    for (const auto& e : edges)
        by_type[e.type].push_back(e);

    std::vector<std::vector<Edge2D>> groups;

    for (const auto& [type, typed_edges] : by_type) {
        if (typed_edges.empty()) continue;

        // upper/lower: 按 Y 坐标聚类分行 (区分不同排的桥墩)
        // left/right:  按 X 坐标聚类分列
        float cluster_tol = params_.group_distance;
        auto clusters = cluster_by_y(typed_edges, cluster_tol);

        for (auto& c : clusters) {
            if (static_cast<int>(c.size()) >= params_.min_piers_per_line)
                groups.push_back(std::move(c));
        }
    }

    // 按组大小排序, 最大的排最前 (中间主排)
    std::sort(groups.begin(), groups.end(),
        [](const auto& a, const auto& b) { return a.size() > b.size(); });

    std::cout << "  Grouped into " << groups.size() << " line groups"
              << " (largest: " << (groups.empty() ? 0 : groups[0].size()) << " edges)\n";
    return groups;
}

// ═══════════════════════════════════════════════════════════
// RANSAC 直线拟合: 从一组边中找出主流方向, 自动剔除 outlier
// ═══════════════════════════════════════════════════════════
FittedLine LineFitter::fit_consensus_line(const std::vector<Edge2D>& group)
{
    FittedLine result;
    if (group.size() < 2) return result;

    // 收集所有边的端点 (每条边贡献 2 个点)
    std::vector<cv::Point2f> pts;
    for (const auto& e : group) {
        pts.push_back(e.p1);
        pts.push_back(e.p2);
    }

    // RANSAC
    const int max_iters = 200;
    const float thresh = params_.line_tolerance;  // 0.15m
    int best_inliers = 0;
    float best_a = 0, best_b = 0, best_c = 0;
    std::vector<bool> best_mask;

    std::srand(42);
    for (int iter = 0; iter < max_iters; ++iter) {
        // 随机选 2 个点定直线
        int i1 = std::rand() % pts.size();
        int i2 = std::rand() % pts.size();
        if (i1 == i2) continue;

        float dx = pts[i2].x - pts[i1].x;
        float dy = pts[i2].y - pts[i1].y;
        float len = std::sqrt(dx*dx + dy*dy);
        if (len < 1e-6f) continue;
        float a = dy / len;
        float b = -dx / len;
        float c = -(a * pts[i1].x + b * pts[i1].y);

        // 数内点
        int inliers = 0;
        std::vector<bool> mask(pts.size(), false);
        for (size_t j = 0; j < pts.size(); ++j) {
            float d = std::fabs(a * pts[j].x + b * pts[j].y + c);
            if (d < thresh) { inliers++; mask[j] = true; }
        }

        if (inliers > best_inliers) {
            best_inliers = inliers;
            best_a = a; best_b = b; best_c = c;
            best_mask = std::move(mask);
        }
    }

    // 用所有内点做最小二乘精修
    float mx = 0, my = 0;
    int refit_n = 0;
    for (size_t j = 0; j < pts.size(); ++j) {
        if (best_mask[j]) { mx += pts[j].x; my += pts[j].y; refit_n++; }
    }
    if (refit_n < 2) return result;
    mx /= refit_n; my /= refit_n;

    float sxx = 0, sxy = 0, syy = 0;
    for (size_t j = 0; j < pts.size(); ++j) {
        if (!best_mask[j]) continue;
        float dx = pts[j].x - mx, dy = pts[j].y - my;
        sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
    }
    Eigen::Matrix2f cov;
    cov << sxx, sxy, sxy, syy;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eig(cov);
    result.a = eig.eigenvectors().col(0).x();
    result.b = eig.eigenvectors().col(0).y();
    result.c = -(result.a * mx + result.b * my);

    // 线段端点: 沿方向投影取 min/max
    cv::Point2f dir(-result.b, result.a);
    float min_proj = 1e9, max_proj = -1e9;
    for (size_t j = 0; j < pts.size(); ++j) {
        if (!best_mask[j]) continue;
        float proj = (pts[j].x - mx) * dir.x + (pts[j].y - my) * dir.y;
        if (proj < min_proj) min_proj = proj;
        if (proj > max_proj) max_proj = proj;
    }
    result.start = {mx + min_proj * dir.x, my + min_proj * dir.y};
    result.end   = {mx + max_proj * dir.x, my + max_proj * dir.y};

    // 记录参与的 boundary_id
    for (const auto& e : group) {
        // 用边中点到拟合直线的距离判断是否参与
        float d = std::fabs(result.a * e.midpoint.x + result.b * e.midpoint.y + result.c);
        if (d < thresh * 2) // 放宽到 2x
            result.boundary_ids.push_back(e.boundary_id);
    }
    std::sort(result.boundary_ids.begin(), result.boundary_ids.end());
    result.boundary_ids.erase(
        std::unique(result.boundary_ids.begin(), result.boundary_ids.end()),
        result.boundary_ids.end());

    result.edge_type = group[0].type;
    result.total_points = static_cast<int>(pts.size());
    result.inlier_ratio = static_cast<float>(best_inliers) / pts.size();

    return result;
}

// ═══════════════════════════════════════════════════════════
// 主入口
// ═══════════════════════════════════════════════════════════
std::vector<FittedLine> LineFitter::fit(const std::vector<Boundary>& boundaries)
{
    std::cout << "\n========== Line Fitting ==========\n";
    std::cout << "Input boundaries: " << boundaries.size() << "\n";

    if (boundaries.size() < 2) {
        std::cout << "  Too few boundaries, skipping.\n";
        return {};
    }

    // 步骤1: 提取每条边界的 OBB 边
    auto edges = extract_edges(boundaries);

    // 步骤2: 分组
    auto groups = group_edges(edges);

    // 步骤3: 每组拟合直线
    std::vector<FittedLine> results;
    for (size_t g = 0; g < groups.size(); ++g) {
        auto line = fit_consensus_line(groups[g]);
        if (line.boundary_ids.size() >= 2) {
            results.push_back(line);
            std::cout << "  Line #" << results.size() - 1
                      << " type=" << line.edge_type
                      << " piers=" << line.boundary_ids.size()
                      << " inlier=" << line.inlier_ratio << "\n";
        }
    }

    std::cout << "Total fitted lines: " << results.size() << "\n";

    // 只保留中间主排的 2 条线 (upper + lower), 其余侧墩不拟合
    if (results.size() > 2) {
        // 按参与墩数排序, 取前 2
        std::sort(results.begin(), results.end(),
            [](const FittedLine& a, const FittedLine& b) {
                return a.boundary_ids.size() > b.boundary_ids.size();
            });
        std::cout << "  Keeping top 2 (middle row), discarding "
                  << (results.size() - 2) << " side pier lines\n";
        results.resize(2);
    }

    return results;
}

// ═══════════════════════════════════════════════════════════
// 两侧桥墩:
//   1) 返回侧墩自身 OBB 四边 (作为边界)
//   2) 用中轴线方向做参考: 垂直中线方向分行, 沿中线方向排序
//   3) 每行内相邻: k.right_mid → k+1.left_mid
//   left/right 按桥轴线方向重新分类 (不依赖 OBB 自身的分类)
// ═══════════════════════════════════════════════════════════
std::vector<FittedLine> LineFitter::connect_neighbors(
    const std::vector<Boundary>& boundaries,
    const std::vector<FittedLine>& middle_lines)
{
    std::vector<FittedLine> results;

    // ── 收集中间主排的 bridge_id ──────────────────────
    std::set<int> middle_ids;
    for (const auto& l : middle_lines)
        for (int id : l.boundary_ids)
            middle_ids.insert(id);

    // ── 桥轴线方向 (从 middle line 取) ────────────────
    cv::Point2f bridge_dir(1.0f, 0.0f);          // 默认 X 方向
    cv::Point2f bridge_perp(0.0f, 1.0f);          // 垂直方向
    if (!middle_lines.empty()) {
        // middle_lines[0] 是 ax+by+c=0, 方向 = (-b, a)
        const auto& ml = middle_lines[0];
        float len = std::sqrt(ml.a*ml.a + ml.b*ml.b);
        if (len > 1e-6f) {
            bridge_dir.x = -ml.b / len;
            bridge_dir.y =  ml.a / len;
            bridge_perp.x = ml.a / len;
            bridge_perp.y = ml.b / len;
        }
    }

    // ── 提取侧墩 OBB 边 ──────────────────────────────
    struct PierData {
        int bid;
        cv::Point2f center;
        std::vector<Edge2D> edges;     // 全部 4 条 OBB 边
        float along;                    // 沿桥轴线投影
        float perp;                     // 垂直桥轴线投影
    };
    std::vector<PierData> piers;

    for (const auto& b : boundaries) {
        if (middle_ids.count(b.id)) continue;

        auto edges = obb_edges(b);
        PierData pd;
        pd.bid = b.id;
        pd.edges = edges;

        // 重心
        cv::Point2f c(0, 0);
        for (const auto& p : b.points_m) { c.x += p.x; c.y += p.y; }
        c.x /= b.points_m.size();
        c.y /= b.points_m.size();
        pd.center = c;

        // 沿桥轴线 + 垂直方向的投影 (以第一个 middle 墩中心为原点)
        pd.along = c.x * bridge_dir.x + c.y * bridge_dir.y;
        pd.perp  = c.x * bridge_perp.x + c.y * bridge_perp.y;

        // OBB 四边作为边界
        for (const auto& e : edges) {
            FittedLine fl;
            fl.edge_type = "obb_" + e.type;
            fl.start = e.p1; fl.end = e.p2;
            fl.boundary_ids = {b.id};
            fl.inlier_ratio = 1.0f;
            results.push_back(fl);
        }

        piers.push_back(pd);
    }

    std::cout << "\n--- Side Piers ---\n";
    std::cout << "  Total bounds: " << boundaries.size()
              << " | Middle: " << middle_ids.size()
              << " | Side: " << piers.size()
              << " | Bridge dir: (" << bridge_dir.x << "," << bridge_dir.y << ")\n";

    if (piers.size() < 2) return results;

    // ── 垂直桥中线方向聚类分行 (gap-based) ────────────
    std::sort(piers.begin(), piers.end(),
        [](const PierData& a, const PierData& b) { return a.perp < b.perp; });

    // 找出 perp 值的间隙, 在间隙处分行
    std::vector<std::vector<size_t>> rows;
    std::vector<size_t> current;
    for (size_t i = 0; i < piers.size(); ++i) {
        if (current.empty()) {
            current.push_back(i);
        } else {
            // 当前墩和前一个墩的 perp 差距
            float gap = piers[i].perp - piers[current.back()].perp;
            if (gap > params_.row_tolerance) {
                rows.push_back(current);
                current.clear();
            }
            current.push_back(i);
        }
    }
    if (!current.empty()) rows.push_back(current);

    // ── 每行内沿桥轴线排序, 连接相邻 ───────────────────
    int connected = 0, skipped_dist = 0, skipped_edge = 0;
    int solo_rows = 0;
    for (const auto& row : rows) {
        if (row.size() < 2) { solo_rows++; continue; }

        // 行内沿桥轴线方向排序
        std::vector<size_t> sorted = row;
        std::sort(sorted.begin(), sorted.end(),
            [&](size_t a, size_t b) {
                return piers[a].along < piers[b].along;
            });

        // 相邻配对: k.right → k+1.left
        for (size_t i = 0; i + 1 < sorted.size(); ++i) {
            PierData& k   = piers[sorted[i]];
            PierData& kp1 = piers[sorted[i + 1]];

            float dx = kp1.center.x - k.center.x;
            float dy = kp1.center.y - k.center.y;
            float d = std::sqrt(dx*dx + dy*dy);

            if (d < params_.connect_min_dist || d > params_.connect_max_dist) {
                skipped_dist++;
                continue;
            }

            // 只选法线沿桥轴线方向的边 (即面对左/右的边, 排除上/下边)
            auto is_along_edge = [&](const Edge2D& e) -> bool {
                float dot_along = std::fabs(e.normal.x * bridge_dir.x
                                          + e.normal.y * bridge_dir.y);
                float dot_perp  = std::fabs(e.normal.x * bridge_perp.x
                                          + e.normal.y * bridge_perp.y);
                return dot_along > dot_perp;  // 法线更接近桥轴线方向
            };

            auto proj_along = [&](const Edge2D& e) {
                return (e.midpoint.x - k.center.x) * bridge_dir.x
                     + (e.midpoint.y - k.center.y) * bridge_dir.y;
            };

            // k 的"右"边: along 投影最大 且 法线沿桥轴线
            int k_right_idx = -1, kp1_left_idx = -1;
            float k_max_proj = -1e9, kp1_min_proj = 1e9;
            for (size_t j = 0; j < k.edges.size();  ++j) {
                if (!is_along_edge(k.edges[j])) continue;
                float proj = proj_along(k.edges[j]);
                if (proj > k_max_proj) { k_max_proj = proj; k_right_idx = static_cast<int>(j); }
            }
            for (size_t j = 0; j < kp1.edges.size(); ++j) {
                if (!is_along_edge(kp1.edges[j])) continue;
                float proj = proj_along(kp1.edges[j]);
                if (proj < kp1_min_proj) { kp1_min_proj = proj; kp1_left_idx = static_cast<int>(j); }
            }

            if (k_right_idx >= 0 && kp1_left_idx >= 0) {
                FittedLine fl;
                fl.edge_type = "connect_R_to_L";
                fl.start = k.edges[k_right_idx].midpoint;
                fl.end   = kp1.edges[kp1_left_idx].midpoint;
                fl.boundary_ids = {k.bid, kp1.bid};
                fl.inlier_ratio = 1.0f;
                results.push_back(fl);
                connected++;
            } else {
                skipped_edge++;
            }
        }
    }

    std::cout << "  Solo rows: " << solo_rows
              << " | Skipped: " << skipped_dist << " (out of dist range), "
              << skipped_edge << " (no along edge found)\n";

    std::cout << "  Rows: " << rows.size()
              << " (tol=" << params_.row_tolerance << "m)"
              << " | Connections: " << connected
              << " | Dist: [" << params_.connect_min_dist
              << ", " << params_.connect_max_dist << "]m\n";
    for (size_t r = 0; r < rows.size(); ++r) {
        float perp_min = piers[rows[r].front()].perp;
        float perp_max = piers[rows[r].back()].perp;
        std::cout << "    Row " << r << ": " << rows[r].size()
                  << " piers, perp [" << perp_min << ", " << perp_max << "]\n";
    }
    return results;
}

// ═══════════════════════════════════════════════════════════
// 可视化
// ═══════════════════════════════════════════════════════════
void LineFitter::save_visualization(
    const std::string& path,
    const cv::Mat& base_image,
    const std::vector<Boundary>& boundaries,
    const std::vector<FittedLine>& lines,
    float u_min, float v_min, float pixel_size) const
{
    cv::Mat vis;
    cv::cvtColor(base_image, vis, cv::COLOR_GRAY2BGR);

    // 世界 → 像素转换
    auto w2p = [&](const cv::Point2f& w) -> cv::Point {
        return cv::Point(
            static_cast<int>((w.x - u_min) / pixel_size),
            static_cast<int>((w.y - v_min) / pixel_size)
        );
    };

    // 先画边界轮廓 (绿色细线)
    for (const auto& b : boundaries) {
        if (b.points_px.size() >= 2) {
            for (size_t i = 0; i < b.points_px.size() - 1; ++i)
                cv::line(vis, b.points_px[i], b.points_px[i+1],
                         cv::Scalar(0, 255, 0), 1);
            if (b.is_closed)
                cv::line(vis, b.points_px.back(), b.points_px.front(),
                         cv::Scalar(0, 255, 0), 1);
        }
    }

    // 画拟合直线 (彩色粗线)
    std::map<std::string, cv::Scalar> colors = {
        {"upper",         cv::Scalar(0, 255, 255)},   // 黄
        {"lower",         cv::Scalar(255, 0, 255)},   // 品红
        {"obb_left",      cv::Scalar(255, 255, 0)},   // 青 (侧墩左边)
        {"obb_right",     cv::Scalar(0, 255, 0)},     // 绿 (侧墩右边)
        {"obb_upper",     cv::Scalar(255, 200, 0)},   // 浅蓝
        {"obb_lower",     cv::Scalar(0, 200, 255)},   // 浅绿
        {"connect_R_to_L",cv::Scalar(255, 128, 0)}    // 橙 (k右→k+1左)
    };

    for (const auto& line : lines) {
        cv::Scalar color = colors.count(line.edge_type)
            ? colors[line.edge_type] : cv::Scalar(255, 255, 255);

        cv::line(vis, w2p(line.start), w2p(line.end), color, 2, cv::LINE_AA);

        // 标注类型文字
        cv::putText(vis, line.edge_type,
                    w2p(line.start) + cv::Point(5, -5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1);
    }

    cv::imwrite(path, vis);
    std::cout << "  Saved combined visualization: " << path << "\n";
}
