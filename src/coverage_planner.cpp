//=============================================================================
// coverage_planner.cpp — Two-Phase Coverage Path Planning V4
//
// Phase 1: 条带行放置 — 沿自适应行逐行放置 base, expand 填满可用空间
// Phase 2: 残区补全 — 扫描剩余未覆盖区域, 贪心放置
//=============================================================================
#include "coverage_planner.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <set>
#include <queue>

CoveragePlanner::CoveragePlanner(const CoverageParams& params)
    : params_(params) {}

// ── 坐标转换 ────────────────────────────────────────────
cv::Point CoveragePlanner::world_to_px(const cv::Point2f& w,
                                        float u_min, float v_min) const
{
    return cv::Point(
        static_cast<int>((w.x - u_min) / params_.pixel_size + 0.5f),
        static_cast<int>((w.y - v_min) / params_.pixel_size + 0.5f));
}

cv::Point2f CoveragePlanner::px_to_world(const cv::Point& px,
                                          float u_min, float v_min) const
{
    return {u_min + px.x * params_.pixel_size,
            v_min + px.y * params_.pixel_size};
}

// ── base 碰撞检测 ──────────────────────────────────────
bool CoveragePlanner::is_base_valid(const cv::Mat& work_area,
                                     const cv::Point2f& base_center,
                                     float bx, float by,
                                     float u_min, float v_min,
                                     const cv::Mat* uncovered) const
{
    float hx = bx / 2.0f;
    float hy = by / 2.0f;
    cv::Point tl = world_to_px({base_center.x - hx, base_center.y - hy}, u_min, v_min);
    cv::Point br = world_to_px({base_center.x + hx, base_center.y + hy}, u_min, v_min);

    if (tl.x < 0 || tl.y < 0 || br.x >= work_area.cols || br.y >= work_area.rows)
        return false;

    for (int r = tl.y; r <= br.y; ++r) {
        const cv::Vec3b* row = work_area.ptr<cv::Vec3b>(r);
        const uint8_t* urow = uncovered ? uncovered->ptr<uint8_t>(r) : nullptr;
        for (int c = tl.x; c <= br.x; ++c) {
            if (row[c] != cv::Vec3b(128, 128, 128))  // 必须纯灰
                return false;
            if (urow && urow[c] == 0)                 // 不能与已覆盖区域重叠
                return false;
        }
    }
    return true;
}

// ── base 自适应扩展 ────────────────────────────────────
void CoveragePlanner::expand_base(const cv::Mat& work_area,
                                   const cv::Point2f& base_center,
                                   float& out_x, float& out_y,
                                   float u_min, float v_min,
                                   const cv::Mat* uncovered) const
{
    float step = 0.05f;
    float max_dim = params_.base_max;
    bool changed = true;
    while (changed) {
        changed = false;
        if (out_x + step * 2 <= max_dim &&
            is_base_valid(work_area, base_center, out_x + step * 2, out_y, u_min, v_min, uncovered)) {
            out_x += step * 2; changed = true;
        }
        if (out_y + step * 2 <= max_dim &&
            is_base_valid(work_area, base_center, out_x, out_y + step * 2, u_min, v_min, uncovered)) {
            out_y += step * 2; changed = true;
        }
    }
}

// ── 桥墩在左约束 ──────────────────────────────────────
bool CoveragePlanner::check_pier_left(const cv::Mat& work_area,
                                       const cv::Point2f& baselink_center,
                                       int direction,
                                       float u_min, float v_min) const
{
    float y_off = (direction > 0) ? 0.5f : -0.5f;
    cv::Point pt = world_to_px({baselink_center.x, baselink_center.y + y_off}, u_min, v_min);

    if (pt.x < 0 || pt.x >= work_area.cols || pt.y < 0 || pt.y >= work_area.rows)
        return false;
    cv::Vec3b p = work_area.at<cv::Vec3b>(pt.y, pt.x);
    return !(p[0] == 0 && p[1] == 0 && p[2] == 0);
}

// ── 统计 base 覆盖的未覆盖灰色像素数 ──────────────────
static int count_uncovered(const cv::Mat& uncovered,
                           const cv::Point& tl, const cv::Point& br)
{
    int cnt = 0;
    for (int r = tl.y; r <= br.y; ++r) {
        const uint8_t* row = uncovered.ptr<uint8_t>(r);
        for (int c = tl.x; c <= br.x; ++c)
            if (row[c] > 0) cnt++;
    }
    return cnt;
}

// ═══════════════════════════════════════════════════════════
// 主入口
// ═══════════════════════════════════════════════════════════
CoverageResult CoveragePlanner::plan(const cv::Mat& work_area,
                                      float u_min, float v_min)
{
    CoverageResult result;
    if (work_area.empty()) return result;

    std::cout << "\n========== Coverage Planning V4 ==========\n";
    int rows = work_area.rows, cols = work_area.cols;

    // ── A1. 构建 mask ────────────────────────────────
    cv::Mat uncovered(rows, cols, CV_8UC1, cv::Scalar(0));
    cv::Mat gray_mask(rows, cols, CV_8UC1, cv::Scalar(0));
    cv::Mat covered(rows, cols, CV_8UC1, cv::Scalar(0));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (work_area.at<cv::Vec3b>(r, c) == cv::Vec3b(128, 128, 128)) {
                uncovered.at<uint8_t>(r, c) = 255;
                gray_mask.at<uint8_t>(r, c) = 255;
            }
        }
    }

    int total_gray = cv::countNonZero(gray_mask);
    std::cout << "Gray pixels: " << total_gray << "\n";
    if (total_gray == 0) return result;

    // ── A2. 滑动窗口网格 ────────────────────────────
    // 找灰色区域范围
    int y_min_px = rows, y_max_px = 0;
    int x_min_px = cols, x_max_px = 0;
    for (int r = 0; r < rows; ++r) {
        const uint8_t* gr = gray_mask.ptr<uint8_t>(r);
        for (int c = 0; c < cols; ++c) {
            if (gr[c] > 0) {
                y_min_px = std::min(y_min_px, r); y_max_px = std::max(y_max_px, r);
                x_min_px = std::min(x_min_px, c); x_max_px = std::max(x_max_px, c);
            }
        }
    }

    // 网格步长: X=1.4m, Y=1.17m
    float step_x = 1.4f;
    float step_y = params_.base_max;  // 1.17m
    int step_x_px = static_cast<int>(step_x / params_.pixel_size);
    int step_y_px = static_cast<int>(step_y / params_.pixel_size);

    // 起始位置: 第一个能完整放下 base_max 的位置
    float start_x = u_min + (x_min_px + params_.base_max / 2.0f / params_.pixel_size) * params_.pixel_size;
    float start_y = v_min + (y_min_px + params_.base_max / 2.0f / params_.pixel_size) * params_.pixel_size;

    std::cout << "Sliding window: step_x=" << step_x << "m step_y=" << step_y
              << "m grid=" << (x_max_px - x_min_px) / step_x_px << "x"
              << (y_max_px - y_min_px) / step_y_px << "\n";

    // ══════════════════════════════════════════════════════
    // Phase 1: 滑动窗口 (左→右, 上→下)
    // ══════════════════════════════════════════════════════
    std::vector<Waypoint> waypoints;
    int wp_id = 0;

    for (float wy = start_y; ; wy += step_y) {
        int gy = static_cast<int>((wy - v_min) / params_.pixel_size);
        if (gy > y_max_px) break;

        for (float wx = start_x; ; wx += step_x) {
            int gx = static_cast<int>((wx - u_min) / params_.pixel_size);
            if (gx > x_max_px) break;

            // 跳过显然不在灰色区域内的位置
            if (gx < 0 || gx >= cols || gy < 0 || gy >= rows) continue;
            if (gray_mask.at<uint8_t>(gy, gx) == 0) continue;

            // ── 尝试放置: 优先 1.17×1.17, 受阻则调整中心, 再缩小 ──
            float bx = params_.base_max, by = params_.base_max;
            cv::Point2f best_bc(wx, wy);
            bool placed = false;

            // 1) 原始中心 + 原始尺寸
            if (is_base_valid(work_area, best_bc, bx, by, u_min, v_min, &uncovered)) {
                placed = true;
            }

            // 2) 尝试移动中心 (±0.3m 范围, 5cm 步长)
            if (!placed) {
                for (float dy = -0.30f; dy <= 0.30f && !placed; dy += 0.05f) {
                    for (float dx = -0.30f; dx <= 0.30f && !placed; dx += 0.05f) {
                        if (dx == 0 && dy == 0) continue;
                        cv::Point2f shifted(wx + dx, wy + dy);
                        if (is_base_valid(work_area, shifted, bx, by, u_min, v_min, &uncovered)) {
                            best_bc = shifted; placed = true;
                        }
                    }
                }
            }

            // 3) 缩小窗口 (保持中心, 逐步缩小到 0.2m)
            if (!placed) {
                for (float scale = 0.95f; scale >= 0.15f; scale -= 0.05f) {
                    float sx = params_.base_max * scale;
                    float sy = params_.base_max * scale;
                    // 先试原始中心
                    if (is_base_valid(work_area, best_bc, sx, sy, u_min, v_min, &uncovered)) {
                        bx = sx; by = sy; placed = true; break;
                    }
                    // 再试移动中心
                    for (float dy = -0.20f; dy <= 0.20f && !placed; dy += 0.10f) {
                        for (float dx = -0.20f; dx <= 0.20f && !placed; dx += 0.10f) {
                            cv::Point2f shifted(wx + dx, wy + dy);
                            if (is_base_valid(work_area, shifted, sx, sy, u_min, v_min, &uncovered)) {
                                best_bc = shifted; bx = sx; by = sy; placed = true;
                            }
                        }
                    }
                    if (placed) break;
                }
            }

            if (!placed) continue;  // 无法放置
            if (bx < 0.15f || by < 0.15f) continue;

            // ── baselink 方向 ────────────────────────
            cv::Point bc_px = world_to_px(best_bc, u_min, v_min);
            bool near_boundary = false;
            int dist_px = static_cast<int>(0.6f / params_.pixel_size);
            for (int rr = std::max(0, bc_px.y - dist_px);
                 rr <= std::min(rows - 1, bc_px.y + dist_px) && !near_boundary; ++rr) {
                const cv::Vec3b* wr = work_area.ptr<cv::Vec3b>(rr);
                for (int cc = std::max(0, bc_px.x - dist_px);
                     cc <= std::min(cols - 1, bc_px.x + dist_px) && !near_boundary; ++cc)
                    if (wr[cc] == cv::Vec3b(0, 0, 255)) near_boundary = true;
            }
            int best_dir = 0;
            cv::Point2f best_bl;
            for (int d : {+1, -1}) {
                cv::Point2f bl_ctr(best_bc.x - params_.center_offset * d, best_bc.y);
                bool bl_red = false;
                float bl_hx = params_.baselink_x / 2, bl_hy = params_.baselink_y / 2;
                cv::Point tl = world_to_px({bl_ctr.x - bl_hx, bl_ctr.y - bl_hy}, u_min, v_min);
                cv::Point br = world_to_px({bl_ctr.x + bl_hx, bl_ctr.y + bl_hy}, u_min, v_min);
                for (int rr = std::max(0, tl.y); rr <= std::min(rows - 1, br.y) && !bl_red; ++rr)
                    for (int cc = std::max(0, tl.x); cc <= std::min(cols - 1, br.x) && !bl_red; ++cc)
                        if (work_area.at<cv::Vec3b>(rr, cc) == cv::Vec3b(0, 0, 255)) bl_red = true;
                if (bl_red) continue;
                if (!near_boundary || check_pier_left(work_area, bl_ctr, d, u_min, v_min)) {
                    best_dir = d; best_bl = bl_ctr; break;
                }
            }
            if (best_dir == 0) continue;

            Waypoint wp;
            wp.id = wp_id++;
            wp.base_center = best_bc;
            wp.baselink_center = best_bl;
            wp.base_x_size = bx;
            wp.base_y_size = by;
            wp.stripe = 0;
            wp.direction = best_dir;
            waypoints.push_back(wp);

            // 标记已覆盖
            int hx = static_cast<int>(bx / 2 / params_.pixel_size);
            int hy = static_cast<int>(by / 2 / params_.pixel_size);
            cv::Point placed_px = world_to_px(best_bc, u_min, v_min);
            cv::Rect roi(placed_px.x - hx, placed_px.y - hy, hx * 2, hy * 2);
            roi &= cv::Rect(0, 0, cols, rows);
            for (int rr = roi.y; rr < roi.y + roi.height; ++rr)
                for (int cc = roi.x; cc < roi.x + roi.width; ++cc)
                    if (work_area.at<cv::Vec3b>(rr, cc) == cv::Vec3b(128, 128, 128))
                        { uncovered.at<uint8_t>(rr, cc) = 0; covered.at<uint8_t>(rr, cc) = 255; }
        }
    }

    int phase1_wp = wp_id;
    std::cout << "Phase 1: " << phase1_wp << " waypoints\n";

    // ══════════════════════════════════════════════════════
    // Phase 2: 残区补全
    // ══════════════════════════════════════════════════════
    int phase2_added = 0;
    int uncovered_remaining = cv::countNonZero(uncovered);
    if (uncovered_remaining > 0)
        std::cout << "Phase 2: " << uncovered_remaining << " uncovered pixels remain\n";

    int pass2_iter = 0;
    const int PASS2_MAX_ITERS = 100;
    while (uncovered_remaining > 0 && pass2_iter < PASS2_MAX_ITERS) {
        pass2_iter++;

        int u_min_x = cols, u_max_x = 0, u_min_y = rows, u_max_y = 0;
        for (int r = 0; r < rows; ++r) {
            const uint8_t* urow = uncovered.ptr<uint8_t>(r);
            for (int c = 0; c < cols; ++c) {
                if (urow[c] > 0) {
                    u_min_x = std::min(u_min_x, c); u_max_x = std::max(u_max_x, c);
                    u_min_y = std::min(u_min_y, r); u_max_y = std::max(u_max_y, r);
                }
            }
        }
        if (u_min_x > u_max_x) break;

        int bb_w = u_max_x - u_min_x + 1;
        int bb_h = u_max_y - u_min_y + 1;
        int scan_step = std::max(3, static_cast<int>(
            std::sqrt(static_cast<double>(bb_w) * bb_h / 5000.0)));

        int best_x = -1, best_y = -1, best_cnt = 0;
        float best_bx = 0, best_by = 0;
        for (int sy = u_min_y; sy <= u_max_y; sy += scan_step) {
            const uint8_t* urow = uncovered.ptr<uint8_t>(sy);
            for (int sx = u_min_x; sx <= u_max_x; sx += scan_step) {
                if (urow[sx] == 0) continue;
                cv::Point2f bc2 = px_to_world({sx, sy}, u_min, v_min);
                float bx2 = 0.2f, by2 = 0.2f;
                expand_base(work_area, bc2, bx2, by2, u_min, v_min, &uncovered);
                int hx = static_cast<int>(bx2 / 2 / params_.pixel_size);
                int hy = static_cast<int>(by2 / 2 / params_.pixel_size);
                cv::Point tl(std::max(0, sx - hx), std::max(0, sy - hy));
                cv::Point br(std::min(cols - 1, sx + hx), std::min(rows - 1, sy + hy));
                int cnt = count_uncovered(uncovered, tl, br);
                if (cnt > best_cnt) { best_cnt = cnt; best_x = sx; best_y = sy; best_bx = bx2; best_by = by2; }
            }
        }
        if (best_cnt < 1) break;

        cv::Point2f bc2 = px_to_world({best_x, best_y}, u_min, v_min);
        cv::Point bc_px2 = world_to_px(bc2, u_min, v_min);
        bool near_boundary2 = false;
        int dist_px2 = static_cast<int>(0.6f / params_.pixel_size);
        for (int rr = std::max(0, bc_px2.y - dist_px2);
             rr <= std::min(rows - 1, bc_px2.y + dist_px2) && !near_boundary2; ++rr) {
            const cv::Vec3b* wr = work_area.ptr<cv::Vec3b>(rr);
            for (int cc = std::max(0, bc_px2.x - dist_px2);
                 cc <= std::min(cols - 1, bc_px2.x + dist_px2) && !near_boundary2; ++cc)
                if (wr[cc] == cv::Vec3b(0, 0, 255)) near_boundary2 = true;
        }
        int best_dir2 = 0;
        cv::Point2f best_bl2;
        for (int d : {+1, -1}) {
            cv::Point2f bl_ctr(bc2.x - params_.center_offset * d, bc2.y);
            bool bl_red = false;
            float bl_hx = params_.baselink_x / 2, bl_hy = params_.baselink_y / 2;
            cv::Point tl = world_to_px({bl_ctr.x - bl_hx, bl_ctr.y - bl_hy}, u_min, v_min);
            cv::Point br = world_to_px({bl_ctr.x + bl_hx, bl_ctr.y + bl_hy}, u_min, v_min);
            for (int rr = std::max(0, tl.y); rr <= std::min(rows - 1, br.y) && !bl_red; ++rr)
                for (int cc = std::max(0, tl.x); cc <= std::min(cols - 1, br.x) && !bl_red; ++cc)
                    if (work_area.at<cv::Vec3b>(rr, cc) == cv::Vec3b(0, 0, 255)) bl_red = true;
            if (bl_red) continue;
            if (!near_boundary2 || check_pier_left(work_area, bl_ctr, d, u_min, v_min)) {
                best_dir2 = d; best_bl2 = bl_ctr; break;
            }
        }
        if (best_dir2 == 0) { uncovered.at<uint8_t>(best_y, best_x) = 0; continue; }

        Waypoint wp;
        wp.id = wp_id++;
        wp.base_center = bc2;
        wp.baselink_center = best_bl2;
        wp.base_x_size = best_bx; wp.base_y_size = best_by;
        wp.stripe = 99; wp.direction = best_dir2;
        waypoints.push_back(wp);
        phase2_added++;

        int hx = static_cast<int>(best_bx / 2 / params_.pixel_size);
        int hy = static_cast<int>(best_by / 2 / params_.pixel_size);
        cv::Rect roi(best_x - hx, best_y - hy, hx * 2, hy * 2);
        roi &= cv::Rect(0, 0, cols, rows);
        for (int rr = roi.y; rr < roi.y + roi.height; ++rr)
            for (int cc = roi.x; cc < roi.x + roi.width; ++cc)
                if (work_area.at<cv::Vec3b>(rr, cc) == cv::Vec3b(128, 128, 128))
                    { uncovered.at<uint8_t>(rr, cc) = 0; covered.at<uint8_t>(rr, cc) = 255; }

        uncovered_remaining = cv::countNonZero(uncovered);
        if (pass2_iter % 5 == 0 || uncovered_remaining == 0)
            std::cout << "  Phase2 iter " << pass2_iter << ": +" << phase2_added
                      << " waypoints, " << uncovered_remaining << " left\n" << std::flush;
    }

    std::cout << "Total: " << wp_id << " waypoints (phase1=" << phase1_wp
              << " + phase2=" << phase2_added << ")\n";

    // ══════════════════════════════════════════════════════
    // 后处理: 蛇形排序 + 路径插值
    // ══════════════════════════════════════════════════════
    if (waypoints.size() > 1) {
        std::sort(waypoints.begin(), waypoints.end(),
            [](const Waypoint& a, const Waypoint& b) {
                if (std::fabs(a.baselink_center.y - b.baselink_center.y) > 0.3f)
                    return a.baselink_center.y < b.baselink_center.y;
                return a.baselink_center.x < b.baselink_center.x;
            });
        std::vector<std::vector<Waypoint>> wp_rows;
        std::vector<Waypoint> cur;
        float ly = waypoints[0].baselink_center.y - 2;
        for (auto& wp : waypoints) {
            if (std::fabs(wp.baselink_center.y - ly) > 0.3f) {
                if (!cur.empty()) wp_rows.push_back(std::move(cur));
                cur.clear();
                ly = wp.baselink_center.y;
            }
            cur.push_back(std::move(wp));
        }
        if (!cur.empty()) wp_rows.push_back(std::move(cur));

        // 过滤孤立行
        std::vector<std::vector<Waypoint>> clean_rows;
        for (size_t r = 0; r < wp_rows.size(); ++r) {
            if (wp_rows[r].size() <= 2 && wp_rows.size() > 3) {
                float best_d = 1e9f;
                int best_ri = -1;
                for (size_t rr = 0; rr < wp_rows.size(); ++rr) {
                    if (rr == r || wp_rows[rr].size() <= 2) continue;
                    float d = std::fabs(wp_rows[r][0].baselink_center.y
                                        - wp_rows[rr][0].baselink_center.y);
                    if (d < best_d) { best_d = d; best_ri = static_cast<int>(rr); }
                }
                if (best_ri >= 0) {
                    for (auto& wp : wp_rows[r])
                        wp_rows[best_ri].push_back(std::move(wp));
                } else {
                    clean_rows.push_back(std::move(wp_rows[r]));
                }
            } else {
                clean_rows.push_back(std::move(wp_rows[r]));
            }
        }

        waypoints.clear();
        for (size_t r = 0; r < clean_rows.size(); ++r) {
            std::sort(clean_rows[r].begin(), clean_rows[r].end(),
                [](const Waypoint& a, const Waypoint& b) {
                    return a.baselink_center.x < b.baselink_center.x;
                });
            if (r % 2 == 1)
                std::reverse(clean_rows[r].begin(), clean_rows[r].end());
            for (auto& wp : clean_rows[r])
                waypoints.push_back(std::move(wp));
        }
        for (size_t i = 0; i < waypoints.size(); ++i)
            waypoints[i].id = static_cast<int>(i);

        // 生成路径
        std::vector<cv::Point2f> path;
        const float max_gap = 1.4f;
        for (const auto& wp : waypoints) {
            cv::Point2f pt = wp.baselink_center;
            if (!path.empty()) {
                cv::Point2f& prev = path.back();
                float dx = pt.x - prev.x, dy = pt.y - prev.y;
                float d = std::sqrt(dx * dx + dy * dy);
                int sp = std::max(1, static_cast<int>(std::ceil(d / max_gap)));
                for (int s = 1; s <= sp; ++s) {
                    float t = static_cast<float>(s) / sp;
                    path.push_back({prev.x + dx * t, prev.y + dy * t});
                }
            } else {
                path.push_back(pt);
            }
        }

        result.path = path;
        result.total_path_length_m = 0;
        for (size_t i = 1; i < path.size(); ++i) {
            float dx = path[i].x - path[i - 1].x;
            float dy = path[i].y - path[i - 1].y;
            result.total_path_length_m += std::sqrt(dx * dx + dy * dy);
        }
    }

    result.waypoints = waypoints;
    result.total_waypoints = static_cast<int>(waypoints.size());

    // ── 覆盖率 ──────────────────────────────────────
    int covered_cnt = cv::countNonZero(covered);
    result.coverage_ratio = total_gray > 0
        ? static_cast<float>(covered_cnt) / total_gray : 0.0f;
    result.overlap_ratio = 0.0f;

    std::cout << "Waypoints: " << result.total_waypoints << "\n"
              << "Coverage: " << result.coverage_ratio * 100 << "%\n"
              << "Path len: " << result.total_path_length_m << "m\n";

    return result;
}

// ═══════════════════════════════════════════════════════════
// 可视化 (多图拆分)
// ═══════════════════════════════════════════════════════════
void CoveragePlanner::save_visualization(const std::string& out_dir,
                                          const cv::Mat& work_area,
                                          const CoverageResult& result,
                                          float u_min, float v_min) const
{
    auto w2p = [&](const cv::Point2f& w) -> cv::Point {
        return world_to_px(w, u_min, v_min);
    };

    // ── 图1: base 覆盖区域 (半透明绿) + 编号 ─────────
    {
        cv::Mat base_img = work_area.clone();
        cv::Mat overlay = base_img.clone();
        for (size_t i = 0; i < result.waypoints.size(); ++i) {
            const auto& wp = result.waypoints[i];
            cv::Point bs_px = w2p(wp.base_center);
            int bs_hx = static_cast<int>(wp.base_x_size / params_.pixel_size / 2);
            int bs_hy = static_cast<int>(wp.base_y_size / params_.pixel_size / 2);
            cv::Rect base_rect(bs_px.x - bs_hx, bs_px.y - bs_hy, bs_hx * 2, bs_hy * 2);
            cv::rectangle(overlay, base_rect, cv::Scalar(0, 255, 0), -1);
            cv::rectangle(base_img, base_rect, cv::Scalar(0, 200, 0), 1);
            cv::putText(base_img, std::to_string(i), bs_px + cv::Point(2, 4),
                        cv::FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(0, 0, 0), 1);
        }
        cv::addWeighted(base_img, 0.7, overlay, 0.3, 0, base_img);
        cv::imwrite(out_dir + "/01_base_coverage.png", base_img);
    }

    // ── 图2: baselink 位置 + 编号 + 行进方向箭头 ──
    {
        cv::Mat bl_img = work_area.clone();
        for (size_t i = 0; i < result.waypoints.size(); ++i) {
            const auto& wp = result.waypoints[i];
            cv::Point bl_px = w2p(wp.baselink_center);
            int bl_hx = static_cast<int>(params_.baselink_x / params_.pixel_size / 2);
            int bl_hy = static_cast<int>(params_.baselink_y / params_.pixel_size / 2);
            cv::Scalar color = cv::Scalar(255, 0, 0);

            cv::rectangle(bl_img,
                cv::Point(bl_px.x - bl_hx, bl_px.y - bl_hy),
                cv::Point(bl_px.x + bl_hx, bl_px.y + bl_hy),
                color, 1);
            cv::circle(bl_img, bl_px, 2, color, -1);
            cv::putText(bl_img, std::to_string(i), bl_px + cv::Point(5, -5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1);

            cv::Point arrow_tip(bl_px.x + wp.direction * 15, bl_px.y);
            cv::arrowedLine(bl_img, bl_px, arrow_tip, cv::Scalar(0, 0, 255), 1, cv::LINE_8, 0, 0.2);
        }
        cv::imwrite(out_dir + "/02_baselink_positions.png", bl_img);
    }

    // ── 图3: 路径连线 (跳变段半透明黄色标注) ──
    {
        cv::Mat path_img = work_area.clone();
        cv::Mat jump_ov = path_img.clone();

        std::vector<float> seg_lens;
        for (size_t i = 1; i < result.path.size(); ++i) {
            float dx = result.path[i].x - result.path[i-1].x;
            float dy = result.path[i].y - result.path[i-1].y;
            seg_lens.push_back(std::sqrt(dx*dx + dy*dy));
        }
        std::sort(seg_lens.begin(), seg_lens.end());
        float median_step = seg_lens.empty() ? 1.0f : seg_lens[seg_lens.size()/2];
        float jump_thresh = std::max(3.0f, median_step * 3.0f);

        int jump_count = 0;
        for (size_t i = 1; i < result.path.size(); ++i) {
            cv::Point p1 = w2p(result.path[i-1]);
            cv::Point p2 = w2p(result.path[i]);
            float dx = result.path[i].x - result.path[i-1].x;
            float dy = result.path[i].y - result.path[i-1].y;
            float d = std::sqrt(dx*dx + dy*dy);
            bool is_jump = (d > jump_thresh);

            cv::Scalar color = is_jump ? cv::Scalar(0, 255, 255)
                           : cv::Scalar(0, 255 * (1 - (float)i/result.path.size()),
                                        255 * (float)i/result.path.size());

            if (is_jump) {
                cv::line(jump_ov, p1, p2, cv::Scalar(0, 255, 255), 3);
                jump_count++;
            }
            cv::line(path_img, p1, p2, color, 1);

            if (i % 3 == 0 && !is_jump) {
                cv::Point mid((p1.x+p2.x)/2, (p1.y+p2.y)/2);
                cv::Point dir(p2.x - p1.x, p2.y - p1.y);
                float len = std::sqrt(dir.x*dir.x + dir.y*dir.y);
                if (len > 1) { dir.x=dir.x*5/len; dir.y=dir.y*5/len;
                    cv::arrowedLine(path_img, mid, mid+dir, color, 1, cv::LINE_8, 0, 0.3); }
            }
        }

        cv::addWeighted(path_img, 0.75, jump_ov, 0.25, 0, path_img);

        if (!result.path.empty()) {
            cv::circle(path_img, w2p(result.path.front()), 4, cv::Scalar(0,255,0), -1);
            cv::circle(path_img, w2p(result.path.back()), 4, cv::Scalar(0,0,255), -1);
            cv::putText(path_img, "START", w2p(result.path.front())+cv::Point(5,-5), cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,255,0),1);
            cv::putText(path_img, "END",   w2p(result.path.back()) +cv::Point(5,-5), cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,0,255),1);
        }
        std::cout << "  Jump segments: " << jump_count << " (thresh=" << jump_thresh << "m)\n";
        cv::imwrite(out_dir + "/03_path.png", path_img);
    }

    // ── 图4: baselink 路径序号 ───────────────────
    {
        cv::Mat seq_img = work_area.clone();
        for (size_t r = 0; r < result.waypoints.size(); ++r) {
            const auto& wp = result.waypoints[r];
            cv::Point bl_px = w2p(wp.baselink_center);
            cv::circle(seq_img, bl_px, 3, cv::Scalar(255, 0, 0), -1);
            cv::putText(seq_img, std::to_string(r),
                        bl_px + cv::Point(3, -3),
                        cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255), 1);
        }
        cv::imwrite(out_dir + "/04_baselink_sequence.png", seq_img);
    }

    // ── 图5: 综合图 (轻量) ──────────────────────
    {
        cv::Mat all = work_area.clone();
        for (size_t i = 1; i < result.path.size(); ++i)
            cv::line(all, w2p(result.path[i-1]), w2p(result.path[i]),
                     cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
        for (const auto& wp : result.waypoints) {
            cv::Point bs_px = w2p(wp.base_center);
            int bs_hx = static_cast<int>(wp.base_x_size / params_.pixel_size / 2);
            int bs_hy = static_cast<int>(wp.base_y_size / params_.pixel_size / 2);
            cv::rectangle(all,
                cv::Point(bs_px.x - bs_hx, bs_px.y - bs_hy),
                cv::Point(bs_px.x + bs_hx, bs_px.y + bs_hy),
                cv::Scalar(0, 255, 0), 1);
        }
        cv::imwrite(out_dir + "/05_overview.png", all);
    }

    std::cout << "  Saved: " << out_dir << "/01-05_*.png\n";
}

void CoveragePlanner::save_json(const std::string& path,
                                 const CoverageResult& result) const
{
    std::ofstream f(path);
    f << "{\n  \"waypoints\": [\n";
    for (size_t i = 0; i < result.waypoints.size(); ++i) {
        const auto& wp = result.waypoints[i];
        f << "    {\"id\":" << wp.id
          << ",\"baselink\":[" << wp.baselink_center.x << "," << wp.baselink_center.y << "]"
          << ",\"base_center\":[" << wp.base_center.x << "," << wp.base_center.y << "]"
          << ",\"base_size\":[" << wp.base_x_size << "," << wp.base_y_size << "]"
          << ",\"dir\":" << wp.direction << "}";
        if (i < result.waypoints.size() - 1) f << ",";
        f << "\n";
    }
    f << "  ],\n"
      << "  \"statistics\": {\"total_waypoints\":" << result.total_waypoints
      << ",\"coverage_ratio\":" << result.coverage_ratio
      << ",\"path_length_m\":" << result.total_path_length_m << "}\n}\n";
    f.close();
    std::cout << "  Saved: " << path << "\n";
}
