//=============================================================================
// coverage_planner.cpp — Two-Phase Coverage Path Planning V8
//
// Phase 1: 边界环 — 沿侧墩 AABB 四边平铺, 方向保证边界在左侧
// Phase 2: 内部滑动窗口 — X 轴方向规则填充
// 后处理: 连通区域聚类 + 区域内蛇形 + 区域间 2-opt
//=============================================================================
#include "coverage_planner.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <set>
#include <map>
#include <queue>

CoveragePlanner::CoveragePlanner(const CoverageParams& params)
    : params_(params) {}

namespace {
    const float MIN_BASE_EDGE = 0.4f;       // 最小 base 边长
    const float MIN_BASE_AREA = 0.16f;       // 最小 base 面积 = 0.4²
}

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
            if (row[c] != cv::Vec3b(128, 128, 128))
                return false;
            if (urow && urow[c] == 0)
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

// ── 桥墩在左约束 (X 轴行进用) ──────────────────────────
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

// ── Phase 1: 沿边界轮廓采样 ────────────────────────────
static std::vector<cv::Point> sample_boundary_contours(
    const cv::Mat& work_area, const cv::Mat& uncovered, float pixel_size)
{
    std::vector<cv::Point> samples;
    int rows = work_area.rows, cols = work_area.cols;

    // 提取红色边界 mask
    cv::Mat red_mask(rows, cols, CV_8UC1, cv::Scalar(0));
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            if (work_area.at<cv::Vec3b>(r, c) == cv::Vec3b(0, 0, 255))
                red_mask.at<uint8_t>(r, c) = 255;

    // 找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(red_mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    // 沿每条轮廓采样
    int sample_interval = std::max(2, static_cast<int>(0.15f / pixel_size));  // 每 0.15m 采一个点

    for (const auto& cnt : contours) {
        int n = static_cast<int>(cnt.size());
        if (n < 10) continue;  // 太短的轮廓跳过

        for (int i = 0; i < n; i += sample_interval) {
            cv::Point pt = cnt[i];

            // 向灰色区域方向偏移 0.5m, 找最近的未覆盖灰色像素作为 base 中心候选
            // 探测 4 个方向, 找灰色区域
            int probe_dist = static_cast<int>(0.6f / pixel_size);
            int best_off_x = 0, best_off_y = 0;
            int best_gray = 0;

            for (int dy = -probe_dist; dy <= probe_dist; dy += std::max(1, probe_dist / 5)) {
                for (int dx = -probe_dist; dx <= probe_dist; dx += std::max(1, probe_dist / 5)) {
                    int px = pt.x + dx, py = pt.y + dy;
                    if (px < 0 || px >= cols || py < 0 || py >= rows) continue;
                    if (work_area.at<cv::Vec3b>(py, px) == cv::Vec3b(128, 128, 128) &&
                        uncovered.at<uint8_t>(py, px) > 0) {
                        // 找到了, 记录偏移
                        if (std::abs(dx) + std::abs(dy) < std::abs(best_off_x) + std::abs(best_off_y)
                            || (best_off_x == 0 && best_off_y == 0)) {
                            best_off_x = dx; best_off_y = dy;
                        }
                    }
                }
            }
            if (best_off_x == 0 && best_off_y == 0) continue;

            // 从边界向内偏移, 但不越界
            int cx = pt.x + best_off_x / 2;  // 取一半偏移, 让 base 靠近边界
            int cy = pt.y + best_off_y / 2;
            cx = std::max(0, std::min(cols - 1, cx));
            cy = std::max(0, std::min(rows - 1, cy));
            samples.push_back(cv::Point(cx, cy));
        }
    }
    return samples;
}

// ── Phase 1: 检查边界是否在 base 方向左侧 ──────────────
// dir: 0=+X(左+Y), 1=-X(左-Y), 2=+Y(左-X), 3=-Y(左+X)
bool CoveragePlanner::boundary_on_left(const cv::Mat& work_area,
                                        const cv::Point2f& center,
                                        float /*bx*/, float /*by*/, int dir,
                                        float u_min, float v_min) const
{
    float check_dist = params_.base_max + 0.6f;  // 需要覆盖 base 半径 + 到边界的距离
    int cols = work_area.cols, rows = work_area.rows;

    float step_x = 0, step_y = 0;
    switch (dir) {
        case 0: step_y = +0.05f; break;  // +X → 左=+Y
        case 1: step_y = -0.05f; break;  // -X → 左=-Y
        case 2: step_x = -0.05f; break;  // +Y → 左=-X
        case 3: step_x = +0.05f; break;  // -Y → 左=+X
    }

    for (float d = 0; d <= check_dist; d += 0.05f) {
        float sx = center.x + step_x * (d / 0.05f);
        float sy = center.y + step_y * (d / 0.05f);
        cv::Point pt = world_to_px({sx, sy}, u_min, v_min);
        if (pt.x < 0 || pt.x >= cols || pt.y < 0 || pt.y >= rows) continue;
        if (work_area.at<cv::Vec3b>(pt.y, pt.x) == cv::Vec3b(0, 0, 255))
            return true;
    }
    return false;
}

// ── baselink 方向从 base 方向推导 ──────────────────────
bool CoveragePlanner::base_dir_to_baselink(const cv::Mat& work_area,
                                            const cv::Point2f& base_center,
                                            float bx, float by, int base_dir,
                                            float u_min, float v_min,
                                            cv::Point2f& out_bl) const
{
    int rows = work_area.rows, cols = work_area.cols;
    float bl_hx = params_.baselink_x / 2, bl_hy = params_.baselink_y / 2;

    // baselink 偏移方向: base_dir 的反方向偏移 center_offset
    float off_x = 0, off_y = 0;
    switch (base_dir) {
        case 0: off_x = -params_.center_offset; break;  // +X → baselink 在左(-X)
        case 1: off_x = +params_.center_offset; break;  // -X → baselink 在右(+X)
        case 2: off_y = -params_.center_offset; break;  // +Y → baselink 在下(-Y)
        case 3: off_y = +params_.center_offset; break;  // -Y → baselink 在上(+Y)
    }

    cv::Point2f bl_ctr(base_center.x + off_x, base_center.y + off_y);

    // 检查 baselink 是否碰到红色
    cv::Point tl = world_to_px({bl_ctr.x - bl_hx, bl_ctr.y - bl_hy}, u_min, v_min);
    cv::Point br = world_to_px({bl_ctr.x + bl_hx, bl_ctr.y + bl_hy}, u_min, v_min);
    for (int rr = std::max(0, tl.y); rr <= std::min(rows - 1, br.y); ++rr)
        for (int cc = std::max(0, tl.x); cc <= std::min(cols - 1, br.x); ++cc)
            if (work_area.at<cv::Vec3b>(rr, cc) == cv::Vec3b(0, 0, 255))
                return false;

    out_bl = bl_ctr;
    return true;
}

// ═══════════════════════════════════════════════════════════
// 主入口: 三阶段覆盖规划
// ═══════════════════════════════════════════════════════════
CoverageResult CoveragePlanner::plan(const cv::Mat& work_area,
                                      float u_min, float v_min)
{
    CoverageResult result;
    if (work_area.empty()) return result;

    std::cout << "\n========== Coverage Planning V6 ==========\n";

    // ── 边界简化: 侧墩轮廓 → AABB 矩形 ──────────────
    cv::Mat wa = work_area.clone();
    std::vector<cv::Rect> pier_rects;  // 收集所有侧墩 AABB
    {
        int rows = wa.rows, cols = wa.cols;
        cv::Mat red_mask(rows, cols, CV_8UC1, cv::Scalar(0));
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                if (wa.at<cv::Vec3b>(r, c) == cv::Vec3b(0, 0, 255))
                    red_mask.at<uint8_t>(r, c) = 255;

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(red_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                if (wa.at<cv::Vec3b>(r, c) == cv::Vec3b(0, 0, 255))
                    wa.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);

        for (const auto& cnt : contours) {
            if (cnt.size() < 5) continue;
            cv::Rect rect = cv::boundingRect(cnt);
            int margin = static_cast<int>(0.05f / params_.pixel_size);
            rect.x = std::max(0, rect.x - margin);
            rect.y = std::max(0, rect.y - margin);
            rect.width  = std::min(cols - rect.x, rect.width  + margin * 2);
            rect.height = std::min(rows - rect.y, rect.height + margin * 2);
            for (int rr = rect.y; rr < rect.y + rect.height; ++rr)
                for (int cc = rect.x; cc < rect.x + rect.width; ++cc)
                    wa.at<cv::Vec3b>(rr, cc) = cv::Vec3b(0, 0, 255);
            pier_rects.push_back(rect);
        }
        std::cout << "Simplified piers: " << pier_rects.size() << " → AABB\n";
    }

    int rows = wa.rows, cols = wa.cols;

    // ── 构建 mask ──────────────────────────────────
    cv::Mat uncovered(rows, cols, CV_8UC1, cv::Scalar(0));
    cv::Mat gray_mask(rows, cols, CV_8UC1, cv::Scalar(0));
    cv::Mat covered(rows, cols, CV_8UC1, cv::Scalar(0));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (wa.at<cv::Vec3b>(r, c) == cv::Vec3b(128, 128, 128)) {
                uncovered.at<uint8_t>(r, c) = 255;
                gray_mask.at<uint8_t>(r, c) = 255;
            }
        }
    }

    int total_gray = cv::countNonZero(gray_mask);
    std::cout << "Gray pixels: " << total_gray << "\n";
    if (total_gray == 0) return result;

    std::vector<Waypoint> waypoints;
    int wp_id = 0;

    // 灰色区域范围
    int x_min_px = cols, x_max_px = 0, y_min_px = rows, y_max_px = 0;
    for (int r = 0; r < rows; ++r) {
        const uint8_t* gr = gray_mask.ptr<uint8_t>(r);
        for (int c = 0; c < cols; ++c)
            if (gr[c] > 0) {
                x_min_px = std::min(x_min_px, c); x_max_px = std::max(x_max_px, c);
                y_min_px = std::min(y_min_px, r); y_max_px = std::max(y_max_px, r);
            }
    }

    // ══════════════════════════════════════════════════
    // Phase 1: 每个侧墩周围固定位置放置 base (蓝色)
    // ══════════════════════════════════════════════════
    // Phase 1: 沿侧墩 AABB 四边平铺一圈 base (蓝色)
    //
    //   每条边均匀步长 base_max (1.17m), 中心偏移 offset=0.75m
    //   每条边固定行进方向使侧墩边界在左侧
    // ══════════════════════════════════════════════════
    {
        float offset = 0.75f;  // > baselink_x/2, 保证 baselink 不碰红
        int off_px = static_cast<int>(offset / params_.pixel_size);
        int base_max_px = static_cast<int>(params_.base_max / params_.pixel_size);
        int p1_count = 0;

        for (const auto& pier : pier_rects) {
            if (pier.width * params_.pixel_size > 3.0f) continue;

            int x0 = pier.x, y0 = pier.y;
            int x1 = pier.x + pier.width, y1 = pier.y + pier.height;

            // 四条边的候选中心 + 固定方向
            struct { std::vector<cv::Point> centers; int dir; } edges[4];

            // 顶边 (墩上方, base_y = y0 - off_px, 沿 X 均匀分布)
            {
                int n = std::max(1, static_cast<int>(std::ceil(
                    static_cast<float>(pier.width) / base_max_px)));
                for (int k = 0; k < n; ++k) {
                    float frac = (k + 0.5f) / n;
                    int cx = x0 + static_cast<int>(frac * pier.width);
                    edges[0].centers.push_back(cv::Point(cx, y0 - off_px));
                }
                edges[0].dir = 0;  // +X → 左=+Y=向下=墩在下方 ✓
            }
            // 底边 (墩下方, base_y = y1 + off_px, 沿 X 均匀分布)
            {
                int n = std::max(1, static_cast<int>(std::ceil(
                    static_cast<float>(pier.width) / base_max_px)));
                for (int k = 0; k < n; ++k) {
                    float frac = (k + 0.5f) / n;
                    int cx = x0 + static_cast<int>(frac * pier.width);
                    edges[1].centers.push_back(cv::Point(cx, y1 + off_px));
                }
                edges[1].dir = 1;  // -X → 左=-Y=向上=墩在上方 ✓
            }
            // 左边 (墩左侧, base_x = x0 - off_px, 沿 Y 均匀分布)
            {
                int n = std::max(1, static_cast<int>(std::ceil(
                    static_cast<float>(pier.height) / base_max_px)));
                for (int k = 0; k < n; ++k) {
                    float frac = (k + 0.5f) / n;
                    int cy = y0 + static_cast<int>(frac * pier.height);
                    edges[2].centers.push_back(cv::Point(x0 - off_px, cy));
                }
                edges[2].dir = 3;  // -Y → 左=+X=向右=墩在右侧 ✓
            }
            // 右边 (墩右侧, base_x = x1 + off_px, 沿 Y 均匀分布)
            {
                int n = std::max(1, static_cast<int>(std::ceil(
                    static_cast<float>(pier.height) / base_max_px)));
                for (int k = 0; k < n; ++k) {
                    float frac = (k + 0.5f) / n;
                    int cy = y0 + static_cast<int>(frac * pier.height);
                    edges[3].centers.push_back(cv::Point(x1 + off_px, cy));
                }
                edges[3].dir = 2;  // +Y → 左=-X=向左=墩在左侧 ✓
            }

            for (const auto& edge : edges) {
                for (const auto& pt : edge.centers) {
                    if (pt.x < 0 || pt.x >= cols || pt.y < 0 || pt.y >= rows) continue;
                    if (uncovered.at<uint8_t>(pt.y, pt.x) == 0) continue;
                    if (wa.at<cv::Vec3b>(pt.y, pt.x) != cv::Vec3b(128, 128, 128)) continue;

                    cv::Point2f origin = px_to_world(pt, u_min, v_min);
                    float bx = 0.2f, by = 0.2f;
                    expand_base(wa, origin, bx, by, u_min, v_min, &uncovered);
                    if (bx < MIN_BASE_EDGE || by < MIN_BASE_EDGE) continue;
                    if (!boundary_on_left(wa, origin, bx, by, edge.dir, u_min, v_min))
                        continue;

                    cv::Point2f bl;
                    if (!base_dir_to_baselink(wa, origin, bx, by, edge.dir,
                                               u_min, v_min, bl))
                        continue;

                    Waypoint wp;
                    wp.id = wp_id++;
                    wp.base_center = origin;
                    wp.baselink_center = bl;
                    wp.base_x_size = bx;
                    wp.base_y_size = by;
                    wp.stripe = 0;
                    wp.direction = edge.dir;
                    waypoints.push_back(wp);
                    p1_count++;

                    cv::Point px = world_to_px(origin, u_min, v_min);
                    int hx = static_cast<int>(bx / 2 / params_.pixel_size);
                    int hy = static_cast<int>(by / 2 / params_.pixel_size);
                    cv::Rect roi(px.x - hx, px.y - hy, hx * 2, hy * 2);
                    roi &= cv::Rect(0, 0, cols, rows);
                    for (int rr = roi.y; rr < roi.y + roi.height; ++rr)
                        for (int cc = roi.x; cc < roi.x + roi.width; ++cc)
                            if (wa.at<cv::Vec3b>(rr, cc) == cv::Vec3b(128, 128, 128))
                                { uncovered.at<uint8_t>(rr, cc) = 0; covered.at<uint8_t>(rr, cc) = 255; }
                }
            }
        }
        std::cout << "Phase 1: " << p1_count << " waypoints (ring around "
                  << pier_rects.size() << " piers)\n";
    }

    // ══════════════════════════════════════════════════
    // Phase 2: 内部滑动窗口 (绿色, X 轴方向)
    // ══════════════════════════════════════════════════
    {
        float step_x = 1.4f;
        float step_y = params_.base_max;  // 1.17m
        float start_x = u_min + (x_min_px + params_.base_max / 2.0f / params_.pixel_size) * params_.pixel_size;
        float start_y = v_min + (y_min_px + params_.base_max / 2.0f / params_.pixel_size) * params_.pixel_size;
        int p2_count = 0;

        for (float wy = start_y; ; wy += step_y) {
            int gy = static_cast<int>((wy - v_min) / params_.pixel_size);
            if (gy > y_max_px) break;

            for (float wx = start_x; ; wx += step_x) {
                int gx = static_cast<int>((wx - u_min) / params_.pixel_size);
                if (gx > x_max_px) break;
                if (gx < 0 || gx >= cols || gy < 0 || gy >= rows) continue;
                if (gray_mask.at<uint8_t>(gy, gx) == 0 || uncovered.at<uint8_t>(gy, gx) == 0) continue;

                cv::Point2f bc(wx, wy);
                float bx = params_.base_max, by = params_.base_max;
                bool placed = false;

                // 1) 原始中心 + 全尺寸
                if (is_base_valid(wa, bc, bx, by, u_min, v_min, &uncovered)) {
                    placed = true;
                }

                // 2) 移动中心 (±0.3m)
                if (!placed) {
                    for (float dy = -0.30f; dy <= 0.30f && !placed; dy += 0.05f) {
                        for (float dx = -0.30f; dx <= 0.30f && !placed; dx += 0.05f) {
                            if (dx == 0 && dy == 0) continue;
                            cv::Point2f shifted(wx + dx, wy + dy);
                            if (is_base_valid(wa, shifted, bx, by, u_min, v_min, &uncovered)) {
                                bc = shifted; placed = true;
                            }
                        }
                    }
                }

                // 3) 缩小
                if (!placed) {
                    for (float scale = 0.95f; scale * params_.base_max >= MIN_BASE_EDGE; scale -= 0.05f) {
                        float sx = params_.base_max * scale;
                        float sy = params_.base_max * scale;
                        if (is_base_valid(wa, bc, sx, sy, u_min, v_min, &uncovered)) {
                            bx = sx; by = sy; placed = true; break;
                        }
                        for (float dy = -0.20f; dy <= 0.20f && !placed; dy += 0.10f) {
                            for (float dx = -0.20f; dx <= 0.20f && !placed; dx += 0.10f) {
                                cv::Point2f shifted(wx + dx, wy + dy);
                                if (is_base_valid(wa, shifted, sx, sy, u_min, v_min, &uncovered)) {
                                    bc = shifted; bx = sx; by = sy; placed = true;
                                }
                            }
                        }
                        if (placed) break;
                    }
                }

                if (!placed) continue;

                // 扩展 base 到最大合法尺寸
                expand_base(wa, bc, bx, by, u_min, v_min, &uncovered);
                if (bx < MIN_BASE_EDGE || by < MIN_BASE_EDGE) continue;

                // 方向: X 轴交替, 但需满足 pier-on-left (近边界时强制)
                int base_dir = (static_cast<int>(wy / step_y) % 2 == 0) ? 0 : 1;  // 0=+X, 1=-X
                bool dir_ok = false;
                cv::Point2f bl;
                for (int try_dir : {base_dir, (base_dir == 0 ? 1 : 0)}) {
                    // 近边界 (<0.6m) 时强制 boundary_on_left
                    bool near_red = false;
                    cv::Point bc_px = world_to_px(bc, u_min, v_min);
                    int ck = static_cast<int>(0.6f / params_.pixel_size);
                    for (int rr = std::max(0, bc_px.y - ck);
                         rr <= std::min(rows - 1, bc_px.y + ck) && !near_red; ++rr)
                        for (int cc = std::max(0, bc_px.x - ck);
                             cc <= std::min(cols - 1, bc_px.x + ck) && !near_red; ++cc)
                            if (wa.at<cv::Vec3b>(rr, cc) == cv::Vec3b(0, 0, 255))
                                near_red = true;

                    if (near_red && !boundary_on_left(wa, bc, bx, by, try_dir, u_min, v_min))
                        continue;

                    if (!base_dir_to_baselink(wa, bc, bx, by, try_dir, u_min, v_min, bl))
                        continue;

                    base_dir = try_dir;
                    dir_ok = true;
                    break;
                }
                if (!dir_ok) continue;

                Waypoint wp;
                wp.id = wp_id++;
                wp.base_center = bc;
                wp.baselink_center = bl;
                wp.base_x_size = bx;
                wp.base_y_size = by;
                wp.stripe = 1;   // Phase 2 = 绿色
                wp.direction = base_dir;
                waypoints.push_back(wp);
                p2_count++;

                cv::Point px = world_to_px(bc, u_min, v_min);
                int hx = static_cast<int>(bx / 2 / params_.pixel_size);
                int hy = static_cast<int>(by / 2 / params_.pixel_size);
                cv::Rect roi(px.x - hx, px.y - hy, hx * 2, hy * 2);
                roi &= cv::Rect(0, 0, cols, rows);
                for (int rr = roi.y; rr < roi.y + roi.height; ++rr)
                    for (int cc = roi.x; cc < roi.x + roi.width; ++cc)
                        if (wa.at<cv::Vec3b>(rr, cc) == cv::Vec3b(128, 128, 128))
                            { uncovered.at<uint8_t>(rr, cc) = 0; covered.at<uint8_t>(rr, cc) = 255; }
            }
        }
        std::cout << "Phase 2: " << p2_count << " waypoints\n";
    }

    // Phase 3: MER disabled for comparison

    std::cout << "Total: " << wp_id << " waypoints\n";

    // ══════════════════════════════════════════════════
    // 后处理 V7: 连通区域聚类 → 区域内短轴蛇形 → 区域间贪心NN+2-opt
    // ══════════════════════════════════════════════════
    if (waypoints.size() > 1) {
        // ── Pass A: 连通区域标记 ──────────────────────
        // gray_mask 反映 AABB 简化后的拓扑, 4-连通避免对角粘连
        cv::Mat labels;
        cv::connectedComponents(gray_mask, labels, 4, CV_32S);

        auto label_at = [&](const cv::Point2f& bc) -> int {
            cv::Point p = world_to_px(bc, u_min, v_min);
            p.x = std::max(0, std::min(cols - 1, p.x));
            p.y = std::max(0, std::min(rows - 1, p.y));
            int L = labels.at<int>(p.y, p.x);
            // 落在被腐蚀像素(0)时, 邻域找最近非零 label
            for (int rad = 1; rad <= 5 && L == 0; ++rad)
                for (int dy = -rad; dy <= rad && L == 0; ++dy)
                    for (int dx = -rad; dx <= rad && L == 0; ++dx) {
                        int yy = p.y + dy, xx = p.x + dx;
                        if (yy >= 0 && yy < rows && xx >= 0 && xx < cols &&
                            labels.at<int>(yy, xx) != 0)
                            L = labels.at<int>(yy, xx);
                    }
            return L;
        };

        struct RegionGroup {
            std::vector<Waypoint> ring;   // stripe==0 (P1 侧墩环)
            std::vector<Waypoint> fill;   // stripe 1/2 (P2/P3 填充)
            std::vector<Waypoint> ordered;
            cv::Point2f centroid{0, 0};
            float x_min = 1e9f, x_max = -1e9f, y_min = 1e9f, y_max = -1e9f;
            cv::Point2f entry{0, 0}, exit{0, 0};
        };

        std::map<int, RegionGroup> groups;  // 有序 → 确定性
        for (auto& wp : waypoints) {
            wp.region = label_at(wp.base_center);
            auto& g = groups[wp.region];
            (wp.stripe == 0 ? g.ring : g.fill).push_back(wp);
        }

        // 收集到 vector, 顺带算 bbox/centroid
        std::vector<RegionGroup> regs;
        regs.reserve(groups.size());
        for (auto& kv : groups) {
            RegionGroup& g = kv.second;
            double sx = 0, sy = 0; int n = 0;
            auto acc = [&](const std::vector<Waypoint>& v) {
                for (const auto& wp : v) {
                    sx += wp.base_center.x; sy += wp.base_center.y; ++n;
                    g.x_min = std::min(g.x_min, wp.base_center.x);
                    g.x_max = std::max(g.x_max, wp.base_center.x);
                    g.y_min = std::min(g.y_min, wp.base_center.y);
                    g.y_max = std::max(g.y_max, wp.base_center.y);
                }
            };
            acc(g.ring); acc(g.fill);
            if (n > 0) { g.centroid = {static_cast<float>(sx / n),
                                        static_cast<float>(sy / n)}; }
            regs.push_back(std::move(g));
        }

        // 微区域(总航点≤2)并入空间最近的大区域 fill
        {
            std::vector<RegionGroup> big, small;
            for (auto& g : regs) {
                if (g.ring.size() + g.fill.size() <= 2) small.push_back(std::move(g));
                else big.push_back(std::move(g));
            }
            if (big.empty()) { big.swap(small); small.clear(); }
            for (auto& s : small) {
                int best = 0; float bd = 1e18f;
                for (size_t i = 0; i < big.size(); ++i) {
                    float dx = big[i].centroid.x - s.centroid.x;
                    float dy = big[i].centroid.y - s.centroid.y;
                    float d = dx * dx + dy * dy;
                    if (d < bd) { bd = d; best = static_cast<int>(i); }
                }
                for (auto& wp : s.ring)  big[best].fill.push_back(std::move(wp));
                for (auto& wp : s.fill)  big[best].fill.push_back(std::move(wp));
            }
            regs.swap(big);
        }

        // ── Pass B: 区域内排序 ───────────────────────
        float band_w = params_.stripe_width;  // 1.17m, 每物理行一带
        for (auto& g : regs) {
            // ring: 极角排序成环 (CCW), 每墩 ≤8 点
            std::sort(g.ring.begin(), g.ring.end(),
                [&](const Waypoint& a, const Waypoint& b) {
                    return std::atan2(a.base_center.y - g.centroid.y,
                                      a.base_center.x - g.centroid.x)
                         < std::atan2(b.base_center.y - g.centroid.y,
                                      b.base_center.x - g.centroid.x);
                });

            // fill: 按 bbox 选带轴 (短轴分带, 长轴扫) → 蛇形
            bool bandY = (g.x_max - g.x_min) >= (g.y_max - g.y_min);
            float axis_min = bandY ? g.y_min : g.x_min;
            std::stable_sort(g.fill.begin(), g.fill.end(),
                [&](const Waypoint& a, const Waypoint& b) {
                    float ca = bandY ? a.baselink_center.y : a.baselink_center.x;
                    float cb = bandY ? b.baselink_center.y : b.baselink_center.x;
                    int ba = static_cast<int>(std::floor((ca - axis_min) / band_w));
                    int bb = static_cast<int>(std::floor((cb - axis_min) / band_w));
                    if (ba != bb) return ba < bb;
                    float sa = bandY ? a.baselink_center.x : a.baselink_center.y;
                    float sb = bandY ? b.baselink_center.x : b.baselink_center.y;
                    if (std::fabs(sa - sb) > 1e-4f) return sa < sb;
                    return a.direction < b.direction;
                });
            // 奇数带反向 (boustrophedon)
            {
                size_t i = 0;
                while (i < g.fill.size()) {
                    float ci = bandY ? g.fill[i].baselink_center.y
                                     : g.fill[i].baselink_center.x;
                    int bi = static_cast<int>(std::floor((ci - axis_min) / band_w));
                    size_t j = i;
                    while (j < g.fill.size()) {
                        float cj = bandY ? g.fill[j].baselink_center.y
                                         : g.fill[j].baselink_center.x;
                        int bj = static_cast<int>(std::floor((cj - axis_min) / band_w));
                        if (bj != bi) break;
                        ++j;
                    }
                    if (bi % 2 != 0)
                        std::reverse(g.fill.begin() + i, g.fill.begin() + j);
                    i = j;
                }
            }

            // ring 旋转使环起点靠近 fill 入口
            if (!g.ring.empty() && !g.fill.empty()) {
                cv::Point2f fin = g.fill.front().baselink_center;
                size_t best = 0; float bd = 1e18f;
                for (size_t k = 0; k < g.ring.size(); ++k) {
                    float dx = g.ring[k].baselink_center.x - fin.x;
                    float dy = g.ring[k].baselink_center.y - fin.y;
                    float d = dx * dx + dy * dy;
                    if (d < bd) { bd = d; best = k; }
                }
                std::rotate(g.ring.begin(), g.ring.begin() + best, g.ring.end());
            }

            g.ordered = g.ring;
            g.ordered.insert(g.ordered.end(), g.fill.begin(), g.fill.end());
            if (!g.ordered.empty()) {
                g.entry = g.ordered.front().baselink_center;
                g.exit  = g.ordered.back().baselink_center;
            }
        }
        // 去掉空区域
        regs.erase(std::remove_if(regs.begin(), regs.end(),
            [](const RegionGroup& g) { return g.ordered.empty(); }), regs.end());

        // ── Pass C: 区域间链接 (贪心NN + 2-opt) ────────
        int R = static_cast<int>(regs.size());
        std::vector<int> order;
        std::vector<bool> flip(R, false);
        if (R > 0) {
            std::vector<bool> used(R, false);
            // 起点: entry/exit 最小 X 的区域
            int start = 0; float minx = 1e18f;
            for (int i = 0; i < R; ++i) {
                float mx = std::min(regs[i].entry.x, regs[i].exit.x);
                if (mx < minx) { minx = mx; start = i; }
            }
            bool startFlip = regs[start].exit.x < regs[start].entry.x;
            order.push_back(start); used[start] = true; flip[start] = startFlip;
            cv::Point2f tail = startFlip ? regs[start].entry : regs[start].exit;
            for (int step = 1; step < R; ++step) {
                int best = -1; float bd = 1e18f; bool bflip = false;
                for (int i = 0; i < R; ++i) {
                    if (used[i]) continue;
                    float de = std::hypot(regs[i].entry.x - tail.x,
                                          regs[i].entry.y - tail.y);
                    float dx = std::hypot(regs[i].exit.x - tail.x,
                                          regs[i].exit.y - tail.y);
                    if (de < bd) { bd = de; best = i; bflip = false; }
                    if (dx < bd) { bd = dx; best = i; bflip = true; }
                }
                used[best] = true; order.push_back(best); flip[best] = bflip;
                tail = bflip ? regs[best].entry : regs[best].exit;
            }

            // 端点取用辅助
            auto seg_start = [&](int idx) {
                return flip[order[idx]] ? regs[order[idx]].exit
                                        : regs[order[idx]].entry;
            };
            auto seg_end = [&](int idx) {
                return flip[order[idx]] ? regs[order[idx]].entry
                                        : regs[order[idx]].exit;
            };
            auto connector_len = [&]() {
                float s = 0;
                for (int i = 1; i < R; ++i) {
                    cv::Point2f a = seg_end(i - 1), b = seg_start(i);
                    s += std::hypot(a.x - b.x, a.y - b.y);
                }
                return s;
            };
            // 2-opt: 反转区间(含朝向翻转), ≤4 趟
            for (int pass = 0; pass < 4; ++pass) {
                bool improved = false;
                float cur_len = connector_len();
                for (int i = 0; i < R - 1; ++i) {
                    for (int k = i + 1; k < R; ++k) {
                        std::reverse(order.begin() + i, order.begin() + k + 1);
                        for (int t = i; t <= k; ++t) flip[order[t]] = !flip[order[t]];
                        float nl = connector_len();
                        if (nl + 1e-3f < cur_len) {
                            cur_len = nl; improved = true;
                        } else {
                            std::reverse(order.begin() + i, order.begin() + k + 1);
                            for (int t = i; t <= k; ++t) flip[order[t]] = !flip[order[t]];
                        }
                    }
                }
                if (!improved) break;
            }
        }

        // ── Pass D: 展平 + 重编号 ────────────────────
        waypoints.clear();
        for (int idx : order) {
            std::vector<Waypoint> seq = regs[idx].ordered;
            if (flip[idx]) std::reverse(seq.begin(), seq.end());
            for (auto& wp : seq) waypoints.push_back(std::move(wp));
        }
        for (size_t i = 0; i < waypoints.size(); ++i)
            waypoints[i].id = static_cast<int>(i);

        std::cout << "Ordering: " << R << " regions, "
                  << waypoints.size() << " waypoints\n";

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
// 可视化 (三色区分 Phase + 方向箭头)
// ═══════════════════════════════════════════════════════════
void CoveragePlanner::save_visualization(const std::string& out_dir,
                                          const cv::Mat& work_area,
                                          const CoverageResult& result,
                                          float u_min, float v_min) const
{
    auto w2p = [&](const cv::Point2f& w) -> cv::Point {
        return world_to_px(w, u_min, v_min);
    };

    // 方向→箭头偏移
    auto dir_arrow = [](int dir) -> cv::Point {
        switch (dir) {
            case 0: return cv::Point(15, 0);   // +X →
            case 1: return cv::Point(-15, 0);  // -X ←
            case 2: return cv::Point(0, 15);   // +Y ↓
            case 3: return cv::Point(0, -15);  // -Y ↑
            default: return cv::Point(15, 0);
        }
    };

    // Phase 颜色: 0=蓝 1=绿 2=黄
    auto phase_fill = [](int phase) -> cv::Scalar {
        switch (phase) {
            case 0: return cv::Scalar(255, 0, 0);    // 蓝色
            case 1: return cv::Scalar(0, 255, 0);    // 绿色
            case 2: return cv::Scalar(0, 255, 255);  // 黄色
            default: return cv::Scalar(0, 255, 0);
        }
    };
    auto phase_border = [](int phase) -> cv::Scalar {
        switch (phase) {
            case 0: return cv::Scalar(200, 0, 0);
            case 1: return cv::Scalar(0, 200, 0);
            case 2: return cv::Scalar(0, 200, 200);
            default: return cv::Scalar(0, 200, 0);
        }
    };

    // ── 图1: base 覆盖区域 (三色) + 方向箭头 ─────────
    {
        cv::Mat base_img = work_area.clone();
        cv::Mat overlay = base_img.clone();
        for (size_t i = 0; i < result.waypoints.size(); ++i) {
            const auto& wp = result.waypoints[i];
            cv::Point bs_px = w2p(wp.base_center);
            int bs_hx = static_cast<int>(wp.base_x_size / params_.pixel_size / 2);
            int bs_hy = static_cast<int>(wp.base_y_size / params_.pixel_size / 2);
            cv::Rect base_rect(bs_px.x - bs_hx, bs_px.y - bs_hy, bs_hx * 2, bs_hy * 2);
            int phase = (wp.stripe >= 0 && wp.stripe <= 2) ? wp.stripe : 1;
            cv::rectangle(overlay, base_rect, phase_fill(phase), -1);
            cv::rectangle(base_img, base_rect, phase_border(phase), 1);
            // 方向箭头
            cv::Point arrow = dir_arrow(wp.direction);
            cv::arrowedLine(base_img, bs_px, bs_px + arrow,
                            cv::Scalar(0, 0, 0), 1, cv::LINE_8, 0, 0.2);
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
                cv::Point(bl_px.x + bl_hx, bl_px.y + bl_hy), color, 1);
            cv::circle(bl_img, bl_px, 2, color, -1);
            cv::putText(bl_img, std::to_string(i), bl_px + cv::Point(5, -5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1);
            // baselink 方向沿用 base 方向箭头
            cv::Point arrow = dir_arrow(wp.direction);
            cv::arrowedLine(bl_img, bl_px, bl_px + arrow,
                            cv::Scalar(0, 0, 255), 1, cv::LINE_8, 0, 0.2);
        }
        cv::imwrite(out_dir + "/02_baselink_positions.png", bl_img);
    }

    // ── 图3: 路径连线 ──────────────────────────────
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
            if (is_jump) { cv::line(jump_ov, p1, p2, cv::Scalar(0, 255, 255), 3); jump_count++; }
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
            cv::putText(seq_img, std::to_string(r), bl_px + cv::Point(3, -3),
                        cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255), 1);
        }
        cv::imwrite(out_dir + "/04_baselink_sequence.png", seq_img);
    }

    // ── 图5: 综合图 (三色边框 + 方向箭头) ──────────
    {
        cv::Mat all = work_area.clone();
        for (size_t i = 1; i < result.path.size(); ++i)
            cv::line(all, w2p(result.path[i-1]), w2p(result.path[i]),
                     cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
        for (const auto& wp : result.waypoints) {
            cv::Point bs_px = w2p(wp.base_center);
            int bs_hx = static_cast<int>(wp.base_x_size / params_.pixel_size / 2);
            int bs_hy = static_cast<int>(wp.base_y_size / params_.pixel_size / 2);
            int phase = (wp.stripe >= 0 && wp.stripe <= 2) ? wp.stripe : 1;
            cv::rectangle(all,
                cv::Point(bs_px.x - bs_hx, bs_px.y - bs_hy),
                cv::Point(bs_px.x + bs_hx, bs_px.y + bs_hy),
                phase_border(phase), 1);
            cv::Point arrow = dir_arrow(wp.direction);
            cv::arrowedLine(all, bs_px, bs_px + arrow,
                            phase_border(phase), 1, cv::LINE_8, 0, 0.2);
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
          << ",\"dir\":" << wp.direction << ",\"phase\":" << wp.stripe << "}";
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
