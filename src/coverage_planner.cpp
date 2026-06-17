//=============================================================================
// coverage_planner.cpp — Greedy Coverage Path Planning V2
//
// 核心思想:
//   用 base 方块（最大 1.17×1.17）贪心地"占据"灰色（可喷涂）像素。
//   每次放置一个 base 覆盖尽可能多的未覆盖灰色像素，
//   baselink 可全向行驶，仅需满足"最近边界在行进方向左侧"约束。
//
// 算法:
//   1. 扫描工作区 → 找到所有灰色连通域
//   2. 对每个连通域：
//      a. 滑动窗口找最佳 base 位置（覆盖最多未覆盖灰色像素）
//      b. 扩展 base 尺寸（0.05→1.17）适应灰色区域形状
//      c. 确定行进方向（最近边界在左）
//      d. 计算 baselink 位置
//      e. 标记覆盖，重复直到该域无可覆盖像素
//   3. 按最近邻连接所有 waypoint → 路径
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
                                     float u_min, float v_min) const
{
    float hx = bx / 2.0f;
    float hy = by / 2.0f;
    cv::Point tl = world_to_px({base_center.x - hx, base_center.y - hy}, u_min, v_min);
    cv::Point br = world_to_px({base_center.x + hx, base_center.y + hy}, u_min, v_min);

    if (tl.x < 0 || tl.y < 0 || br.x >= work_area.cols || br.y >= work_area.rows)
        return false;

    for (int r = tl.y; r <= br.y; ++r) {
        const cv::Vec3b* row = work_area.ptr<cv::Vec3b>(r);
        for (int c = tl.x; c <= br.x; ++c) {
            if (row[c] != cv::Vec3b(128, 128, 128))  // 必须纯灰
                return false;
        }
    }
    return true;
}

// ── base 自适应扩展 ────────────────────────────────────
void CoveragePlanner::expand_base(const cv::Mat& work_area,
                                   const cv::Point2f& base_center,
                                   float& out_x, float& out_y,
                                   float u_min, float v_min) const
{
    float step = 0.05f;
    float max_dim = params_.base_max;
    bool changed = true;
    while (changed) {
        changed = false;
        if (out_x + step * 2 <= max_dim &&
            is_base_valid(work_area, base_center, out_x + step * 2, out_y, u_min, v_min)) {
            out_x += step * 2; changed = true;
        }
        if (out_y + step * 2 <= max_dim &&
            is_base_valid(work_area, base_center, out_x, out_y + step * 2, u_min, v_min)) {
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
    // direction = +1 → +X行进 → 左侧 = +Y
    // direction = -1 → -X行进 → 左侧 = -Y
    float y_off = (direction > 0) ? 0.5f : -0.5f;
    cv::Point pt = world_to_px({baselink_center.x, baselink_center.y + y_off}, u_min, v_min);

    if (pt.x < 0 || pt.x >= work_area.cols || pt.y < 0 || pt.y >= work_area.rows)
        return false;
    cv::Vec3b p = work_area.at<cv::Vec3b>(pt.y, pt.x);
    // 左侧必须是灰色(有船体)或红色(边界) — 不能是纯黑(区域外)
    return !(p[0] == 0 && p[1] == 0 && p[2] == 0);
}

// ── 统计 base 覆盖的未覆盖灰色像素数 ──────────────────
static int count_uncovered(const cv::Mat& gray_mask,
                           const cv::Point& tl, const cv::Point& br)
{
    int cnt = 0;
    for (int r = tl.y; r <= br.y; ++r) {
        const uint8_t* row = gray_mask.ptr<uint8_t>(r);
        for (int c = tl.x; c <= br.x; ++c)
            if (row[c] > 0) cnt++;
    }
    return cnt;
}

// ═══════════════════════════════════════════════════════════
// 主入口: 贪心占据
// ═══════════════════════════════════════════════════════════
CoverageResult CoveragePlanner::plan(const cv::Mat& work_area,
                                      float u_min, float v_min)
{
    CoverageResult result;
    if (work_area.empty()) return result;

    std::cout << "\n========== Coverage Planning V2 ==========\n";
    int rows = work_area.rows, cols = work_area.cols;

    // ── 1. 提取灰色 mask (未覆盖 = 255) ──────────────
    cv::Mat uncovered(rows, cols, CV_8UC1, cv::Scalar(0));
    cv::Mat gray_mask(rows, cols, CV_8UC1, cv::Scalar(0));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (work_area.at<cv::Vec3b>(r,c) == cv::Vec3b(128,128,128)) {
                uncovered.at<uint8_t>(r,c) = 255;
                gray_mask.at<uint8_t>(r,c) = 255;
            }
        }
    }

    int total_gray = cv::countNonZero(gray_mask);
    std::cout << "Gray pixels: " << total_gray << "\n";
    if (total_gray == 0) return result;

    // ── 2. 规则网格放置 ──────────────────────────
    // Y 行间距 = 1.17m, X 列间距 = 1.17m
    // 对每个网格点尝试放置 base, 有覆盖即保留
    std::vector<Waypoint> waypoints;
    cv::Mat covered = cv::Mat::zeros(rows, cols, CV_8UC1);
    int wp_id = 0;

    // ── 2. 自适应行放置 (基于 Y 投影) ────────────
    // 计算每行 grey 像素数 (Y 投影)
    std::vector<int> y_proj(rows, 0);
    for (int r = 0; r < rows; ++r) {
        const uint8_t* gr = gray_mask.ptr<uint8_t>(r);
        for (int c = 0; c < cols; ++c)
            if (gr[c] > 0) y_proj[r]++;
    }

    // 找到 grey 像素连续带 → 每带中心放一行
    std::vector<int> row_ys;  // 每行的 gy 坐标
    float min_row_gap_px = 0.3f / params_.pixel_size;  // 行间距至少 30cm
    int in_band = 0, band_start = 0;
    for (int r = 0; r < rows; ++r) {
        if (y_proj[r] > 0) {
            if (in_band == 0) band_start = r;
            in_band++;
        } else {
            if (in_band > 0) {
                // 该带内放几行: band 高度 / 最大覆盖(1.17m) → 行数
                float band_h = in_band * params_.pixel_size;
                int n_rows = std::max(1, static_cast<int>(std::ceil(band_h / (params_.base_max * 0.95f))));
                for (int k = 0; k < n_rows; ++k) {
                    float frac = (k + 0.5f) / n_rows;
                    int gy = band_start + static_cast<int>(frac * in_band);
                    // 与前一行间隔检查
                    if (row_ys.empty() || std::abs(gy - row_ys.back()) >= min_row_gap_px)
                        row_ys.push_back(gy);
                }
                in_band = 0;
            }
        }
    }
    // tail band
    if (in_band > 0) {
        float band_h = in_band * params_.pixel_size;
        int n_rows = std::max(1, static_cast<int>(std::ceil(band_h / (params_.base_max * 0.95f))));
        for (int k = 0; k < n_rows; ++k) {
            float frac = (k + 0.5f) / n_rows;
            int gy = band_start + static_cast<int>(frac * in_band);
            if (row_ys.empty() || std::abs(gy - row_ys.back()) >= min_row_gap_px)
                row_ys.push_back(gy);
        }
    }
    std::cout << "Adaptive rows: " << row_ys.size() << "\n";

    float grid_step_x = params_.base_max * 0.8f;  // ~0.94m 列间距 (20% 重叠)
    int grid_px_x = static_cast<int>(grid_step_x / params_.pixel_size);

    // 找到工作区范围
    int x_min_px = cols, x_max_px = 0, y_min_px = rows, y_max_px = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (gray_mask.at<uint8_t>(r,c) > 0) {
                x_min_px = std::min(x_min_px, c); x_max_px = std::max(x_max_px, c);
                y_min_px = std::min(y_min_px, r); y_max_px = std::max(y_max_px, r);
            }
        }
    }

    // 按自适应行放置
    int row_idx = 0;
    for (int gy : row_ys) {
        std::cout << "  Row " << (++row_idx) << "/" << row_ys.size() << "\r" << std::flush;
        int gx = x_min_px;
        while (gx <= x_max_px) {
            while (gx <= x_max_px && uncovered.at<uint8_t>(gy, gx) == 0) gx++;
            if (gx > x_max_px) break;
            cv::Point grid_pt(gx, gy);
            if (grid_pt.x < 0 || grid_pt.x >= cols || grid_pt.y < 0 || grid_pt.y >= rows)
                continue;
            if (uncovered.at<uint8_t>(gy, gx) == 0) continue;

            // base 覆盖 = 1.17×1.17 框内所有灰色像素 (任意形状)
            cv::Point2f bc_w = px_to_world({gx, gy}, u_min, v_min);
            int half_px = static_cast<int>(params_.base_max / 2.0f / params_.pixel_size);
            cv::Rect roi(gx - half_px, gy - half_px, half_px * 2, half_px * 2);
            roi &= cv::Rect(0, 0, cols, rows);

            // 框内灰色像素 = 实际覆盖
            int new_cnt = 0;
            int min_x = cols, max_x = 0, min_y = rows, max_y = 0;
            for (int rr = roi.y; rr < roi.y + roi.height; ++rr) {
                const uint8_t* ur = uncovered.ptr<uint8_t>(rr);
                const cv::Vec3b* wr = work_area.ptr<cv::Vec3b>(rr);
                for (int cc = roi.x; cc < roi.x + roi.width; ++cc) {
                    if (ur[cc] > 0 && wr[cc] == cv::Vec3b(128, 128, 128)) {
                        new_cnt++;
                        min_x = std::min(min_x, cc); max_x = std::max(max_x, cc);
                        min_y = std::min(min_y, rr); max_y = std::max(max_y, rr);
                    }
                }
            }
            if (new_cnt < 1) { gx += 1; continue; }

            // base 框内不能碰红色边界
            bool base_touches_red = false;
            for (int rr = roi.y; rr < roi.y + roi.height && !base_touches_red; ++rr) {
                const cv::Vec3b* wr = work_area.ptr<cv::Vec3b>(rr);
                for (int cc = roi.x; cc < roi.x + roi.width && !base_touches_red; ++cc)
                    if (wr[cc] == cv::Vec3b(0, 0, 255)) base_touches_red = true;
            }
            if (base_touches_red) { gx += 1; continue; }

            float bx = (max_x - min_x + 1) * params_.pixel_size;
            float by = (max_y - min_y + 1) * params_.pixel_size;

            // 确定行进方向 + baselink 碰撞检测
            // 先测 base 到最近红色边界的最小距离
            bool near_boundary = false;
            int dist_px = static_cast<int>(0.6f / params_.pixel_size);
            cv::Point bc_px = world_to_px(bc_w, u_min, v_min);
            for (int rr = std::max(0, bc_px.y - dist_px);
                 rr <= std::min(rows-1, bc_px.y + dist_px) && !near_boundary; ++rr) {
                const cv::Vec3b* row = work_area.ptr<cv::Vec3b>(rr);
                for (int cc = std::max(0, bc_px.x - dist_px);
                     cc <= std::min(cols-1, bc_px.x + dist_px) && !near_boundary; ++cc)
                    if (row[cc] == cv::Vec3b(0, 0, 255)) near_boundary = true;
            }

            int best_dir = 0;
            cv::Point2f best_bl;
            for (int d : {+1, -1}) {
                cv::Point2f bl_ctr(bc_w.x - params_.center_offset * d, bc_w.y);
                // baselink 不能碰红色
                bool bl_touches_red = false;
                float bl_hx = params_.baselink_x/2, bl_hy = params_.baselink_y/2;
                cv::Point tl = world_to_px({bl_ctr.x-bl_hx,bl_ctr.y-bl_hy}, u_min, v_min);
                cv::Point br = world_to_px({bl_ctr.x+bl_hx,bl_ctr.y+bl_hy}, u_min, v_min);
                for (int rr=std::max(0,tl.y); rr<=std::min(rows-1,br.y)&&!bl_touches_red; ++rr)
                    for (int cc=std::max(0,tl.x); cc<=std::min(cols-1,br.x)&&!bl_touches_red; ++cc)
                        if (work_area.at<cv::Vec3b>(rr,cc) == cv::Vec3b(0,0,255)) bl_touches_red=true;
                if (bl_touches_red) continue;

                // 仅在 base 距边界 < 0.6m 时强制 pier-left
                if (!near_boundary || check_pier_left(work_area, bl_ctr, d, u_min, v_min)) {
                    best_dir = d; best_bl = bl_ctr; break;
                }
            }
            if (best_dir == 0) { gx += 1; continue; }

            Waypoint wp;
            wp.id = wp_id++;
            wp.base_center = bc_w;
            wp.baselink_center = best_bl;
            // 强制 baselink Y = 行中心 (同行所有 waypoint 对齐)
            float row_y = px_to_world({0, gy}, u_min, v_min).y;
            wp.baselink_center.y = row_y;
            wp.base_center.y     = row_y;   // ★ base 与 baselink 刚性连接, Y 必须一致
            wp.base_x_size = bx;
            wp.base_y_size = by;
            wp.stripe = 0;
            wp.direction = best_dir;
            waypoints.push_back(wp);

            // 只标记灰色像素为已覆盖 (任意形状)
            for (int rr = roi.y; rr < roi.y + roi.height; ++rr)
                for (int cc = roi.x; cc < roi.x + roi.width; ++cc)
                    if (work_area.at<cv::Vec3b>(rr, cc) == cv::Vec3b(128, 128, 128))
                        { uncovered.at<uint8_t>(rr, cc) = 0; covered.at<uint8_t>(rr, cc) = 255; }

            // 前进: 基于实际覆盖范围的右边界, 确保不漏空隙
            // (原来 gx += half_px*2 固定跳步, 当灰色区域窄时会留下大段空白)
            gx = max_x + 1;
        }
    }

    // ── 3b. 第二遍: 贪心覆盖剩余灰色区域 ─────────
    int pass2_added = 0;
    while (cv::countNonZero(uncovered) > 0) {
        // 动态缩小扫描范围: 只扫描未覆盖像素的 bounding box
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
        if (u_min_x > u_max_x) break;  // 无未覆盖像素

        // 扫描找未覆盖灰色像素最多的位置
        int best_x = -1, best_y = -1, best_cnt = 0;
        float best_bx = 0, best_by = 0;
        int scan_step = std::max(1, static_cast<int>(0.1f / params_.pixel_size));

        for (int r = u_min_y; r <= u_max_y; r += scan_step) {
            const uint8_t* urow = uncovered.ptr<uint8_t>(r);
            for (int c = u_min_x; c <= u_max_x; c += scan_step) {
                if (urow[c] == 0) continue;
                cv::Point2f bc_w = px_to_world({c, r}, u_min, v_min);
                float bx = 0.5f, by = 0.5f;
                expand_base(work_area, bc_w, bx, by, u_min, v_min);
                int hx = static_cast<int>(bx/2/params_.pixel_size);
                int hy = static_cast<int>(by/2/params_.pixel_size);
                cv::Point tl(std::max(0,c-hx), std::max(0,r-hy));
                cv::Point br(std::min(cols-1,c+hx), std::min(rows-1,r+hy));
                int cnt = count_uncovered(uncovered, tl, br);
                if (cnt > best_cnt) { best_cnt=cnt; best_x=c; best_y=r; best_bx=bx; best_by=by; }
            }
        }
        if (best_cnt < 1) break;  // 剩余碎片忽略

        cv::Point2f bc_w = px_to_world({best_x, best_y}, u_min, v_min);
        int best_dir = 0;
        cv::Point2f best_bl;

        // 测 base 到最近红色边界距离
        bool near_boundary2 = false;
        int dist_px2 = static_cast<int>(0.6f / params_.pixel_size);
        cv::Point bc_px2 = world_to_px(bc_w, u_min, v_min);
        for (int rr=std::max(0,bc_px2.y-dist_px2); rr<=std::min(rows-1,bc_px2.y+dist_px2)&&!near_boundary2; ++rr)
            for (int cc=std::max(0,bc_px2.x-dist_px2); cc<=std::min(cols-1,bc_px2.x+dist_px2)&&!near_boundary2; ++cc)
                if (work_area.at<cv::Vec3b>(rr,cc) == cv::Vec3b(0,0,255)) near_boundary2=true;

        for (int d : {+1, -1}) {
            cv::Point2f bl_ctr(bc_w.x - params_.center_offset * d, bc_w.y);
            bool bl_red = false;
            float bl_hx = params_.baselink_x/2, bl_hy = params_.baselink_y/2;
            cv::Point tl = world_to_px({bl_ctr.x-bl_hx,bl_ctr.y-bl_hy}, u_min, v_min);
            cv::Point br = world_to_px({bl_ctr.x+bl_hx,bl_ctr.y+bl_hy}, u_min, v_min);
            for (int rr=std::max(0,tl.y); rr<=std::min(rows-1,br.y)&&!bl_red; ++rr)
                for (int cc=std::max(0,tl.x); cc<=std::min(cols-1,br.x)&&!bl_red; ++cc)
                    if (work_area.at<cv::Vec3b>(rr,cc) == cv::Vec3b(0,0,255)) bl_red=true;
            if (bl_red) continue;
            if (!near_boundary2 || check_pier_left(work_area, bl_ctr, d, u_min, v_min)) {
                best_dir=d; best_bl=bl_ctr; break;
            }
        }
        if (best_dir == 0) break;

        Waypoint wp;
        wp.id = wp_id++;
        wp.base_center = bc_w;
        wp.baselink_center = best_bl;
        wp.base_x_size = best_bx; wp.base_y_size = best_by;

        // 对齐到最近自适应行 — 先验证对齐后 base 仍然合法
        float snap_y = best_bl.y;  // 默认不修改
        float min_d = 1e9;
        for (int gy : row_ys) {
            float ry = px_to_world({0, gy}, u_min, v_min).y;
            float d = std::fabs(best_bl.y - ry);
            if (d < min_d) { min_d = d; snap_y = ry; }
        }
        // 试探对齐位置: 更新 base 和 baselink 的 Y, 检查 base 是否仍在灰色区域内
        cv::Point2f snapped_bc = bc_w;
        snapped_bc.y = snap_y;
        if (is_base_valid(work_area, snapped_bc, best_bx, best_by, u_min, v_min)) {
            wp.base_center.y     = snap_y;
            wp.baselink_center.y = snap_y;
        }
        // 否则保留原始 Y (不对齐, 后处理蛇形排序时会按 tolerance 自动分组)
        wp.stripe = 99; wp.direction = best_dir;
        waypoints.push_back(wp);
        pass2_added++;

        int hx = static_cast<int>(best_bx/2/params_.pixel_size);
        int hy = static_cast<int>(best_by/2/params_.pixel_size);
        cv::Rect roi(best_x-hx, best_y-hy, hx*2, hy*2);
        roi &= cv::Rect(0,0,cols,rows);
        // 只标记灰色像素为已覆盖 (与 pass 1 一致, 避免 inflate coverage ratio)
        for (int rr = roi.y; rr < roi.y + roi.height; ++rr)
            for (int cc = roi.x; cc < roi.x + roi.width; ++cc)
                if (work_area.at<cv::Vec3b>(rr, cc) == cv::Vec3b(128, 128, 128))
                    { uncovered.at<uint8_t>(rr, cc) = 0; covered.at<uint8_t>(rr, cc) = 255; }
    }

    result.waypoints = waypoints;
    result.total_waypoints = waypoints.size();
    std::cout << "Grid waypoints + pass2: " << pass2_added << " extra\n";

    // ── 3. 后处理: 按行蛇形重排, 过滤孤立行 ──
    if (waypoints.size() > 1) {
        std::sort(waypoints.begin(), waypoints.end(),
            [](const Waypoint& a, const Waypoint& b) {
                if(std::fabs(a.baselink_center.y-b.baselink_center.y)>0.3f)
                    return a.baselink_center.y < b.baselink_center.y;
                return a.baselink_center.x < b.baselink_center.x;
            });
        std::vector<std::vector<Waypoint>> rows;
        std::vector<Waypoint> cur; float ly=waypoints[0].baselink_center.y-2;
        for(auto& wp:waypoints){if(std::fabs(wp.baselink_center.y-ly)>0.3f){if(!cur.empty())rows.push_back(std::move(cur));cur.clear();ly=wp.baselink_center.y;}cur.push_back(std::move(wp));}
        if(!cur.empty())rows.push_back(std::move(cur));

        // 过滤孤立行 (≤2个点, 且上下行都有很多点时 → 合并到最近行)
        std::vector<std::vector<Waypoint>> clean_rows;
        for(size_t r=0;r<rows.size();++r){
            if(rows[r].size()<=2 && rows.size()>3){
                // 合并到最近的非孤立行
                float best_d=1e9; int best_ri=-1;
                for(size_t rr=0;rr<rows.size();++rr){if(rr==r||rows[rr].size()<=2)continue;
                    float d=std::fabs(rows[r][0].baselink_center.y-rows[rr][0].baselink_center.y);
                    if(d<best_d){best_d=d;best_ri=static_cast<int>(rr);}}
                if(best_ri>=0){for(auto& wp:rows[r])rows[best_ri].push_back(std::move(wp));}
                else clean_rows.push_back(std::move(rows[r]));
            }else clean_rows.push_back(std::move(rows[r]));
        }

        waypoints.clear();
        for(size_t r=0;r<clean_rows.size();++r){
            std::sort(clean_rows[r].begin(),clean_rows[r].end(),[](const Waypoint&a,const Waypoint&b){return a.baselink_center.x<b.baselink_center.x;});
            if(r%2==1)std::reverse(clean_rows[r].begin(),clean_rows[r].end());
            for(auto& wp:clean_rows[r]) waypoints.push_back(std::move(wp));
        }
        for(size_t i=0;i<waypoints.size();++i) waypoints[i].id=static_cast<int>(i);

        // 生成路径
        std::vector<cv::Point2f> path;
        const float max_gap = 1.4f;
        for (const auto& wp : waypoints) {
            cv::Point2f pt = wp.baselink_center;
            if (!path.empty()) {
                cv::Point2f& prev = path.back();
                float dx = pt.x - prev.x, dy = pt.y - prev.y, d = std::sqrt(dx*dx+dy*dy);
                int sp = std::max(1, static_cast<int>(std::ceil(d/max_gap)));
                for (int s = 1; s <= sp; ++s) { float t=(float)s/sp; path.push_back({prev.x+dx*t,prev.y+dy*t}); }
            } else path.push_back(pt);
        }

        result.path = path;
        result.total_path_length_m = 0;
        for (size_t i = 1; i < path.size(); ++i) {
            float dx = path[i].x - path[i-1].x;
            float dy = path[i].y - path[i-1].y;
            result.total_path_length_m += std::sqrt(dx*dx + dy*dy);
        }
    }

    // ── 5. 覆盖率 ──────────────────────────────
    int covered_cnt = cv::countNonZero(covered);
    result.coverage_ratio = total_gray > 0
        ? static_cast<float>(covered_cnt) / total_gray : 0.0f;
    result.overlap_ratio = 0.0f;  // TODO

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
            cv::rectangle(overlay, base_rect, cv::Scalar(0, 255, 0), -1);  // 实心绿
            cv::rectangle(base_img, base_rect, cv::Scalar(0, 200, 0), 1);  // 深绿边框
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
            cv::Scalar color = cv::Scalar(255, 0, 0);  // 蓝 = baselink

            cv::rectangle(bl_img,
                cv::Point(bl_px.x - bl_hx, bl_px.y - bl_hy),
                cv::Point(bl_px.x + bl_hx, bl_px.y + bl_hy),
                color, 1);
            cv::circle(bl_img, bl_px, 2, color, -1);
            cv::putText(bl_img, std::to_string(i), bl_px + cv::Point(5, -5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1);

            // 行进方向箭头
            cv::Point arrow_tip(bl_px.x + wp.direction * 15, bl_px.y);
            cv::arrowedLine(bl_img, bl_px, arrow_tip, cv::Scalar(0, 0, 255), 1, cv::LINE_8, 0, 0.2);
        }
        cv::imwrite(out_dir + "/02_baselink_positions.png", bl_img);
    }

    // ── 图3: 路径连线 (跳变段半透明黄色标注) ──
    {
        cv::Mat path_img = work_area.clone();
        cv::Mat jump_ov = path_img.clone();

        // 计算中位步长用于跳变检测
        std::vector<float> seg_lens;
        for (size_t i = 1; i < result.path.size(); ++i) {
            float dx = result.path[i].x - result.path[i-1].x;
            float dy = result.path[i].y - result.path[i-1].y;
            seg_lens.push_back(std::sqrt(dx*dx + dy*dy));
        }
        std::sort(seg_lens.begin(), seg_lens.end());
        float median_step = seg_lens.empty() ? 1.0f : seg_lens[seg_lens.size()/2];
        float jump_thresh = std::max(3.0f, median_step * 3.0f);  // 3m 或 3x中位步长

        int jump_count = 0;
        for (size_t i = 1; i < result.path.size(); ++i) {
            cv::Point p1 = w2p(result.path[i-1]);
            cv::Point p2 = w2p(result.path[i]);
            float dx = result.path[i].x - result.path[i-1].x;
            float dy = result.path[i].y - result.path[i-1].y;
            float d = std::sqrt(dx*dx + dy*dy);
            bool is_jump = (d > jump_thresh);

            cv::Scalar color = is_jump ? cv::Scalar(0, 255, 255)   // 黄色=跳变
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

        // 半透明叠加跳变段
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
        // 按 path_rows 顺序给每个 baselink 赋序号
        for (size_t r = 0; r < result.waypoints.size(); ++r) {
            // waypoints 已按路径顺序排列 (path_rows 顺序)
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
        // 路径底色
        for (size_t i = 1; i < result.path.size(); ++i)
            cv::line(all, w2p(result.path[i-1]), w2p(result.path[i]),
                     cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
        // base 框 (细线)
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
