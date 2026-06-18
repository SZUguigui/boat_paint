//=============================================================================
// main.cpp — 船底地面点剔除程序入口
//
// 目标:
//   从桥梁扫描PCD点云中, 利用法线方向 + 迭代RANSAC平面拟合,
//   移除弧形船底 (ground), 保留桥墩等结构 (non-ground).
//
// 使用:
//   pier_detect.exe <input.pcd> [options]
//
//   选项:
//     --skip-ground        跳过地面剔除, 直接做边界提取 (用于已处理过的点云)
//     --normal-radius N    法线搜索半径, 米 (默认: 0.10)
//     --normal-z-thresh N  法线Z分量阈值, |nz|>此值视为地面候选 (默认: 0.70)
//     --plane-threshold N  RANSAC平面拟合容忍度, 米 (默认: 0.10)
//     --plane-iters N      RANSAC最大迭代次数, 每平面 (默认: 500)
//     --plane-rounds N     迭代RANSAC轮数 (默认: 5)
//     --min-inliers N      最小内点数, 低于此值停止 (默认: 1000)
//     --output DIR         输出目录 (默认: output)
//     --help               显示帮助
//
// 输出:
//   output/non_ground.pcd  — 移除地面后的点云 (仅非 skip 模式)
//   output/ground.pcd      — 被移除的地面点     (仅非 skip 模式)
//   output/boundaries.png  — 边界可视化图像
//
// 依赖:
//   PCL 1.13+ (io, features, filters, segmentation, search)
//   OpenCV 4+  (可视化)
//=============================================================================
//这是一个branch的测试
#include "ground_remover.h"
#include "boundary_extractor.h"
#include "line_fitter.h"
#include "coverage_planner.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/common/common.h>
#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>

// ── 打印使用说明 ─────────────────────────────────────
void print_usage(const char* prog)
{
    std::cout << "Usage: " << prog << " <input.pcd> [options]\n\n"
              << "Options:\n"
              << "  --normal-radius N    Normal estimation search radius, m (default: 0.10)\n"
              << "  --normal-z-thresh N  |nz| > threshold → ground candidate (default: 0.70)\n"
              << "  --plane-threshold N  RANSAC plane distance tolerance, m (default: 0.10)\n"
              << "  --plane-iters N      RANSAC max iterations per plane (default: 500)\n"
              << "  --plane-rounds N     Max number of RANSAC planes to extract (default: 5)\n"
              << "  --min-inliers N      Min inliers per plane, stop below this (default: 1000)\n"
              << "  --output DIR         Output directory (default: output)\n"
              << "  --skip-ground        Skip ground removal, run boundary extraction directly\n"
              << "  --ground-pcd PATH    Load ground PCD for work area (overrides ground removal)\n"
              << "  --help               Show this help\n\n"
              << "Output files:\n"
              << "  output/non_ground.pcd - After ground removal  (unless --skip-ground)\n"
              << "  output/ground.pcd     - Removed ground points (unless --skip-ground)\n"
              << "  output/boundaries.png - Boundary visualization\n";
}

// ── 打印点云信息 ─────────────────────────────────────
void print_cloud_info(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                      const std::string& label)
{
    if (!cloud || cloud->empty()) {
        std::cout << "[" << label << "] Empty\n";
        return;
    }
    Eigen::Vector4f min_pt, max_pt;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);
    std::cout << "[" << label << "] "
              << "points=" << cloud->size()
              << " | X:[" << min_pt[0] << ", " << max_pt[0] << "]"
              << " Y:[" << min_pt[1] << ", " << max_pt[1] << "]"
              << " Z:[" << min_pt[2] << ", " << max_pt[2] << "]\n";
}

// ── 入口 ────────────────────────────────────────────
int main(int argc, char* argv[])
{
    if (argc < 2 || std::string(argv[1]) == "--help") {
        print_usage(argv[0]);
        return (argc < 2) ? 1 : 0;
    }

    // ── 解析参数 ─────────────────────────────────
    std::string pcd_path = argv[1];
    GroundRemovalParams params;
    std::string output_dir = "output";
    bool skip_ground = false;
    bool no_side_links = true;   // 默认不画侧墩连线
    std::string ground_pcd_path;  // 单独指定工作区域 PCD

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        try {
            if (arg == "--skip-ground")
                skip_ground = true;
            else if (arg == "--no-side-links")
                no_side_links = true;
            else if (arg == "--ground-pcd" && i + 1 < argc)
                ground_pcd_path = argv[++i];
            else if (arg == "--normal-radius" && i + 1 < argc)
                params.normal_radius = std::atof(argv[++i]);
            else if (arg == "--normal-z-thresh" && i + 1 < argc)
                params.normal_z_threshold = std::atof(argv[++i]);
            else if (arg == "--plane-threshold" && i + 1 < argc)
                params.plane_threshold = std::atof(argv[++i]);
            else if (arg == "--plane-iters" && i + 1 < argc)
                params.plane_max_iters = std::atoi(argv[++i]);
            else if (arg == "--plane-rounds" && i + 1 < argc)
                params.plane_max_iterations = std::atoi(argv[++i]);
            else if (arg == "--min-inliers" && i + 1 < argc)
                params.plane_min_inliers = std::atoi(argv[++i]);
            else if (arg == "--output" && i + 1 < argc)
                output_dir = argv[++i];
            else {
                std::cerr << "Unknown option: " << arg << "\n";
                print_usage(argv[0]);
                return 1;
            }
        } catch (...) {
            std::cerr << "Invalid value for " << arg << "\n";
            return 1;
        }
    }

    // ── 打印配置 ─────────────────────────────────
    std::cout << "========== Configuration ==========\n"
              << "PCD path:             " << pcd_path << "\n"
              << "Skip ground:          " << (skip_ground ? "yes" : "no") << "\n"
              << "Normal radius:        " << params.normal_radius << " m\n"
              << "Normal Z threshold:   " << params.normal_z_threshold << "\n"
              << "Plane threshold:      " << params.plane_threshold << " m\n"
              << "Plane max iters:      " << params.plane_max_iters << "\n"
              << "Plane rounds:         " << params.plane_max_iterations << "\n"
              << "Plane min inliers:    " << params.plane_min_inliers << "\n"
              << "Output dir:           " << output_dir << "\n"
              << "====================================\n";

    // ── 加载点云 ─────────────────────────────────
    std::cout << "\nLoading PCD: " << pcd_path << "...\n";
    auto raw = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    if (pcl::io::loadPCDFile(pcd_path, *raw) == -1) {
        std::cerr << "ERROR: Failed to load PCD file.\n";
        return 1;
    }

    // 移除 NaN
    std::vector<int> nan_idx;
    pcl::removeNaNFromPointCloud(*raw, *raw, nan_idx);
    print_cloud_info(raw, "raw");

    // ── 执行地面剔除 (除非跳过) ─────────────────
    std::string cmd = "mkdir \"" + output_dir + "\" 2>nul";
    system(cmd.c_str());

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_for_boundary;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_work_area;  // 工作区域 (地面)

    // ── 加载工作区域点云 (如果指定了 --ground-pcd) ──
    if (!ground_pcd_path.empty()) {
        std::cout << "\nLoading ground PCD for work area: " << ground_pcd_path << "...\n";
        cloud_work_area.reset(new pcl::PointCloud<pcl::PointXYZI>);
        if (pcl::io::loadPCDFile(ground_pcd_path, *cloud_work_area) == -1) {
            std::cerr << "ERROR: Failed to load ground PCD.\n";
            return 1;
        }
        std::vector<int> nan_idx;
        pcl::removeNaNFromPointCloud(*cloud_work_area, *cloud_work_area, nan_idx);
        print_cloud_info(cloud_work_area, "work_area");
    }

    if (skip_ground) {
        std::cout << "\nSkipping ground removal.\n";
        cloud_for_boundary = raw;
        if (!cloud_work_area) cloud_work_area = raw;
    } else {
        auto result = remove_boat_bottom(raw, params);
        cloud_work_area = result.ground;  // ★ 地面 = 工作区域

        std::string non_ground_path = output_dir + "/non_ground.pcd";
        std::string ground_path     = output_dir + "/ground.pcd";

        pcl::io::savePCDFileBinary(non_ground_path, *result.non_ground);
        pcl::io::savePCDFileBinary(ground_path,     *result.ground);

        std::cout << "\nResults saved:\n"
                  << "  " << non_ground_path << " (" << result.non_ground->size() << " pts)\n"
                  << "  " << ground_path     << " (" << result.ground->size() << " pts)\n";

        float kept_pct = raw->size() > 0
            ? (100.0f * result.non_ground->size() / raw->size()) : 0.0f;
        float removed_pct = raw->size() > 0
            ? (100.0f * result.ground->size() / raw->size()) : 0.0f;

        std::cout << "\n--- Ground Removal Summary ---\n"
                  << "Total:      " << raw->size() << " pts\n"
                  << "Kept:       " << result.non_ground->size()
                  << " pts (" << kept_pct << "%)\n"
                  << "Removed:    " << result.ground->size()
                  << " pts (" << removed_pct << "%)\n";

        cloud_for_boundary = result.non_ground;
    }

    // ── 边界提取 ──────────────────────────────────
    BoundaryParams boundary_params;
    boundary_params.method = BoundaryMethod::FINDCONTOURS;
    BoundaryExtractor extractor(boundary_params);
    auto boundaries = extractor.extract(cloud_for_boundary);

    // ── 直线拟合 ──────────────────────────────────
    LineFitterParams lf_params;
    LineFitter fitter(lf_params);
    auto fitted_lines = fitter.fit(boundaries);

    // 两侧桥墩邻居连接
    auto side_results = fitter.connect_neighbors(boundaries, fitted_lines);
    auto connections = side_results;  // 全部 (OBB边 + 连线)
    if (no_side_links) {
        // 仅删连线, 保留 OBB 边
        connections.erase(
            std::remove_if(connections.begin(), connections.end(),
                [](const FittedLine& l) { return l.edge_type.rfind("connect_", 0) == 0; }),
            connections.end());
    }

    // 保存 CSV (fit + connect)
    {
        std::ofstream line_file(output_dir + "/fitted_lines.csv");
        line_file << "id,type,a,b,c,start_x,start_y,end_x,end_y,n_piers,inlier\n";
        auto write_lines = [&](const std::vector<FittedLine>& lines, int& id) {
            for (const auto& l : lines) {
                line_file << id++ << "," << l.edge_type << ","
                          << l.a << "," << l.b << "," << l.c << ","
                          << l.start.x << "," << l.start.y << ","
                          << l.end.x << "," << l.end.y << ","
                          << l.boundary_ids.size() << ","
                          << l.inlier_ratio << "\n";
            }
        };
        int id = 0;
        write_lines(fitted_lines, id);
        write_lines(connections, id);
        line_file.close();
    }

    // 可视化: 边界 + 拟合线 + 邻居连线 叠加在同一张图
    auto all_lines = fitted_lines;
    all_lines.insert(all_lines.end(), connections.begin(), connections.end());
    const auto& raster = extractor.get_raster_data();
    fitter.save_visualization(
        output_dir + "/combined_edges_lines.png",
        raster.occupancy, boundaries, all_lines,
        raster.u_min, raster.v_min, boundary_params.pixel_size);
    std::cout << "  Saved: " << output_dir + "/combined_edges_lines.png\n";

    // ── 生成工作区域图 ──────────────────────────────
    // 白色 = work area (ground convex hull)
    // 红色 = boundary lines
    // 灰色 = paintable area (white - closed red regions)
    {
        float ps = boundary_params.pixel_size;

        // 计算 ground 点云 XY bounding box
        float gu_min = 1e9, gv_min = 1e9, gu_max = -1e9, gv_max = -1e9;
        for (const auto& p : cloud_work_area->points) {
            gu_min = std::min(gu_min, p.x); gu_max = std::max(gu_max, p.x);
            gv_min = std::min(gv_min, p.y); gv_max = std::max(gv_max, p.y);
        }
        int g_cols = static_cast<int>((gu_max - gu_min) / ps) + 1;
        int g_rows = static_cast<int>((gv_max - gv_min) / ps) + 1;
        g_cols = std::max(1, std::min(g_cols, 20000));
        g_rows = std::max(1, std::min(g_rows, 20000));

        auto gw2p = [&](const cv::Point2f& w) -> cv::Point {
            return cv::Point(
                static_cast<int>((w.x - gu_min) / ps + 0.5f),
                static_cast<int>((w.y - gv_min) / ps + 0.5f));
        };
        auto p2w = [&](const cv::Point& px) -> cv::Point2f {
            return {gu_min + px.x * ps, gv_min + px.y * ps};
        };

        cv::Mat work_area = cv::Mat::zeros(g_rows, g_cols, CV_8UC3);

        // 1. 地面点 → 凸包 → 填充白色
        std::vector<cv::Point2f> gpts;
        gpts.reserve(cloud_work_area->size());
        for (const auto& p : cloud_work_area->points)
            gpts.emplace_back(p.x, p.y);

        std::vector<cv::Point2f> hull;
        cv::convexHull(gpts, hull);
        std::vector<cv::Point> hull_px;
        for (const auto& p : hull) hull_px.push_back(gw2p(p));
        std::vector<cv::Point> hull_smooth;
        cv::approxPolyDP(hull_px, hull_smooth, 5.0, true);
        std::vector<std::vector<cv::Point>> hull_contour = {hull_smooth};
        cv::drawContours(work_area, hull_contour, 0,
                         cv::Scalar(255, 255, 255), -1);

        // 2. 扩展中轴拟合线到凸包边缘 + 在边缘处封闭
        // 辅助: 求射线与凸包多边形的交点 (沿方向的最远两个)
        auto clip_line_to_hull = [&](const cv::Point2f& origin,
                                      const cv::Point2f& dir) -> std::pair<cv::Point2f, cv::Point2f> {
            float t_min = 1e9, t_max = -1e9;
            cv::Point2f p_min, p_max;
            for (size_t j = 0; j < hull.size(); ++j) {
                cv::Point2f e1 = hull[j], e2 = hull[(j+1)%hull.size()];
                cv::Point2f seg = e2 - e1;
                float cross = dir.x * seg.y - dir.y * seg.x;
                if (std::fabs(cross) < 1e-9f) continue;
                float t = ((e1.x - origin.x) * seg.y - (e1.y - origin.y) * seg.x) / cross;
                float u = ((e1.x - origin.x) * dir.y - (e1.y - origin.y) * dir.x) / cross;
                if (u >= -0.001f && u <= 1.001f) {
                    cv::Point2f pt(origin.x + t * dir.x, origin.y + t * dir.y);
                    if (t < t_min) { t_min = t; p_min = pt; }
                    if (t > t_max) { t_max = t; p_max = pt; }
                }
            }
            return {p_min, p_max};
        };

        std::vector<FittedLine> ext_lines = all_lines;
        cv::Point2f up_start, up_end, lo_start, lo_end;
        for (size_t i = 0; i < fitted_lines.size(); ++i) {
            auto& l = ext_lines[i];
            if (l.edge_type != "upper" && l.edge_type != "lower") continue;

            cv::Point2f dir(-l.b, l.a);
            float len = std::sqrt(dir.x*dir.x + dir.y*dir.y);
            if (len < 1e-6f) continue;
            dir.x /= len; dir.y /= len;

            cv::Point2f mid((l.start.x + l.end.x) * 0.5f,
                            (l.start.y + l.end.y) * 0.5f);
            auto [p1, p2] = clip_line_to_hull(mid, dir);
            l.start = p1; l.end = p2;

            if (l.edge_type == "upper") { up_start = p1; up_end = p2; }
            if (l.edge_type == "lower") { lo_start = p1; lo_end = p2; }
        }
        // 在凸包边缘处封闭
        FittedLine cl, cr;
        cl.edge_type = "close_middle"; cr.edge_type = "close_middle";
        cl.start = up_start; cl.end = lo_start;
        cr.start = up_end;   cr.end = lo_end;
        ext_lines.push_back(cl);
        ext_lines.push_back(cr);

        // 3. 画红色边界线 (LINE_8: 无抗锯齿, 保证颜色纯净)
        for (const auto& l : ext_lines) {
            cv::line(work_area, gw2p(l.start), gw2p(l.end),
                     cv::Scalar(0, 0, 255), 3, cv::LINE_8);
        }

        // 4. 找出红色封闭区域 → 填充 → 从白色中减去 → 灰色 = 待喷涂
        cv::Mat red_mask;
        cv::Mat gray_channel(work_area.rows, work_area.cols, CV_8UC1, cv::Scalar(0));

        // 只取红色通道
        cv::Mat channels[3];
        cv::split(work_area, channels);
        cv::Mat red_only = channels[2] - cv::max(channels[0], channels[1]);
        cv::threshold(red_only, red_only, 200, 255, cv::THRESH_BINARY);

        // 膨胀红色连接断点 (先膨胀再闭运算焊死小缺口)
        cv::Mat kernel_sm = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::Mat kernel_lg = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
        cv::dilate(red_only, red_only, kernel_sm);
        cv::morphologyEx(red_only, red_only, cv::MORPH_CLOSE, kernel_lg);

        // 找红色围成的封闭轮廓 → 填充
        std::vector<std::vector<cv::Point>> red_contours;
        cv::findContours(red_only, red_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat red_filled = cv::Mat::zeros(work_area.rows, work_area.cols, CV_8UC1);
        for (const auto& c : red_contours) {
            if (cv::contourArea(c) > 100.0)  // 忽略碎片
                cv::drawContours(red_filled, std::vector<std::vector<cv::Point>>{c},
                                 0, cv::Scalar(255), -1);
        }

        // 红色封闭区域内部 → 黑色; 外部白色 → 灰色 (待喷涂)
        for (int r = 0; r < work_area.rows; ++r) {
            for (int c = 0; c < work_area.cols; ++c) {
                cv::Vec3b p = work_area.at<cv::Vec3b>(r, c);
                bool is_white = (p[0] == 255 && p[1] == 255 && p[2] == 255);
                bool is_inside = (red_filled.at<uint8_t>(r, c) > 0);
                if (is_white && is_inside) {
                    work_area.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);  // black = 不可达
                } else if (is_white && !is_inside) {
                    work_area.at<cv::Vec3b>(r, c) = cv::Vec3b(128, 128, 128);  // gray = 待喷涂
                    gray_channel.at<uint8_t>(r, c) = 255;
                }
            }
        }

        // 腐蚀灰色区域, 提供碰撞安全边距
        cv::Mat gray_only;
        cv::inRange(work_area, cv::Scalar(128,128,128), cv::Scalar(128,128,128), gray_only);
        cv::Mat erode_k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5));
        cv::erode(gray_only, gray_only, erode_k);
        for (int r = 0; r < work_area.rows; ++r)
            for (int c = 0; c < work_area.cols; ++c)
                if (work_area.at<cv::Vec3b>(r,c) == cv::Vec3b(128,128,128) &&
                    gray_only.at<uint8_t>(r,c) == 0)
                    work_area.at<cv::Vec3b>(r,c) = cv::Vec3b(0,0,0);  // 安全边距 → 黑色

        cv::imwrite(output_dir + "/work_area.png", work_area);
        std::cout << "  Saved: " << output_dir + "/work_area.png"
                  << " (gray=paint, red=boundary, black=inaccessible)\n";

        // ── Coverage Path Planning ───────────────────
        float plan_ps = 0.01f;  // 全分辨率 (0.01m/pixel)
        int scale = static_cast<int>(plan_ps / ps);  // 5x
        cv::Mat wa_small;
        cv::resize(work_area, wa_small, cv::Size(), 1.0/scale, 1.0/scale, cv::INTER_NEAREST);
        CoverageParams cp_params;
        cp_params.pixel_size = plan_ps;
        CoveragePlanner planner(cp_params);
        auto cov_result = planner.plan(wa_small, gu_min, gv_min);
        planner.save_visualization(output_dir, wa_small, cov_result, gu_min, gv_min);
        planner.save_json(output_dir + "/coverage_plan.json", cov_result);
    }

    std::cout << "\n========== Final Summary ==========\n"
              << "Input points:       " << raw->size() << "\n"
              << "After ground removal:" << cloud_for_boundary->size() << "\n"
              << "Boundaries found:   " << boundaries.size() << "\n"
              << "Fitted lines:       " << fitted_lines.size() << "\n"
              << "====================================\n";

    return 0;
}
