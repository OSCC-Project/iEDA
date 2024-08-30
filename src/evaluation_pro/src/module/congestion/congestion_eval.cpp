/*
 * @FilePath: congestion_eval.cpp
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#include "congestion_eval.h"

#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "general_ops.h"
#include "init_egr.h"
#include "wirelength_lut.h"

namespace ieval {

CongestionEval::CongestionEval()
{
}

CongestionEval::~CongestionEval()
{
}

string CongestionEval::evalHoriEGR(string map_path)
{
  return evalEGR(map_path, "horizontal", "egr_horizontal.csv");
}

string CongestionEval::evalVertiEGR(string map_path)
{
  return evalEGR(map_path, "vertical", "egr_vertical.csv");
}

string CongestionEval::evalUnionEGR(string map_path)
{
  return evalEGR(map_path, "union", "egr_union.csv");
}

string CongestionEval::evalHoriRUDY(CongestionNets nets, CongestionRegion region, int32_t grid_size)
{
  return evalRUDY(nets, region, grid_size, "horizontal", "rudy_horizontal.csv");
}

string CongestionEval::evalVertiRUDY(CongestionNets nets, CongestionRegion region, int32_t grid_size)
{
  return evalRUDY(nets, region, grid_size, "vertical", "rudy_vertical.csv");
}

string CongestionEval::evalUnionRUDY(CongestionNets nets, CongestionRegion region, int32_t grid_size)
{
  return evalRUDY(nets, region, grid_size, "union", "rudy_union.csv");
}

string CongestionEval::evalHoriLUTRUDY(CongestionNets nets, CongestionRegion region, int32_t grid_size)
{
  return evalLUTRUDY(nets, region, grid_size, "horizontal", "lut_rudy_horizontal.csv");
}

string CongestionEval::evalVertiLUTRUDY(CongestionNets nets, CongestionRegion region, int32_t grid_size)
{
  return evalLUTRUDY(nets, region, grid_size, "vertical", "lut_rudy_vertical.csv");
}

string CongestionEval::evalUnionLUTRUDY(CongestionNets nets, CongestionRegion region, int32_t grid_size)
{
  return evalLUTRUDY(nets, region, grid_size, "union", "lut_rudy_union.csv");
}

int32_t CongestionEval::evalTotalOverflow(string map_path)
{
  int32_t total_overflow = 0;
  return total_overflow;
}

int32_t CongestionEval::evalMaxOverflow(string map_path)
{
  int32_t max_overflow = 0;
  return max_overflow;
}

float CongestionEval::evalAvgOverflow(string map_path)
{
  float avg_overflow = 0;
  return avg_overflow;
}

float CongestionEval::evalMaxUtilization(string map_path)
{
  float max_utilization = 0;
  return max_utilization;
}

float CongestionEval::evalAvgUtilization(string map_path)
{
  float avg_utilization = 0;
  return avg_utilization;
}

string CongestionEval::reportHotspot(float threshold)
{
  return "hotspot_report.csv";
}

string CongestionEval::reportOverflow(float threshold)
{
  return "overflow_report.csv";
}

string CongestionEval::evalEGR(string map_path, string egr_type, string output_filename)
{
  InitEGR init_egr;
  // init_egr.runEGR();

  std::unordered_map<std::string, LayerDirection> LayerDirections = init_egr.parseLayerDirection(map_path + "/initial_router/route.guide");

  // for (const auto& [layer, direction] : LayerDirections) {
  //   std::cout << "Layer: " << layer << ", Direction: " << (direction == LayerDirection::Horizontal ? "Horizontal" : "Vertical")
  //             << std::endl;
  // }
  std::vector<std::string> targetLayers;
  std::string dirPath = map_path + "/initial_router/";

  if (egr_type == "horizontal" || egr_type == "vertical") {
    LayerDirection targetDirection = (egr_type == "horizontal") ? LayerDirection::Horizontal : LayerDirection::Vertical;

    for (const auto& [layer, direction] : LayerDirections) {
      if (direction == targetDirection) {
        targetLayers.push_back(layer);
      }
    }

    std::vector<std::vector<double>> sumMatrix;
    bool isFirstFile = true;

    for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
      std::string filename = entry.path().filename().string();
      if (filename.find("overflow_map_") != std::string::npos) {
        for (const auto& layer : targetLayers) {
          if (filename.find(layer) != std::string::npos) {
            std::ifstream file(entry.path());
            std::string line;
            size_t row = 0;
            while (std::getline(file, line)) {
              std::istringstream iss(line);
              std::string value;
              int col = 0;
              while (std::getline(iss, value, ',')) {
                double numValue = std::stod(value);
                if (isFirstFile) {
                  if (row >= sumMatrix.size()) {
                    sumMatrix.push_back(std::vector<double>());
                  }
                  sumMatrix[row].push_back(numValue);
                } else {
                  sumMatrix[row][col] += numValue;
                }
                col++;
              }
              row++;
            }
            isFirstFile = false;
            break;
          }
        }
      }
    }

    std::ofstream outFile(dirPath + output_filename);
    for (const auto& row : sumMatrix) {
      for (size_t i = 0; i < row.size(); ++i) {
        outFile << row[i];
        if (i < row.size() - 1) {
          outFile << ",";
        }
      }
      outFile << "\n";
    }
    outFile.close();
    return dirPath + output_filename;
  } else if (egr_type == "union") {
    dirPath = map_path + "/topology_generator/";
    return dirPath + "overflow_map_planar.csv";
  }

  return "";
}

string CongestionEval::evalRUDY(CongestionNets nets, CongestionRegion region, int32_t grid_size, string rudy_type, string output_filename)
{
  int32_t grid_cols = (region.ux - region.lx + grid_size - 1) / grid_size;
  int32_t grid_rows = (region.uy - region.ly + grid_size - 1) / grid_size;

  std::vector<std::vector<double>> density_grid(grid_rows, std::vector<double>(grid_cols, 0.0));

  for (const auto& net : nets) {
    int32_t start_row = grid_rows - 1;
    int32_t end_row = 0;
    int32_t start_col = grid_cols - 1;
    int32_t end_col = 0;
    int32_t net_lx = INT32_MAX;
    int32_t net_ly = INT32_MAX;
    int32_t net_ux = INT32_MIN;
    int32_t net_uy = INT32_MIN;
    for (const auto& pin : net.pins) {
      start_row = std::min(start_row, (pin.ly - region.ly) / grid_size);
      end_row = std::max(end_row, (pin.ly - region.ly) / grid_size);
      start_col = std::min(start_col, (pin.lx - region.lx) / grid_size);
      end_col = std::max(end_col, (pin.lx - region.lx) / grid_size);
      net_lx = std::min(net_lx, pin.lx);
      net_ly = std::min(net_ly, pin.ly);
      net_ux = std::max(net_ux, pin.lx);
      net_uy = std::max(net_uy, pin.ly);
    }
    double hor_rudy = 0.0;
    if (net_uy == net_ly) {
      hor_rudy = 1.0;
    } else {
      hor_rudy = 1.0 / static_cast<double>(net_uy - net_ly);
    }
    double ver_rudy = 0.0;
    if (net_ux == net_lx) {
      ver_rudy = 1.0;
    } else {
      ver_rudy = 1.0 / static_cast<double>(net_ux - net_lx);
    }
    for (int32_t row = start_row; row <= end_row; ++row) {
      for (int32_t col = start_col; col <= end_col; ++col) {
        int32_t grid_lx = region.lx + col * grid_size;
        int32_t grid_ly = region.ly + row * grid_size;
        int32_t grid_ux = std::min(region.lx + (col + 1) * grid_size, region.ux);
        int32_t grid_uy = std::min(region.ly + (row + 1) * grid_size, region.uy);
        int32_t grid_area = (grid_ux - grid_lx) * (grid_uy - grid_ly);

        int32_t overlap_lx = std::max(net_lx, grid_lx);
        int32_t overlap_ly = std::max(net_ly, grid_ly);
        int32_t overlap_ux = std::min(net_ux, grid_ux);
        int32_t overlap_uy = std::min(net_uy, grid_uy);

        int32_t overlap_area = 0;
        if (overlap_lx == overlap_ux) {
          overlap_area = overlap_uy - overlap_ly;  // 假设线宽为1
        } else if (overlap_ly == overlap_uy) {
          overlap_area = overlap_ux - overlap_lx;
        } else {
          overlap_area = (overlap_ux - overlap_lx) * (overlap_uy - overlap_ly);
        }

        if (rudy_type == "horizontal") {
          density_grid[row][col] += static_cast<double>(overlap_area) * hor_rudy / grid_area;
        } else if (rudy_type == "vertical") {
          density_grid[row][col] += static_cast<double>(overlap_area) * ver_rudy / grid_area;
        } else {
          density_grid[row][col] += static_cast<double>(overlap_area) * (hor_rudy + ver_rudy) / grid_area;
        }
      }
    }
  }

  std::ofstream csv_file(output_filename);

  for (size_t row_index = density_grid.size(); row_index-- > 0;) {
    const auto& row = density_grid[row_index];
    for (size_t i = 0; i < row.size(); ++i) {
      csv_file << std::fixed << std::setprecision(6) << row[i];
      if (i < row.size() - 1)
        csv_file << ",";
    }
    csv_file << "\n";
  }

  csv_file.close();

  return getAbsoluteFilePath(output_filename);
}

string CongestionEval::evalLUTRUDY(CongestionNets nets, CongestionRegion region, int32_t grid_size, string lutrudy_type,
                                   string output_filename)
{
  int32_t grid_cols = (region.ux - region.lx + grid_size - 1) / grid_size;
  int32_t grid_rows = (region.uy - region.ly + grid_size - 1) / grid_size;

  std::vector<std::vector<double>> density_grid(grid_rows, std::vector<double>(grid_cols, 0.0));

  for (const auto& net : nets) {
    int32_t start_row = grid_rows - 1;
    int32_t end_row = 0;
    int32_t start_col = grid_cols - 1;
    int32_t end_col = 0;
    int32_t net_lx = INT32_MAX;
    int32_t net_ly = INT32_MAX;
    int32_t net_ux = INT32_MIN;
    int32_t net_uy = INT32_MIN;
    for (const auto& pin : net.pins) {
      start_row = std::min(start_row, (pin.ly - region.ly) / grid_size);
      end_row = std::max(end_row, (pin.ly - region.ly) / grid_size);
      start_col = std::min(start_col, (pin.lx - region.lx) / grid_size);
      end_col = std::max(end_col, (pin.lx - region.lx) / grid_size);
      net_lx = std::min(net_lx, pin.lx);
      net_ly = std::min(net_ly, pin.ly);
      net_ux = std::max(net_ux, pin.lx);
      net_uy = std::max(net_uy, pin.ly);
    }
    // 计算引脚数目、纵横比、L-ness
    int pin_num = net.pins.size();
    int aspect_ratio = 1;
    if (net_ux - net_lx >= net_uy - net_ly && net_uy - net_ly != 0) {
      aspect_ratio = std::round((net_ux - net_lx) / (net_uy - net_ly));
    } else if (net_ux - net_lx < net_uy - net_ly && net_ux - net_lx != 0) {
      aspect_ratio = std::round((net_uy - net_ly) / (net_ux - net_lx));
    }
    float l_ness = 0.f;
    if (pin_num < 3) {
      l_ness = 1.f;
    } else if (pin_num <= 15) {
      std::vector<std::pair<int32_t, int32_t>> point_set;
      for (const auto& pin : net.pins) {
        point_set.push_back(std::make_pair(pin.lx, pin.ly));
      }
      l_ness = calculateLness(point_set, net_lx, net_ux, net_ly, net_uy);
    } else {
      l_ness = 0.5f;
    }

    double hor_lutrudy = 0.0;
    if (net_uy == net_ly) {
      hor_lutrudy = 1.0;
    } else {
      hor_lutrudy = getLUT(pin_num, aspect_ratio, l_ness) / static_cast<double>(net_uy - net_ly);
    }
    double ver_lutrudy = 0.0;
    if (net_ux == net_lx) {
      ver_lutrudy = 1.0;
    } else {
      ver_lutrudy = getLUT(pin_num, aspect_ratio, l_ness) / static_cast<double>(net_ux - net_lx);
    }

    for (int32_t row = start_row; row <= end_row; ++row) {
      for (int32_t col = start_col; col <= end_col; ++col) {
        int32_t grid_lx = region.lx + col * grid_size;
        int32_t grid_ly = region.ly + row * grid_size;
        int32_t grid_ux = std::min(region.lx + (col + 1) * grid_size, region.ux);
        int32_t grid_uy = std::min(region.ly + (row + 1) * grid_size, region.uy);
        int32_t grid_area = (grid_ux - grid_lx) * (grid_uy - grid_ly);

        int32_t overlap_lx = std::max(net_lx, grid_lx);
        int32_t overlap_ly = std::max(net_ly, grid_ly);
        int32_t overlap_ux = std::min(net_ux, grid_ux);
        int32_t overlap_uy = std::min(net_uy, grid_uy);

        int32_t overlap_area = 0;
        if (overlap_lx == overlap_ux) {
          overlap_area = overlap_uy - overlap_ly;  // 假设线宽为1
        } else if (overlap_ly == overlap_uy) {
          overlap_area = overlap_ux - overlap_lx;
        } else {
          overlap_area = (overlap_ux - overlap_lx) * (overlap_uy - overlap_ly);
        }

        if (lutrudy_type == "horizontal") {
          density_grid[row][col] += static_cast<double>(overlap_area) * hor_lutrudy / grid_area;
        } else if (lutrudy_type == "vertical") {
          density_grid[row][col] += static_cast<double>(overlap_area) * ver_lutrudy / grid_area;
        } else {
          density_grid[row][col] += static_cast<double>(overlap_area) * (hor_lutrudy + ver_lutrudy) / grid_area;
        }
      }
    }
  }

  std::ofstream csv_file(output_filename);

  for (size_t row_index = density_grid.size(); row_index-- > 0;) {
    const auto& row = density_grid[row_index];
    for (size_t i = 0; i < row.size(); ++i) {
      csv_file << std::fixed << std::setprecision(6) << row[i];
      if (i < row.size() - 1)
        csv_file << ",";
    }
    csv_file << "\n";
  }

  csv_file.close();

  return getAbsoluteFilePath(output_filename);
}

float CongestionEval::calculateLness(std::vector<std::pair<int32_t, int32_t>> point_set, int32_t net_lx, int32_t net_ux, int32_t net_ly,
                                     int32_t net_uy)
{
  int32_t bbox = (net_ux - net_lx) * (net_uy - net_ly);
  int32_t R1 = calcLowerLeftRP(point_set, net_lx, net_ly);
  int32_t R2 = calcLowerRightRP(point_set, net_ux, net_ly);
  int32_t R3 = calcUpperLeftRP(point_set, net_lx, net_uy);
  int32_t R4 = calcUpperRightRP(point_set, net_ux, net_uy);
  int32_t R = std::max({R1, R2, R3, R4});
  float l_ness;
  if (bbox != 0) {
    l_ness = R / bbox;
  } else {
    l_ness = 1.0;
  }
  return l_ness;
}

int32_t CongestionEval::calcLowerLeftRP(std::vector<std::pair<int32_t, int32_t>> point_set, int32_t x_min, int32_t y_min)
{
  std::sort(point_set.begin(), point_set.end());  // Sort point_set with x-coordinates in ascending order
  int32_t R = 0, y0 = point_set[0].second;
  for (size_t i = 1; i < point_set.size(); i++) {
    int32_t xi = point_set[i].first;
    if (point_set[i].second <= y0) {
      R = std::max(R, (xi - x_min) * (y0 - y_min));
      y0 = point_set[i].second;
    }
  }
  return R;
}

int32_t CongestionEval::calcLowerRightRP(std::vector<std::pair<int32_t, int32_t>> point_set, int32_t x_max, int32_t y_min)
{
  std::sort(point_set.begin(), point_set.end(), std::greater<std::pair<int32_t, int32_t>>());  // Sort point_set with x-coordinates in
                                                                                               // descending order
  int32_t R = 0, y0 = point_set[0].second, xi;
  for (size_t i = 1; i < point_set.size(); i++) {
    xi = point_set[i].first;
    if (point_set[i].second <= y0) {
      R = std::max(R, (x_max - xi) * (y0 - y_min));
      y0 = point_set[i].second;
    }
  }
  return R;
}

int32_t CongestionEval::calcUpperLeftRP(std::vector<std::pair<int32_t, int32_t>> point_set, int32_t x_min, int32_t y_max)
{
  std::sort(point_set.begin(), point_set.end(), [](const std::pair<int32_t, int32_t>& a, const std::pair<int32_t, int32_t>& b) {
    return a.second > b.second;
  });  // Sort point_set with y-coordinates in descending order
  int32_t R = 0, x0 = point_set[0].first, yi;
  for (size_t i = 1; i < point_set.size(); i++) {
    yi = point_set[i].second;
    if (point_set[i].first <= x0) {
      R = std::max(R, (y_max - yi) * (x0 - x_min));
      x0 = point_set[i].first;
    }
  }
  return R;
}

int32_t CongestionEval::calcUpperRightRP(std::vector<std::pair<int32_t, int32_t>> point_set, int32_t x_max, int32_t y_max)
{
  std::sort(point_set.begin(), point_set.end(), std::greater<std::pair<int32_t, int32_t>>());  // Sort point_set with x-coordinates in
                                                                                               // descending order
  int32_t R = 0, y0 = point_set[0].second, xi;
  for (size_t i = 1; i < point_set.size(); i++) {
    xi = point_set[i].first;
    if (point_set[i].second >= y0) {
      R = std::max(R, (y_max - y0) * (x_max - xi));
      y0 = point_set[i].second;
    }
  }
  return R;
}

double CongestionEval::getLUT(int32_t pin_num, int32_t aspect_ratio, float l_ness)
{
  int ar_index;
  if (aspect_ratio == 1) {
    ar_index = 0;
  } else if (aspect_ratio == 2 || aspect_ratio == 3) {
    ar_index = 1;
  } else if (aspect_ratio == 4) {
    ar_index = 2;
  } else {
    ar_index = 3;  // default
  }

  int pin_index = std::min(pin_num, 15) - 1;

  int l_index;
  if (l_ness <= 0.25)
    l_index = 0;
  else if (l_ness <= 0.50)
    l_index = 1;
  else if (l_ness <= 0.75)
    l_index = 2;
  else
    l_index = 3;

  return WIRELENGTH_LUT[ar_index][pin_index][l_index];
}

}  // namespace ieval