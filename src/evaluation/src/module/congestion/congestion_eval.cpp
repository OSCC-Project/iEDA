/*
 * @FilePath: congestion_eval.cpp
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#include "congestion_eval.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "general_ops.h"
#include "init_egr.h"
#include "init_idb.h"
#include "wirelength_lut.h"

namespace ieval {

#define EVAL_INIT_EGR_INST (ieval::InitEGR::getInst())
#define EVAL_INIT_IDB_INST (ieval::InitIDB::getInst())

CongestionEval* CongestionEval::_congestion_eval = nullptr;

CongestionEval::CongestionEval()
{
}

CongestionEval::~CongestionEval()
{
}

CongestionEval* CongestionEval::getInst()
{
  if (_congestion_eval == nullptr) {
    _congestion_eval = new CongestionEval();
  }

  return _congestion_eval;
}

void CongestionEval::destroyInst()
{
  if (_congestion_eval != nullptr) {
    delete _congestion_eval;
    _congestion_eval = nullptr;
  }
}

string CongestionEval::evalHoriEGR(string rt_dir_path)
{
  return evalEGR(rt_dir_path, "horizontal", "egr_horizontal_overflow.csv");
}

string CongestionEval::evalVertiEGR(string rt_dir_path)
{
  return evalEGR(rt_dir_path, "vertical", "egr_vertical_overflow.csv");
}

string CongestionEval::evalUnionEGR(string rt_dir_path)
{
  return evalEGR(rt_dir_path, "union", "egr_union_overflow.csv");
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

int32_t CongestionEval::evalHoriTotalOverflow(string rt_dir_path)
{
  return evalTotalOverflow(rt_dir_path, "horizontal");
}

int32_t CongestionEval::evalVertiTotalOverflow(string rt_dir_path)
{
  return evalTotalOverflow(rt_dir_path, "vertical");
}

int32_t CongestionEval::evalUnionTotalOverflow(string rt_dir_path)
{
  return evalTotalOverflow(rt_dir_path, "union");
}

int32_t CongestionEval::evalHoriMaxOverflow(string rt_dir_path)
{
  return evalMaxOverflow(rt_dir_path, "horizontal");
}

int32_t CongestionEval::evalVertiMaxOverflow(string rt_dir_path)
{
  return evalMaxOverflow(rt_dir_path, "vertical");
}

int32_t CongestionEval::evalUnionMaxOverflow(string rt_dir_path)
{
  return evalMaxOverflow(rt_dir_path, "union");
}

float CongestionEval::evalHoriAvgOverflow(string rt_dir_path)
{
  return evalAvgOverflow(rt_dir_path, "horizontal");
}

float CongestionEval::evalVertiAvgOverflow(string rt_dir_path)
{
  return evalAvgOverflow(rt_dir_path, "vertical");
}

float CongestionEval::evalUnionAvgOverflow(string rt_dir_path)
{
  return evalAvgOverflow(rt_dir_path, "union");
}

float CongestionEval::evalHoriMaxUtilization(string rudy_dir_path, bool use_lut)
{
  return evalMaxUtilization(rudy_dir_path, "horizontal", use_lut);
}

float CongestionEval::evalVertiMaxUtilization(string rudy_dir_path, bool use_lut)
{
  return evalMaxUtilization(rudy_dir_path, "vertical", use_lut);
}

float CongestionEval::evalUnionMaxUtilization(string rudy_dir_path, bool use_lut)
{
  return evalMaxUtilization(rudy_dir_path, "union", use_lut);
}

float CongestionEval::evalHoriAvgUtilization(string rudy_dir_path, bool use_lut)
{
  return evalAvgUtilization(rudy_dir_path, "horizontal", use_lut);
}

float CongestionEval::evalVertiAvgUtilization(string rudy_dir_path, bool use_lut)
{
  return evalAvgUtilization(rudy_dir_path, "vertical", use_lut);
}

float CongestionEval::evalUnionAvgUtilization(string rudy_dir_path, bool use_lut)
{
  return evalAvgUtilization(rudy_dir_path, "union", use_lut);
}

string CongestionEval::evalEGR(string rt_dir_path, string egr_type, string output_filename)
{
  std::unordered_map<std::string, LayerDirection> layer_directions
      = EVAL_INIT_EGR_INST->parseLayerDirection(rt_dir_path + "/initial_router/route.guide");

  // for (const auto& [layer, direction] : LayerDirections) {
  //   std::cout << "Layer: " << layer << ", Direction: " << (direction == LayerDirection::Horizontal ? "Horizontal" : "Vertical")
  //             << std::endl;
  // }
  std::vector<std::string> target_layers;
  std::string dir_path = rt_dir_path + "/initial_router/";
  std::filesystem::path parent_path = std::filesystem::path(rt_dir_path).parent_path();
  std::filesystem::path out_file_path = parent_path / output_filename;

  if (egr_type == "horizontal" || egr_type == "vertical") {
    LayerDirection target_direction = (egr_type == "horizontal") ? LayerDirection::Horizontal : LayerDirection::Vertical;

    for (const auto& [layer, direction] : layer_directions) {
      if (direction == target_direction) {
        target_layers.push_back(layer);
      }
    }

    std::vector<std::vector<double>> sum_matrix;
    bool is_first_file = true;

    for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
      std::string filename = entry.path().filename().string();
      if (filename.find("overflow_map_") != std::string::npos) {
        for (const auto& layer : target_layers) {
          if (filename.find(layer) != std::string::npos) {
            std::ifstream file(entry.path());
            std::string line;
            size_t row = 0;
            while (std::getline(file, line)) {
              std::istringstream iss(line);
              std::string value;
              int col = 0;
              while (std::getline(iss, value, ',')) {
                double num_value = std::stod(value);
                if (is_first_file) {
                  if (row >= sum_matrix.size()) {
                    sum_matrix.push_back(std::vector<double>());
                  }
                  sum_matrix[row].push_back(num_value);
                } else {
                  sum_matrix[row][col] += num_value;
                }
                col++;
              }
              row++;
            }
            is_first_file = false;
            break;
          }
        }
      }
    }
    
    std::ofstream out_file(out_file_path);
    for (const auto& row : sum_matrix) {
      for (size_t i = 0; i < row.size(); ++i) {
        out_file << row[i];
        if (i < row.size() - 1) {
          out_file << ",";
        }
      }
      out_file << "\n";
    }
    out_file.close();
  } else if (egr_type == "union") {
    dir_path = rt_dir_path + "/topology_generator/";
    std::string source_file = dir_path + "overflow_map_planar.csv";
    std::filesystem::copy_file(source_file, out_file_path, std::filesystem::copy_options::overwrite_existing);
  }
  return out_file_path.string();
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

  std::string output_path = createDirPath("RUDY_map") + "/" + output_filename;
  std::ofstream csv_file(output_path);

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

  return getAbsoluteFilePath(output_path);
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

  std::string output_path = createDirPath("RUDY_map") + "/" + output_filename;
  std::ofstream csv_file(output_path);

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

  return getAbsoluteFilePath(output_path);
}

float CongestionEval::calculateLness(std::vector<std::pair<int32_t, int32_t>> point_set, int32_t net_lx, int32_t net_ux, int32_t net_ly,
                                     int32_t net_uy)
{
  int32_t bbox = (net_ux - net_lx) * (net_uy - net_ly);
  int32_t r1 = calcLowerLeftRP(point_set, net_lx, net_ly);
  int32_t r2 = calcLowerRightRP(point_set, net_ux, net_ly);
  int32_t r3 = calcUpperLeftRP(point_set, net_lx, net_uy);
  int32_t r4 = calcUpperRightRP(point_set, net_ux, net_uy);
  int32_t r = std::max({r1, r2, r3, r4});
  float l_ness;
  if (bbox != 0) {
    l_ness = r / bbox;
  } else {
    l_ness = 1.0;
  }
  return l_ness;
}

int32_t CongestionEval::calcLowerLeftRP(std::vector<std::pair<int32_t, int32_t>> point_set, int32_t x_min, int32_t y_min)
{
  std::sort(point_set.begin(), point_set.end());  // Sort point_set with x-coordinates in ascending order
  int32_t r = 0, y0 = point_set[0].second;
  for (size_t i = 1; i < point_set.size(); i++) {
    int32_t xi = point_set[i].first;
    if (point_set[i].second <= y0) {
      r = std::max(r, (xi - x_min) * (y0 - y_min));
      y0 = point_set[i].second;
    }
  }
  return r;
}

int32_t CongestionEval::calcLowerRightRP(std::vector<std::pair<int32_t, int32_t>> point_set, int32_t x_max, int32_t y_min)
{
  std::sort(point_set.begin(), point_set.end(), std::greater<std::pair<int32_t, int32_t>>());  // Sort point_set with x-coordinates in
                                                                                               // descending order
  int32_t r = 0, y0 = point_set[0].second, xi;
  for (size_t i = 1; i < point_set.size(); i++) {
    xi = point_set[i].first;
    if (point_set[i].second <= y0) {
      r = std::max(r, (x_max - xi) * (y0 - y_min));
      y0 = point_set[i].second;
    }
  }
  return r;
}

int32_t CongestionEval::calcUpperLeftRP(std::vector<std::pair<int32_t, int32_t>> point_set, int32_t x_min, int32_t y_max)
{
  std::sort(point_set.begin(), point_set.end(), [](const std::pair<int32_t, int32_t>& a, const std::pair<int32_t, int32_t>& b) {
    return a.second > b.second;
  });  // Sort point_set with y-coordinates in descending order
  int32_t r = 0, x0 = point_set[0].first, yi;
  for (size_t i = 1; i < point_set.size(); i++) {
    yi = point_set[i].second;
    if (point_set[i].first <= x0) {
      r = std::max(r, (y_max - yi) * (x0 - x_min));
      x0 = point_set[i].first;
    }
  }
  return r;
}

int32_t CongestionEval::calcUpperRightRP(std::vector<std::pair<int32_t, int32_t>> point_set, int32_t x_max, int32_t y_max)
{
  std::sort(point_set.begin(), point_set.end(), std::greater<std::pair<int32_t, int32_t>>());  // Sort point_set with x-coordinates in
                                                                                               // descending order
  int32_t r = 0, y0 = point_set[0].second, xi;
  for (size_t i = 1; i < point_set.size(); i++) {
    xi = point_set[i].first;
    if (point_set[i].second >= y0) {
      r = std::max(r, (y_max - y0) * (x_max - xi));
      y0 = point_set[i].second;
    }
  }
  return r;
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

int32_t CongestionEval::evalTotalOverflow(string rt_dir_path, string overflow_type)
{
  int32_t total_overflow = 0;
  std::string file_name;

  if (overflow_type == "horizontal") {
    file_name =  "egr_horizontal_overflow.csv";
  } else if (overflow_type == "vertical") {
    file_name =  "egr_vertical_overflow.csv";
  } else if (overflow_type == "union") {
    file_name =  "egr_union_overflow.csv";
  } else {
    return -1;
  }
  std::filesystem::path parent_path =  std::filesystem::path(rt_dir_path).parent_path();
  std::filesystem::path file_path = parent_path / file_name;
  std::string file_path_str = file_path.string();

  std::ifstream file(file_path_str);
  if (!file.is_open()) {
    return -1;
  }

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string value;
    while (std::getline(iss, value, ',')) {
      total_overflow += std::stoi(value);
    }
  }

  file.close();
  return total_overflow;
}

int32_t CongestionEval::evalMaxOverflow(string rt_dir_path, string overflow_type)
{
  int32_t max_overflow = -1;
  std::string file_name;

  if (overflow_type == "horizontal") {
    file_name =  "egr_horizontal_overflow.csv";
  } else if (overflow_type == "vertical") {
    file_name =  "egr_vertical_overflow.csv";
  } else if (overflow_type == "union") {
    file_name =  "egr_union_overflow.csv";
  } else {
    return -1;
  }
  std::filesystem::path parent_path =  std::filesystem::path(rt_dir_path).parent_path();
  std::filesystem::path file_path = parent_path / file_name;
  std::string file_path_str = file_path.string();

  std::ifstream file(file_path_str);
  if (!file.is_open()) {
    return -1;
  }

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string value;
    while (std::getline(iss, value, ',')) {
      int32_t current_value = std::stoi(value);
      max_overflow = std::max(max_overflow, current_value);
    }
  }

  file.close();
  return max_overflow;
}

float CongestionEval::evalAvgOverflow(string rt_dir_path, string overflow_type)
{
  float avg_overflow = 0.0f;
  std::string file_name;

  if (overflow_type == "horizontal") {
    file_name =  "egr_horizontal_overflow.csv";
  } else if (overflow_type == "vertical") {
    file_name =  "egr_vertical_overflow.csv";
  } else if (overflow_type == "union") {
    file_name =  "egr_union_overflow.csv";
  } else {
    return -1;
  }
  std::filesystem::path parent_path =  std::filesystem::path(rt_dir_path).parent_path();
  std::filesystem::path file_path = parent_path / file_name;
  std::string file_path_str = file_path.string();

  std::ifstream file(file_path_str);
  if (!file.is_open()) {
    return -1;
  }

  std::vector<int32_t> values;

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string value;
    while (std::getline(iss, value, ',')) {
      int32_t current_value = std::stoi(value);
      values.push_back(current_value);
    }
  }

  file.close();

  std::sort(values.begin(), values.end(), std::greater<int32_t>());

  size_t size = values.size();
  size_t idx_0_5_percent = std::max(size_t(1), static_cast<size_t>(std::ceil(size * 0.005)));
  size_t idx_1_percent = std::max(size_t(1), static_cast<size_t>(std::ceil(size * 0.01)));
  size_t idx_2_percent = std::max(size_t(1), static_cast<size_t>(std::ceil(size * 0.02)));
  size_t idx_5_percent = std::max(size_t(1), static_cast<size_t>(std::ceil(size * 0.05)));

  float sum_05 = 0.0f;
  float sum_1 = 0.0f;
  float sum_2 = 0.0f;
  float sum_5 = 0.0f;

  float weight_05 = 0.4f;
  float weight_1 = 0.3f;
  float weight_2 = 0.2f;
  float weight_5 = 0.1f;

  // 0-0.5%
  for (size_t i = 0; i < idx_0_5_percent; ++i) {
    sum_05 += values[i];
  }
  sum_1 = sum_05;

  // 0.5%-1%
  for (size_t i = idx_0_5_percent; i < idx_1_percent; ++i) {
    sum_1 += values[i];
  }
  sum_2 = sum_1;

  // 1%-2%
  for (size_t i = idx_1_percent; i < idx_2_percent; ++i) {
    sum_2 += values[i];
  }
  sum_5 = sum_2;

  // 2%-5%
  for (size_t i = idx_2_percent; i < idx_5_percent; ++i) {
    sum_5 += values[i];
  }

  avg_overflow = (sum_05 * weight_05 + sum_1 * weight_1 + sum_2 * weight_2 + sum_5 * weight_5) / 4.0;

  return avg_overflow;
}

float CongestionEval::evalMaxUtilization(string map_path, string utilization_type, bool use_lut)
{
  float max_util = -1.0;
  std::string file_path;
  std::string file_name;

  if (use_lut == true) {
    file_name = "/lut_rudy_";
  } else {
    file_name = "/rudy_";
  }

  if (utilization_type == "horizontal") {
    file_path = map_path + file_name + "horizontal.csv";
  } else if (utilization_type == "vertical") {
    file_path = map_path + file_name + "vertical.csv";
  } else if (utilization_type == "union") {
    file_path = map_path + file_name + "union.csv";
  } else {
    return -1.0;
  }

  std::ifstream file(file_path);
  if (!file.is_open()) {
    return -1.0;
  }

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string value;
    while (std::getline(iss, value, ',')) {
      float current_value = std::stof(value);
      max_util = std::fmax(max_util, current_value);
    }
  }

  file.close();
  return max_util;
}

float CongestionEval::evalAvgUtilization(string rudy_dir_path, string utilization_type, bool use_lut)
{
  float avg_util = 0.0f;
  std::string file_path;
  std::string file_name;

  if (use_lut == true) {
    file_name = "/lut_rudy_";
  } else {
    file_name = "/rudy_";
  }

  if (utilization_type == "horizontal") {
    file_path = rudy_dir_path + file_name + "horizontal.csv";
  } else if (utilization_type == "vertical") {
    file_path = rudy_dir_path + file_name + "vertical.csv";
  } else if (utilization_type == "union") {
    file_path = rudy_dir_path + file_name + "union.csv";
  } else {
    return -1.0;
  }

  std::ifstream file(file_path);
  if (!file.is_open()) {
    return -1.0;
  }

  std::vector<float> values;

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string value;
    while (std::getline(iss, value, ',')) {
      float current_value = std::stof(value);
      values.push_back(current_value);
    }
  }

  file.close();

  std::sort(values.begin(), values.end(), std::greater<float>());

  size_t size = values.size();
  size_t idx_0_5_percent = std::max(size_t(1), static_cast<size_t>(std::ceil(size * 0.005)));
  size_t idx_1_percent = std::max(size_t(1), static_cast<size_t>(std::ceil(size * 0.01)));
  size_t idx_2_percent = std::max(size_t(1), static_cast<size_t>(std::ceil(size * 0.02)));
  size_t idx_5_percent = std::max(size_t(1), static_cast<size_t>(std::ceil(size * 0.05)));
  // size_t idx_0_5_percent = std::max(size_t(1), static_cast<size_t>(std::ceil(size * 0.1)));
  // size_t idx_1_percent = std::max(size_t(1), static_cast<size_t>(std::ceil(size * 0.2)));
  // size_t idx_2_percent = std::max(size_t(1), static_cast<size_t>(std::ceil(size * 0.3)));
  // size_t idx_5_percent = std::max(size_t(1), static_cast<size_t>(std::ceil(size * 0.4)));

  float sum_05 = 0.0f;
  float sum_1 = 0.0f;
  float sum_2 = 0.0f;
  float sum_5 = 0.0f;

  float weight_05 = 0.4f;
  float weight_1 = 0.3f;
  float weight_2 = 0.2f;
  float weight_5 = 0.1f;

  // 0-0.5%
  for (size_t i = 0; i < idx_0_5_percent; ++i) {
    sum_05 += values[i];
  }
  sum_1 = sum_05;

  // 0.5%-1%
  for (size_t i = idx_0_5_percent; i < idx_1_percent; ++i) {
    sum_1 += values[i];
  }
  sum_2 = sum_1;

  // 1%-2%
  for (size_t i = idx_1_percent; i < idx_2_percent; ++i) {
    sum_2 += values[i];
  }
  sum_5 = sum_2;

  // 2%-5%
  for (size_t i = idx_2_percent; i < idx_5_percent; ++i) {
    sum_5 += values[i];
  }

  avg_util = (sum_05 * weight_05 + sum_1 * weight_1 + sum_2 * weight_2 + sum_5 * weight_5) / 4.0;

  return avg_util;
}

void CongestionEval::initEGR()
{
  EVAL_INIT_EGR_INST->runEGR();
}

void CongestionEval::destroyEGR()
{
  EVAL_INIT_EGR_INST->destroyInst();
}

void CongestionEval::initIDB()
{
  EVAL_INIT_IDB_INST->initCongestionDB();
}

void CongestionEval::destroyIDB()
{
  EVAL_INIT_IDB_INST->destroyInst();
}

CongestionNets CongestionEval::getCongestionNets()
{
  return EVAL_INIT_IDB_INST->getCongestionNets();
}

CongestionRegion CongestionEval::getCongestionRegion()
{
  return EVAL_INIT_IDB_INST->getCongestionRegion();
}

int32_t CongestionEval::getRowHeight()
{
  return EVAL_INIT_IDB_INST->getRowHeight();
}

void CongestionEval::evalNetInfo()
{
  CongestionNets nets = getCongestionNets();

  for (const auto& net : nets) {
    int32_t net_lx = INT32_MAX;
    int32_t net_ly = INT32_MAX;
    int32_t net_ux = INT32_MIN;
    int32_t net_uy = INT32_MIN;
    for (const auto& pin : net.pins) {
      net_lx = std::min(net_lx, pin.lx);
      net_ly = std::min(net_ly, pin.ly);
      net_ux = std::max(net_ux, pin.lx);
      net_uy = std::max(net_uy, pin.ly);
    }
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

    _name_pin_numer.emplace(net.name, pin_num);
    _name_aspect_ratio.emplace(net.name, aspect_ratio);
    _name_lness.emplace(net.name, l_ness);
  }
}

int CongestionEval::findPinNumber(std::string net_name)
{
  auto it = _name_pin_numer.find(net_name);
  if (it != _name_pin_numer.end()) {
    return it->second;
  }
  throw std::runtime_error("Pin number not found for net: " + net_name);
}

int CongestionEval::findAspectRatio(std::string net_name)
{
  auto it = _name_aspect_ratio.find(net_name);
  if (it != _name_aspect_ratio.end()) {
    return it->second;
  }
  throw std::runtime_error("Aspect ratio not found for net: " + net_name);
}

float CongestionEval::findLness(std::string net_name)
{
  auto it = _name_lness.find(net_name);
  if (it != _name_lness.end()) {
    return it->second;
  }
  throw std::runtime_error("Lness not found for net: " + net_name);
}

std::string CongestionEval::getEGRDirPath()
{
  return EVAL_INIT_EGR_INST->getEGRDirPath();
}

std::string CongestionEval::getDefaultOutputDir()
{
  return getDefaultOutputPath();
}

}  // namespace ieval