/*
 * @FilePath: density_eval.cpp
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#include "density_eval.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_set>

#include "general_ops.h"
#include "init_idb.h"

namespace ieval {

using namespace std;

#define EVAL_INIT_IDB_INST (ieval::InitIDB::getInst())

DensityEval* DensityEval::_density_eval = nullptr;

DensityEval::DensityEval()
{
}

DensityEval::~DensityEval()
{
}

DensityEval* DensityEval::getInst()
{
  if (_density_eval == nullptr) {
    _density_eval = new DensityEval();
  }

  return _density_eval;
}

void DensityEval::destroyInst()
{
  if (_density_eval != nullptr) {
    delete _density_eval;
    _density_eval = nullptr;
  }
}

std::string DensityEval::evalMacroDensity(DensityCells cells, DensityRegion region, int32_t grid_size, std::string stage)
{
  return evalDensity(cells, region, grid_size, "macro", stage + "_macro_density.csv");
}

std::string DensityEval::evalStdCellDensity(DensityCells cells, DensityRegion region, int32_t grid_size, std::string stage)
{
  return evalDensity(cells, region, grid_size, "stdcell", stage + "_stdcell_density.csv");
}

std::string DensityEval::evalAllCellDensity(DensityCells cells, DensityRegion region, int32_t grid_size, std::string stage)
{
  return evalDensity(cells, region, grid_size, "all", stage + "_allcell_density.csv");
}

std::string DensityEval::evalMacroPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size, std::string stage, bool neighbor)
{
  return evalPinDensity(pins, region, grid_size, neighbor, "macro", stage + "_macro_pin_density.csv");
}

std::string DensityEval::evalStdCellPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size, std::string stage, bool neighbor)
{
  return evalPinDensity(pins, region, grid_size, neighbor, "stdcell", stage + "_stdcell_pin_density.csv");
}

std::string DensityEval::evalAllCellPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size, std::string stage, bool neighbor)
{
  return evalPinDensity(pins, region, grid_size, neighbor, "all", stage + "_allcell_pin_density.csv");
}

std::string DensityEval::evalLocalNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size, std::string stage, bool neighbor)
{
  return evalNetDensity(nets, region, grid_size, neighbor, "local", stage + "_local_net_density.csv");
}

std::string DensityEval::evalGlobalNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size, std::string stage, bool neighbor)
{
  return evalNetDensity(nets, region, grid_size, neighbor, "global", stage + "_global_net_density.csv");
}

std::string DensityEval::evalAllNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size, std::string stage, bool neighbor)
{
  return evalNetDensity(nets, region, grid_size, neighbor, "all", stage + "_allnet_density.csv");
}

std::string DensityEval::evalHorizonMargin(DensityCells cells, DensityRegion die, DensityRegion core, int32_t grid_size, std::string stage)
{
  return evalMargin(cells, die, core, grid_size, "horizontal", stage + "_horizontal_margin.csv");
}

std::string DensityEval::evalVerticalMargin(DensityCells cells, DensityRegion die, DensityRegion core, int32_t grid_size, std::string stage)
{
  return evalMargin(cells, die, core, grid_size, "vertical", stage + "_vertical_margin.csv");
}

std::string DensityEval::evalAllMargin(DensityCells cells, DensityRegion die, DensityRegion core, int32_t grid_size, std::string stage)
{
  return evalMargin(cells, die, core, grid_size, "union", stage + "_union_margin.csv");
}

std::string DensityEval::evalDensity(DensityCells cells, DensityRegion region, int32_t grid_size, std::string cell_type,
                                     std::string output_filename)
{
  int32_t grid_cols = (region.ux - region.lx + grid_size - 1) / grid_size;
  int32_t grid_rows = (region.uy - region.ly + grid_size - 1) / grid_size;

  std::vector<std::vector<double>> density_grid(grid_rows, std::vector<double>(grid_cols, 0.0));

  for (const auto& cell : cells) {
    if (cell_type != "all" && cell.type != cell_type) {
      continue;
    }

    int32_t start_row = std::max(0, (cell.ly - region.ly) / grid_size);
    int32_t end_row = std::min(grid_rows - 1, (cell.ly + cell.height - region.ly) / grid_size);
    int32_t start_col = std::max(0, (cell.lx - region.lx) / grid_size);
    int32_t end_col = std::min(grid_cols - 1, (cell.lx + cell.width - region.lx) / grid_size);

    for (int32_t row = start_row; row <= end_row; ++row) {
      for (int32_t col = start_col; col <= end_col; ++col) {
        int32_t grid_lx = region.lx + col * grid_size;
        int32_t grid_ly = region.ly + row * grid_size;
        int32_t grid_ux = std::min(region.lx + (col + 1) * grid_size, region.ux);
        int32_t grid_uy = std::min(region.ly + (row + 1) * grid_size, region.uy);

        int32_t overlap_lx = std::max(cell.lx, grid_lx);
        int32_t overlap_ly = std::max(cell.ly, grid_ly);
        int32_t overlap_ux = std::min(cell.lx + cell.width, grid_ux);
        int32_t overlap_uy = std::min(cell.ly + cell.height, grid_uy);

        int32_t overlap_area = std::max(0, overlap_ux - overlap_lx) * std::max(0, overlap_uy - overlap_ly);
        int32_t grid_area = (grid_ux - grid_lx) * (grid_uy - grid_ly);

        density_grid[row][col] += static_cast<double>(overlap_area) / grid_area;
      }
    }
  }

  std::string stage;
  size_t underscore_pos = output_filename.find('_');
  if (underscore_pos != std::string::npos) {
    stage = output_filename.substr(0, underscore_pos);
  }

  std::string save_dir;
  if (stage.find("place") != std::string::npos || stage.find("pl") != std::string::npos) {
    save_dir = "/pl/density_map";
  } else if (stage.find("cts") != std::string::npos) {
    save_dir = "/cts/density_map";
  } else if (stage.find("route") != std::string::npos) {
    save_dir = "/route/density_map";
  } else {
    save_dir = "other/density_map";
  }

  std::string output_path = createDirPath(save_dir) + "/" + output_filename;
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

std::string DensityEval::evalPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size, bool neighbor, std::string pin_type,
                                        std::string output_filename)
{
  int32_t grid_cols = (region.ux - region.lx + grid_size - 1) / grid_size;
  int32_t grid_rows = (region.uy - region.ly + grid_size - 1) / grid_size;

  std::vector<std::vector<int32_t>> pin_count(grid_rows, std::vector<int32_t>(grid_cols, 0));
  std::vector<std::vector<double>> density_grid(grid_rows, std::vector<double>(grid_cols, 0.0));

  for (const auto& pin : pins) {
    if (pin_type != "all" && pin.type != pin_type) {
      continue;
    }

    int32_t col = (pin.lx - region.lx) / grid_size;
    int32_t row = (pin.ly - region.ly) / grid_size;

    if (col >= 0 && col < grid_cols && row >= 0 && row < grid_rows) {
      pin_count[row][col] += 1;
    }
  }

  if (neighbor) {
    const std::vector<std::vector<int>> kernel = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    std::vector<std::vector<int32_t>> padded_matrix(grid_rows + 2, std::vector<int32_t>(grid_cols + 2, 0));
    for (int32_t row = 0; row < grid_rows; ++row) {
      for (int32_t col = 0; col < grid_cols; ++col) {
        padded_matrix[row + 1][col + 1] = pin_count[row][col];
      }
    }
    std::vector<std::vector<int32_t>> neighbor_pin_count(grid_rows, std::vector<int32_t>(grid_cols, 0));
    for (int32_t row = 0; row < grid_rows; ++row) {
      for (int32_t col = 0; col < grid_cols; ++col) {
        int32_t sum = 0;
        for (int32_t ker_i = 0; ker_i < 3; ++ker_i) {
          for (int32_t ker_j = 0; ker_j < 3; ++ker_j) {
            sum += kernel[ker_i][ker_j] * padded_matrix[row + ker_i][col + ker_j];
          }
        }
        neighbor_pin_count[row][col] = sum;
      }
    }
    for (int32_t row = 0; row < grid_rows; ++row) {
      for (int32_t col = 0; col < grid_cols; ++col) {
        density_grid[row][col] = static_cast<double>(neighbor_pin_count[row][col]);
      }
    }
  } else {
    for (int32_t row = 0; row < grid_rows; ++row) {
      for (int32_t col = 0; col < grid_cols; ++col) {
        // int32_t grid_lx = region.lx + col * grid_size;
        // int32_t grid_ly = region.ly + row * grid_size;
        // int32_t grid_ux = std::min(region.lx + (col + 1) * grid_size, region.ux);
        // int32_t grid_uy = std::min(region.ly + (row + 1) * grid_size, region.uy);

        // double grid_area = static_cast<double>((grid_ux - grid_lx) * (grid_uy - grid_ly));
        // density_grid[row][col] = static_cast<double>(pin_count[row][col]) / grid_area;
        density_grid[row][col] = static_cast<double>(pin_count[row][col]);
      }
    }
  }

  std::string stage;
  size_t underscore_pos = output_filename.find('_');
  if (underscore_pos != std::string::npos) {
    stage = output_filename.substr(0, underscore_pos);
  }

  std::string save_dir;
  if (stage.find("place") != std::string::npos || stage.find("pl") != std::string::npos) {
    save_dir = "/pl/density_map";
  } else if (stage.find("cts") != std::string::npos) {
    save_dir = "/cts/density_map";
  } else if (stage.find("route") != std::string::npos) {
    save_dir = "/route/density_map";
  } else {
    save_dir = "other/density_map";
  }

  std::string output_path;
  if (neighbor) {
    output_path = createDirPath(save_dir) + "/" + "neighbor_" + output_filename;
  } else {
    output_path = createDirPath(save_dir) + "/" + output_filename;
  }

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

std::string DensityEval::evalNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size, bool neighbor, std::string net_type,
                                        std::string output_filename)
{
  int32_t grid_cols = (region.ux - region.lx + grid_size - 1) / grid_size;
  int32_t grid_rows = (region.uy - region.ly + grid_size - 1) / grid_size;

  std::vector<std::vector<int>> net_count(grid_rows, std::vector<int>(grid_cols, 0));

  for (const auto& net : nets) {
    int32_t start_col = std::max(0, (net.lx - region.lx) / grid_size);
    int32_t end_col = std::min(grid_cols - 1, (net.ux - region.lx) / grid_size);
    int32_t start_row = std::max(0, (net.ly - region.ly) / grid_size);
    int32_t end_row = std::min(grid_rows - 1, (net.uy - region.ly) / grid_size);

    bool is_local = (start_col == end_col) && (start_row == end_row);

    if (net_type == "all" || (net_type == "local" && is_local) || (net_type == "global" && !is_local)) {
      if (is_local || net_type == "local") {
        net_count[start_row][start_col]++;
      } else {
        for (int32_t row = start_row; row <= end_row; ++row) {
          for (int32_t col = start_col; col <= end_col; ++col) {
            net_count[row][col]++;
          }
        }
      }
    }
  }

  std::string stage;
  size_t underscore_pos = output_filename.find('_');
  if (underscore_pos != std::string::npos) {
    stage = output_filename.substr(0, underscore_pos);
  }

  std::string save_dir;
  if (stage.find("place") != std::string::npos || stage.find("pl") != std::string::npos) {
    save_dir = "/pl/density_map";
  } else if (stage.find("cts") != std::string::npos) {
    save_dir = "/cts/density_map";
  } else if (stage.find("route") != std::string::npos) {
    save_dir = "/route/density_map";
  } else {
    save_dir = "other/density_map";
  }

  std::string output_path;
  if (neighbor) {
    const std::vector<std::vector<int>> kernel = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    std::vector<std::vector<int32_t>> padded_matrix(grid_rows + 2, std::vector<int32_t>(grid_cols + 2, 0));
    for (int32_t row = 0; row < grid_rows; ++row) {
      for (int32_t col = 0; col < grid_cols; ++col) {
        padded_matrix[row + 1][col + 1] = net_count[row][col];
      }
    }
    std::vector<std::vector<int32_t>> neighbor_net_count(grid_rows, std::vector<int32_t>(grid_cols, 0));
    for (int32_t row = 0; row < grid_rows; ++row) {
      for (int32_t col = 0; col < grid_cols; ++col) {
        int32_t sum = 0;
        for (int32_t ker_i = 0; ker_i < 3; ++ker_i) {
          for (int32_t ker_j = 0; ker_j < 3; ++ker_j) {
            sum += kernel[ker_i][ker_j] * padded_matrix[row + ker_i][col + ker_j];
          }
        }
        neighbor_net_count[row][col] = sum;
      }
    }
    output_path = createDirPath(save_dir) + "/" + "neighbor_" + output_filename;
    std::ofstream csv_file(output_path);
    for (int32_t row = grid_rows - 1; row >= 0; --row) {
      for (int32_t col = 0; col < grid_cols; ++col) {
        double density = neighbor_net_count[row][col];
        csv_file << std::fixed << std::setprecision(6) << density;
        if (col < grid_cols - 1)
          csv_file << ",";
      }
      csv_file << "\n";
    }
    csv_file.close();
  }

  else {
    output_path = createDirPath(save_dir) + "/" + output_filename;
    std::ofstream csv_file(output_path);
    for (int32_t row = grid_rows - 1; row >= 0; --row) {
      for (int32_t col = 0; col < grid_cols; ++col) {
        // int32_t grid_lx = region.lx + col * grid_size;
        // int32_t grid_ly = region.ly + row * grid_size;
        // int32_t grid_ux = std::min(region.lx + (col + 1) * grid_size, region.ux);
        // int32_t grid_uy = std::min(region.ly + (row + 1) * grid_size, region.uy);

        // double grid_area = static_cast<double>((grid_ux - grid_lx) * (grid_uy - grid_ly));

        // double density = net_count[row][col] / grid_area;
        double density = net_count[row][col];

        csv_file << std::fixed << std::setprecision(6) << density;
        if (col < grid_cols - 1)
          csv_file << ",";
      }
      csv_file << "\n";
    }
    csv_file.close();
  }

  return getAbsoluteFilePath(output_path);
}

std::string DensityEval::evalMargin(DensityCells cells, DensityRegion die, DensityRegion core, int32_t grid_size, std::string margin_type,
                                    std::string output_filename)
{
  std::vector<DensityCell> macros;
  for (const auto& cell : cells) {
    if (cell.type == "macro") {
      macros.push_back(cell);
    }
  }

  std::vector<MarginGrid> margin_grid = initMarginGrid(die, grid_size);
  for (size_t i = 0; i < margin_grid.size(); ++i) {
    int32_t h_right = core.ux;
    int32_t h_left = core.lx;
    int32_t v_up = core.uy;
    int32_t v_down = core.ly;
    bool overlap = false;

    if (margin_grid[i].ux <= h_left || margin_grid[i].lx >= h_right || margin_grid[i].uy <= v_down || margin_grid[i].ly >= v_up) {
      continue;
    }

    int32_t overlap_area = 0;
    int32_t grid_area = (margin_grid[i].ux - margin_grid[i].lx) * (margin_grid[i].uy - margin_grid[i].ly);
    for (size_t j = 0; j < macros.size(); ++j) {
      int32_t rect_lx = std::max(margin_grid[i].lx, macros[j].lx);
      int32_t rect_ly = std::max(margin_grid[i].ly, macros[j].ly);
      int32_t rect_ux = std::min(margin_grid[i].ux, macros[j].lx + macros[j].width);
      int32_t rect_uy = std::min(margin_grid[i].uy, macros[j].ly + macros[j].height);
      if (rect_lx < rect_ux && rect_ly < rect_uy) {
        overlap_area += (std::min(margin_grid[i].ux, macros[j].lx + macros[j].width) - std::max(margin_grid[i].lx, macros[j].lx))
                        * (std::min(margin_grid[i].uy, macros[j].ly + macros[j].height) - std::max(margin_grid[i].ly, macros[j].ly));
      }
      if (overlap_area > 0.5 * grid_area) {
        overlap = true;
        break;
      }
    }

    if (!overlap) {
      for (size_t j = 0; j < macros.size(); ++j) {
        int32_t macro_middle_x = macros[j].lx + macros[j].width * 0.5;
        int32_t macro_middle_y = macros[j].ly + macros[j].height * 0.5;
        int32_t grid_middle_x = (margin_grid[i].lx + margin_grid[i].ux) * 0.5;
        int32_t grid_middle_y = (margin_grid[i].ly + margin_grid[i].uy) * 0.5;
        if (grid_middle_y >= macros[j].ly && grid_middle_y <= macros[j].ly + macros[j].height) {
          if (macro_middle_x > grid_middle_x) {
            h_right = std::min(h_right, macros[j].lx);
          } else {
            h_left = std::max(h_left, macros[j].lx + macros[j].width);
          }
        }
        if (grid_middle_x >= macros[j].lx && grid_middle_x <= macros[j].lx + macros[j].width) {
          if (macro_middle_y > grid_middle_y) {
            v_up = std::min(v_up, macros[j].ly);
          } else {
            v_down = std::max(v_down, macros[j].ly + macros[j].height);
          }
        }
      }
      if (margin_type == "horizontal") {
        margin_grid[i].margin = h_right - h_left;
      } else if (margin_type == "vertical") {
        margin_grid[i].margin = v_up - v_down;
      } else {
        margin_grid[i].margin = h_right - h_left + v_up - v_down;
      }
    }
  }

  std::string stage;
  size_t underscore_pos = output_filename.find('_');
  if (underscore_pos != std::string::npos) {
    stage = output_filename.substr(0, underscore_pos);
  }

  std::string save_dir;
  if (stage.find("place") != std::string::npos || stage.find("pl") != std::string::npos) {
    save_dir = "/pl/margin_map";
  } else{
    save_dir = "other/margin_map";
  }

  std::string output_path = createDirPath(save_dir) + "/" + output_filename;
  std::ofstream csv_file(output_path);

  int32_t grid_cols = (die.ux - die.lx + grid_size - 1) / grid_size;
  int32_t grid_rows = (die.uy - die.ly + grid_size - 1) / grid_size;

  for (int32_t row_index = grid_rows; row_index-- > 0;) {
    for (int32_t i = 0; i < grid_cols; ++i) {
      csv_file << std::fixed << std::setprecision(6) << margin_grid[row_index * grid_cols + i].margin;
      if (i < grid_cols - 1)
        csv_file << ",";
    }
    csv_file << "\n";
  }

  csv_file.close();

  return getAbsoluteFilePath(output_path);
}

void DensityEval::initIDB()
{
  EVAL_INIT_IDB_INST->initDensityDB();
}

void DensityEval::destroyIDB()
{
  EVAL_INIT_IDB_INST->destroyInst();
}

void DensityEval::initIDBRegion()
{
  EVAL_INIT_IDB_INST->initDensityDBRegion();
}

void DensityEval::initIDBCells()
{
  EVAL_INIT_IDB_INST->initDensityDBCells();
}

void DensityEval::initIDBNets()
{
  EVAL_INIT_IDB_INST->initDensityDBNets();
}

DensityRegion DensityEval::getDensityRegion()
{
  return EVAL_INIT_IDB_INST->getDensityRegion();
}

DensityRegion DensityEval::getDensityRegionCore()
{
  return EVAL_INIT_IDB_INST->getDensityRegionCore();
}

DensityCells DensityEval::getDensityCells()
{
  return EVAL_INIT_IDB_INST->getDensityCells();
}

DensityPins DensityEval::getDensityPins()
{
  return EVAL_INIT_IDB_INST->getDensityPins();
}

DensityNets DensityEval::getDensityNets()
{
  return EVAL_INIT_IDB_INST->getDensityNets();
}

int32_t DensityEval::getRowHeight()
{
  return EVAL_INIT_IDB_INST->getRowHeight();
}

std::vector<MarginGrid> DensityEval::initMarginGrid(DensityRegion die, int32_t grid_size)
{
  std::vector<MarginGrid> margin_grids;

  int32_t x = die.lx;
  int32_t y = die.ly;
  int32_t grid_cols = (die.ux - die.lx + grid_size - 1) / grid_size;
  int32_t grid_rows = (die.uy - die.ly + grid_size - 1) / grid_size;
  int32_t grid_num = grid_cols * grid_rows;

  margin_grids.reserve(grid_num);
  for (int32_t i = 0; i < grid_num; ++i) {
    MarginGrid margin_grid;
    margin_grid.lx = x;
    margin_grid.ly = y;
    margin_grid.ux = x + grid_size;
    margin_grid.uy = y + grid_size;
    margin_grid.margin = 0;
    x += grid_size;
    if (x >= die.ux) {
      y += grid_size;
      x = die.lx;
    }
    margin_grids.push_back(margin_grid);
  }

  return margin_grids;
}

std::map<int, double> DensityEval::patchCellDensity(DensityCells cells, std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>> patch_coords)
{
  std::map<int, double> patch_cell_density;
  DensityGridIndex index;
  index.grid_size_x = EVAL_INIT_IDB_INST->getDieWidth() / 100 ;
  index.grid_size_y = EVAL_INIT_IDB_INST->getDieHeight() / 100;
  // 构建单元格网格索引
  for (auto& cell : cells) {
    int min_grid_x = cell.lx / index.grid_size_x;
    int max_grid_x = (cell.lx + cell.width) / index.grid_size_x;  
    int min_grid_y = cell.ly / index.grid_size_y;
    int max_grid_y = (cell.ly + cell.height) / index.grid_size_y;

    for (int x = min_grid_x; x <= max_grid_x; ++x) {
      for (int y = min_grid_y; y <= max_grid_y; ++y) {
        index.cell_grid[{x, y}].push_back(cell);
      }
    }
  }

  for (const auto& [patch_id, coord] : patch_coords) {
      double density = 0.0;
      auto [l_range, u_range] = coord;
      
      // 提取当前 patch 的物理边界
      const int patch_lx = l_range.first;
      const int patch_ly = l_range.second;
      const int patch_ux = u_range.first;
      const int patch_uy = u_range.second;

      // 计算 patch 面积（保持与 pin 密度相同的左闭右开区间）
      const int patch_width = patch_ux - patch_lx;
      const int patch_height = patch_uy - patch_ly;
      const int patch_area = patch_width * patch_height;

      // 计算 patch 覆盖的网格范围
      int min_grid_x = patch_lx / index.grid_size_x;
      int max_grid_x = patch_ux / index.grid_size_x;
      int min_grid_y = patch_ly / index.grid_size_y;
      int max_grid_y = patch_uy / index.grid_size_y;

      std::unordered_set<int> processed_cells; // 记录已处理的 cell ID, 避免重复多次处理同一cell

      for (int x = min_grid_x; x <= max_grid_x; ++x) {
          for (int y = min_grid_y; y <= max_grid_y; ++y) {
              auto it = index.cell_grid.find({x, y});
              if (it == index.cell_grid.end()) continue;
   
              for (const auto& cell : it->second) {
                if (processed_cells.count(cell.id)) continue;
                processed_cells.insert(cell.id);
                // 计算重叠面积
                const int overlap_lx = std::max(cell.lx, patch_lx);
                const int overlap_ly = std::max(cell.ly, patch_ly);
                const int overlap_ux = std::min(cell.lx + cell.width, patch_ux);
                const int overlap_uy = std::min(cell.ly + cell.height, patch_uy);

                // 有效重叠面积计算
                const int overlap_width = std::max(0, overlap_ux - overlap_lx);
                const int overlap_height = std::max(0, overlap_uy - overlap_ly);
                const int overlap_area = overlap_width * overlap_height;

                // 累加密度贡献（当且仅当有重叠时）
                if (overlap_area > 0) {
                    density += static_cast<double>(overlap_area) / patch_area;
                }

              }
          }
      }
      
      patch_cell_density[patch_id] = density;
  }

  return patch_cell_density;
}


std::map<int, int> DensityEval::patchPinDensity(DensityPins pins, std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>> patch_coords)
{
  std::map<int, int> patch_pin_density;

  // 构建 pin 网格索引
  DensityGridIndex index;
  index.grid_size_x = EVAL_INIT_IDB_INST->getDieWidth() / 100 ;
  index.grid_size_y = EVAL_INIT_IDB_INST->getDieHeight() / 100;

  for (const auto& pin : pins) {
      int grid_x = pin.lx / index.grid_size_x;
      int grid_y = pin.ly / index.grid_size_y;
      index.pin_grid[{grid_x, grid_y}].push_back(pin);
  }

  for (const auto& [patch_id, coord] : patch_coords) {
      auto [l_range, u_range] = coord;
      const int patch_lx = l_range.first;
      const int patch_ly = l_range.second;
      const int patch_ux = u_range.first;
      const int patch_uy = u_range.second;

      int pin_count = 0;

      // 计算覆盖网格范围
      int min_grid_x = patch_lx / index.grid_size_x;
      int max_grid_x = patch_ux / index.grid_size_x;
      int min_grid_y = patch_ly / index.grid_size_y;
      int max_grid_y = patch_uy / index.grid_size_y;

      for (int x = min_grid_x; x <= max_grid_x; ++x) {
          for (int y = min_grid_y; y <= max_grid_y; ++y) {
              auto it = index.pin_grid.find({x, y});
              if (it == index.pin_grid.end()) continue;

              for (const auto& pin : it->second) {
                  if (pin.lx >= patch_lx && pin.ly >= patch_ly &&
                      pin.lx <= patch_ux && pin.ly <= patch_uy) 
                  {
                      ++pin_count;
                  }
              }
          }
      }
      patch_pin_density[patch_id] = pin_count;
  }
  return patch_pin_density;
}

std::map<int, double> DensityEval::patchNetDensity(DensityNets nets, std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>> patch_coords)
{
  std::map<int, double> patch_net_density;
  DensityGridIndex index;
  index.grid_size_x = EVAL_INIT_IDB_INST->getDieWidth() / 100 ;
  index.grid_size_y = EVAL_INIT_IDB_INST->getDieHeight() / 100;

  // 构建线网网格索引
  for (const auto& net : nets) {
    int min_grid_x = net.lx / index.grid_size_x;
    int max_grid_x = net.ux / index.grid_size_x;  
    int min_grid_y = net.ly / index.grid_size_y;
    int max_grid_y = net.uy / index.grid_size_y;

    for (int x = min_grid_x; x <= max_grid_x; ++x) {
      for (int y = min_grid_y; y <= max_grid_y; ++y) {
        index.net_grid[{x, y}].push_back(net);
      }
    }
  }

  for (const auto& [patch_id, coord] : patch_coords) {
      double density = 0.0;
      auto [l_range, u_range] = coord;
      
      // 提取当前 patch 的物理边界
      const int patch_lx = l_range.first;
      const int patch_ly = l_range.second;
      const int patch_ux = u_range.first;
      const int patch_uy = u_range.second;

      // 计算 patch 面积（保持与 pin 密度相同的左闭右开区间）
      const int patch_width = patch_ux - patch_lx;
      const int patch_height = patch_uy - patch_ly;
      const int patch_area = patch_width * patch_height;

      // 计算 patch 覆盖的网格范围
      int min_grid_x = patch_lx / index.grid_size_x;
      int max_grid_x = patch_ux / index.grid_size_x;
      int min_grid_y = patch_ly / index.grid_size_y;
      int max_grid_y = patch_uy / index.grid_size_y;

      std::unordered_set<int> processed_nets; // 记录已处理的 net ID, 避免重复多次处理同一 net

      for (int x = min_grid_x; x <= max_grid_x; ++x) {
          for (int y = min_grid_y; y <= max_grid_y; ++y) {
              auto it = index.net_grid.find({x, y});
              if (it == index.net_grid.end()) continue;

              for (const auto& net : it->second) {
                if (processed_nets.count(net.id)) continue;
                processed_nets.insert(net.id);
                // 计算重叠面积
                const int overlap_lx = std::max(net.lx, patch_lx);
                const int overlap_ly = std::max(net.ly, patch_ly);
                const int overlap_ux = std::min(net.ux, patch_ux);
                const int overlap_uy = std::min(net.uy, patch_uy);

                // 有效重叠面积计算
                const int overlap_width = std::max(0, overlap_ux - overlap_lx);
                const int overlap_height = std::max(0, overlap_uy - overlap_ly);
                const int overlap_area = overlap_width * overlap_height;

                // 累加密度贡献（当且仅当有重叠时）
                if (overlap_area > 0) {
                    density += static_cast<double>(overlap_area) / patch_area;
                }

              }
          }
      }
      
      patch_net_density[patch_id] = density;
  }

  return patch_net_density;
}

std::map<int, int> DensityEval::patchMacroMargin(DensityCells cells, DensityRegion core, std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>> patch_coords)
{
    std::map<int, int> patch_macro_margin;

    std::vector<DensityCell> macros;
    for (const auto& cell : cells) {
      if (cell.type == "macro") {
        macros.push_back(cell);
      }
    }

    for (const auto& [patch_id, coord] : patch_coords) {
      auto [l_range, u_range] = coord;
      
      // 提取当前 patch 的物理边界
      const int patch_lx = l_range.first;
      const int patch_ly = l_range.second;
      const int patch_ux = u_range.first;
      const int patch_uy = u_range.second;
      // 计算核心区域的边界
      int32_t h_right = core.ux;
      int32_t h_left = core.lx;
      int32_t v_up = core.uy;
      int32_t v_down = core.ly;

      if (patch_ux <= h_left || patch_lx >= h_right || patch_uy <= v_down || patch_ly >= v_up) {
        patch_macro_margin[patch_id] = 0; // 确保所有情况都有赋值
        continue;
      }

      const int patch_width = patch_ux - patch_lx;
      const int patch_height = patch_uy - patch_ly;
      const int patch_area = patch_width * patch_height;

      bool overlap = false;
      int overlap_area = 0;
      int margin = 0;

      for (size_t j = 0; j < macros.size(); ++j) {
        int32_t rect_lx = std::max(patch_lx, macros[j].lx);
        int32_t rect_ly = std::max(patch_ly, macros[j].ly);
        int32_t rect_ux = std::min(patch_ux, macros[j].lx + macros[j].width);
        int32_t rect_uy = std::min(patch_uy, macros[j].ly + macros[j].height);
        if (rect_lx < rect_ux && rect_ly < rect_uy) {
          overlap_area += (std::min(patch_ux, macros[j].lx + macros[j].width) - std::max(patch_lx, macros[j].lx))
                          * (std::min(patch_uy, macros[j].ly + macros[j].height) - std::max(patch_ly, macros[j].ly));
        }
        if (overlap_area > 0.5 * patch_area) {
          overlap = true;
          break;
        }
      }
      
      if (!overlap) {
        for (size_t j = 0; j < macros.size(); ++j) {
          int32_t macro_middle_x = macros[j].lx + macros[j].width * 0.5;
          int32_t macro_middle_y = macros[j].ly + macros[j].height * 0.5;
          int32_t grid_middle_x = (patch_lx + patch_ux) * 0.5;
          int32_t grid_middle_y = (patch_ly + patch_uy) * 0.5;
          if (grid_middle_y >= macros[j].ly && grid_middle_y <= macros[j].ly + macros[j].height) {
            if (macro_middle_x > grid_middle_x) {
              h_right = std::min(h_right, macros[j].lx);
            } else {
              h_left = std::max(h_left, macros[j].lx + macros[j].width);
            }
          }
          if (grid_middle_x >= macros[j].lx && grid_middle_x <= macros[j].lx + macros[j].width) {
            if (macro_middle_y > grid_middle_y) {
              v_up = std::min(v_up, macros[j].ly);
            } else {
              v_down = std::max(v_down, macros[j].ly + macros[j].height);
            }
          }
        }
        margin = h_right - h_left + v_up - v_down;
      }

      patch_macro_margin[patch_id] = margin;
    }
    return patch_macro_margin;
}

DensityValue DensityEval::calCellDensity(int bin_cnt_x, int bin_cnt_y, const std::string& save_path)
{
  DensityRegion region = getDensityRegion();
  DensityCells cells = getDensityCells();

  std::vector<std::vector<double>> density_grid(bin_cnt_y, std::vector<double>(bin_cnt_x, 0.0));

  double grid_size_x = static_cast<double>(region.ux - region.lx) / bin_cnt_x;
  double grid_size_y = static_cast<double>(region.uy - region.ly) / bin_cnt_y;

  for (const auto& cell : cells) {
    int32_t start_row = std::max(0, static_cast<int32_t>((cell.ly - region.ly) / grid_size_y));
    int32_t end_row = std::min(bin_cnt_y - 1, static_cast<int32_t>((cell.ly + cell.height - region.ly) / grid_size_y));

    int32_t start_col = std::max(0, static_cast<int32_t>((cell.lx - region.lx) / grid_size_x));
    int32_t end_col = std::min(bin_cnt_x - 1, static_cast<int32_t>((cell.lx + cell.width - region.lx) / grid_size_x));

    for (int32_t row = start_row; row <= end_row; ++row) {
      for (int32_t col = start_col; col <= end_col; ++col) {
        double grid_lx = region.lx + col * grid_size_x;
        double grid_ly = region.ly + row * grid_size_y;
        double grid_ux = std::min(region.lx + (col + 1) * grid_size_x, static_cast<double>(region.ux));
        double grid_uy = std::min(region.ly + (row + 1) * grid_size_y, static_cast<double>(region.uy));

        double overlap_lx = std::max(static_cast<double>(cell.lx), grid_lx);
        double overlap_ly = std::max(static_cast<double>(cell.ly), grid_ly);
        double overlap_ux = std::min(static_cast<double>(cell.lx + cell.width), grid_ux);
        double overlap_uy = std::min(static_cast<double>(cell.ly + cell.height), grid_uy);

        double overlap_area = std::max(0.0, overlap_ux - overlap_lx) * std::max(0.0, overlap_uy - overlap_ly);
        double grid_area = (grid_ux - grid_lx) * (grid_uy - grid_ly);

        density_grid[row][col] += overlap_area / grid_area;
      }
    }
  }

  if (!save_path.empty()) {
    std::ofstream csv_file(save_path);
    if (csv_file.is_open()) {
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
    }
  }

  double total_density = 0.0;
  double max_density = 0.0;
  
  for (const auto& row : density_grid) {
    for (double density : row) {
      total_density += density;
      max_density = std::max(max_density, density);
    }
  }
  
  DensityValue result;
  result.max_density = max_density;
  result.avg_density = total_density / (bin_cnt_x * bin_cnt_y);
  
  return result;
}

DensityValue DensityEval::calPinDensity(int bin_cnt_x, int bin_cnt_y, const std::string& save_path)
{
  DensityRegion region = getDensityRegion();
  DensityPins pins = getDensityPins();

  std::vector<std::vector<double>> density_grid(bin_cnt_y, std::vector<double>(bin_cnt_x, 0.0));

  double grid_size_x = static_cast<double>(region.ux - region.lx) / bin_cnt_x;
  double grid_size_y = static_cast<double>(region.uy - region.ly) / bin_cnt_y;

  for (const auto& pin : pins) {
    int32_t col = static_cast<int32_t>((pin.lx - region.lx) / grid_size_x);
    int32_t row = static_cast<int32_t>((pin.ly - region.ly) / grid_size_y);

    if (col >= 0 && col < bin_cnt_x && row >= 0 && row < bin_cnt_y) {
      density_grid[row][col] += 1.0;
    }
  }

  if (!save_path.empty()) {
    std::ofstream csv_file(save_path);
    if (csv_file.is_open()) {
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
    }
  }

  double total_density = 0.0;
  double max_density = 0.0;
  
  for (const auto& row : density_grid) {
    for (double density : row) {
      total_density += density;
      max_density = std::max(max_density, density);
    }
  }
  
  DensityValue result;
  result.max_density = max_density;
  result.avg_density = total_density / (bin_cnt_x * bin_cnt_y);
  
  return result;
}

DensityValue DensityEval::calNetDensity(int bin_cnt_x, int bin_cnt_y, const std::string& save_path)
{
  DensityRegion region = getDensityRegion();
  DensityNets nets = getDensityNets();

  std::vector<std::vector<double>> density_grid(bin_cnt_y, std::vector<double>(bin_cnt_x, 0.0));

  double grid_size_x = static_cast<double>(region.ux - region.lx) / bin_cnt_x;
  double grid_size_y = static_cast<double>(region.uy - region.ly) / bin_cnt_y;

  for (const auto& net : nets) {
    int32_t start_col = std::max(0, static_cast<int32_t>((net.lx - region.lx) / grid_size_x));
    int32_t end_col = std::min(bin_cnt_x - 1, static_cast<int32_t>((net.ux - region.lx) / grid_size_x));
    int32_t start_row = std::max(0, static_cast<int32_t>((net.ly - region.ly) / grid_size_y));
    int32_t end_row = std::min(bin_cnt_y - 1, static_cast<int32_t>((net.uy - region.ly) / grid_size_y));

    bool is_local = (start_col == end_col) && (start_row == end_row);

    if (is_local) {
      density_grid[start_row][start_col] += 1.0;
    } else {
      for (int32_t row = start_row; row <= end_row; ++row) {
        for (int32_t col = start_col; col <= end_col; ++col) {
          density_grid[row][col] += 1.0;
        }
      }
    }
  }

  if (!save_path.empty()) {
    std::ofstream csv_file(save_path);
    if (csv_file.is_open()) {
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
    }
  }

  double total_density = 0.0;
  double max_density = 0.0;
  
  for (const auto& row : density_grid) {
    for (double density : row) {
      total_density += density;
      max_density = std::max(max_density, density);
    }
  }
  
  DensityValue result;
  result.max_density = max_density;
  result.avg_density = total_density / (bin_cnt_x * bin_cnt_y);
  
  return result;
}


}  // namespace ieval
