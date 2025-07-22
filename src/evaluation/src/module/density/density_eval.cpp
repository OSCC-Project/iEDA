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

std::string DensityEval::evalHorizonMargin(DensityCells cells, DensityRegion die, DensityRegion core, int32_t grid_size)
{
  return evalMargin(cells, die, core, grid_size, "horizontal", "horizontal_margin.csv");
}

std::string DensityEval::evalVerticalMargin(DensityCells cells, DensityRegion die, DensityRegion core, int32_t grid_size)
{
  return evalMargin(cells, die, core, grid_size, "vertical", "vertical_margin.csv");
}

std::string DensityEval::evalAllMargin(DensityCells cells, DensityRegion die, DensityRegion core, int32_t grid_size)
{
  return evalMargin(cells, die, core, grid_size, "union", "union_margin.csv");
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

  std::string output_path = createDirPath("/density_map") + "/" + output_filename;
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

  std::string output_path;
  if (neighbor) {
    output_path = createDirPath("/density_map") + "/" + "neighbor_" + output_filename;
  } else {
    output_path = createDirPath("/density_map") + "/" + output_filename;
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
    output_path = createDirPath("/density_map") + "/" + "neighbor_" + output_filename;
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
    output_path = createDirPath("/density_map") + "/" + output_filename;
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

  std::string output_path = createDirPath("/margin_map") + "/" + output_filename;
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

}  // namespace ieval
