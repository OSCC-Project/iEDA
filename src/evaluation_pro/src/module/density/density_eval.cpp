#include "density_eval.h"

#include <unistd.h>

#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace ieval {

DensityEval::DensityEval()
{
}

DensityEval::~DensityEval()
{
}

std::string DensityEval::evalMacroDensity(DensityCells cells, DensityRegion region, int32_t grid_size)
{
  return evalDensity(cells, region, grid_size, "macro", "macro_density.csv");
}

std::string DensityEval::evalStdCellDensity(DensityCells cells, DensityRegion region, int32_t grid_size)
{
  return evalDensity(cells, region, grid_size, "stdcell", "stdcell_density.csv");
}

std::string DensityEval::evalAllCellDensity(DensityCells cells, DensityRegion region, int32_t grid_size)
{
  return evalDensity(cells, region, grid_size, "all", "allcell_density.csv");
}

std::string DensityEval::evalMacroPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size)
{
  return evalPinDensity(pins, region, grid_size, "macro", "macro_pin_density.csv");
}

std::string DensityEval::evalStdCellPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size)
{
  return evalPinDensity(pins, region, grid_size, "stdcell", "stdcell_pin_density.csv");
}

std::string DensityEval::evalAllCellPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size)
{
  return evalPinDensity(pins, region, grid_size, "all", "allcell_pin_density.csv");
}

std::string DensityEval::evalLocalNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size)
{
  return evalNetDensity(nets, region, grid_size, "local", "local_net_density.csv");
}

std::string DensityEval::evalGlobalNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size)
{
  return evalNetDensity(nets, region, grid_size, "global", "global_net_density.csv");
}

std::string DensityEval::evalAllNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size)
{
  return evalNetDensity(nets, region, grid_size, "all", "allnet_density.csv");
}

std::string DensityEval::reportMacroDensity(float threshold)
{
  return "macro_density_report.csv";
}

std::string DensityEval::reportStdCellDensity(float threshold)
{
  return "stdcell_density_report.csv";
}

std::string DensityEval::reportAllCellDensity(float threshold)
{
  return "allcell_density_report.csv";
}

std::string DensityEval::reportMacroPinDensity(float threshold)
{
  return "macro_pin_density_report.csv";
}

std::string DensityEval::reportStdCellPinDensity(float threshold)
{
  return "stdcell_pin_density_report.csv";
}

std::string DensityEval::reportAllCellPinDensity(float threshold)
{
  return "allcell_pin_density_report.csv";
}

std::string DensityEval::reportLocalNetDensity(float threshold)
{
  return "local_net_density_report.csv";
}

std::string DensityEval::reportGlobalNetDensity(float threshold)
{
  return "global_net_density_report.csv";
}

std::string DensityEval::reportAllNetDensity(float threshold)
{
  return "allnet_density_report.csv";
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

std::string DensityEval::evalPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size, std::string pin_type,
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

  for (int32_t row = 0; row < grid_rows; ++row) {
    for (int32_t col = 0; col < grid_cols; ++col) {
      int32_t grid_lx = region.lx + col * grid_size;
      int32_t grid_ly = region.ly + row * grid_size;
      int32_t grid_ux = std::min(region.lx + (col + 1) * grid_size, region.ux);
      int32_t grid_uy = std::min(region.ly + (row + 1) * grid_size, region.uy);

      double grid_area = static_cast<double>((grid_ux - grid_lx) * (grid_uy - grid_ly));
      density_grid[row][col] = static_cast<double>(pin_count[row][col]) / grid_area;
    }
  }

  ofstream csv_file(output_filename);

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

std::string DensityEval::evalNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size, std::string net_type,
                                        std::string output_filename)
{
  int32_t grid_cols = (region.ux - region.lx + grid_size - 1) / grid_size;
  int32_t grid_rows = (region.uy - region.ly + grid_size - 1) / grid_size;

  vector<vector<int>> net_count(grid_rows, vector<int>(grid_cols, 0));

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

  ofstream csv_file(output_filename);

  for (int32_t row = grid_rows - 1; row >= 0; --row) {
    for (int32_t col = 0; col < grid_cols; ++col) {
      int32_t grid_lx = region.lx + col * grid_size;
      int32_t grid_ly = region.ly + row * grid_size;
      int32_t grid_ux = std::min(region.lx + (col + 1) * grid_size, region.ux);
      int32_t grid_uy = std::min(region.ly + (row + 1) * grid_size, region.uy);

      double grid_area = static_cast<double>((grid_ux - grid_lx) * (grid_uy - grid_ly));

      double density = net_count[row][col] / grid_area;

      csv_file << fixed << setprecision(6) << density;
      if (col < grid_cols - 1)
        csv_file << ",";
    }
    csv_file << "\n";
  }

  csv_file.close();

  return getAbsoluteFilePath(output_filename);
}

std::string DensityEval::getAbsoluteFilePath(std::string filename)
{
  char currentPath[PATH_MAX];
  if (getcwd(currentPath, sizeof(currentPath)) != nullptr) {
    std::string workingDir(currentPath);

    if (filename[0] == '/') {
      return filename;
    }

    return workingDir + "/" + filename;
  }

  char resolvedPath[PATH_MAX];
  if (realpath(filename.c_str(), resolvedPath) != nullptr) {
    return std::string(resolvedPath);
  }
  return filename;
}

}  // namespace ieval
