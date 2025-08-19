/*
 * @FilePath: density_eval.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#pragma once

#include "density_db.h"
#include <map>
#include <unordered_map>
#include <vector>
#include <utility> 
namespace ieval {

struct MarginGrid
{
  int32_t lx;
  int32_t ly;
  int32_t ux;
  int32_t uy;
  int32_t margin;
};

// 辅助哈希函数
struct DensityPairHash {
  template <class T1, class T2>
  std::size_t operator () (const std::pair<T1,T2> &p) const {
      auto h1 = std::hash<T1>{}(p.first);
      auto h2 = std::hash<T2>{}(p.second);
      return h1 ^ (h2 << 1);
  }
};

// 网格索引结构体
struct DensityGridIndex {
  int grid_size_x = 2000; // 默认x方向网格大小
  int grid_size_y = 2000; // 默认y方向网格大小
  std::unordered_map<std::pair<int, int>, DensityCells, DensityPairHash> cell_grid;
  std::unordered_map<std::pair<int, int>, DensityPins, DensityPairHash> pin_grid;
  std::unordered_map<std::pair<int, int>, DensityNets, DensityPairHash> net_grid;
};

class DensityEval
{
 public:
  DensityEval();
  ~DensityEval();
  static DensityEval* getInst();
  static void destroyInst();

  std::string evalMacroDensity(DensityCells cells, DensityRegion region, int32_t grid_size, std::string stage);
  std::string evalStdCellDensity(DensityCells cells, DensityRegion region, int32_t grid_size, std::string stage);
  std::string evalAllCellDensity(DensityCells cells, DensityRegion region, int32_t grid_size, std::string stage);

  std::string evalMacroPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size, std::string stage, bool neighbor);
  std::string evalStdCellPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size, std::string stage, bool neighbor);
  std::string evalAllCellPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size, std::string stage, bool neighbor);

  std::string evalLocalNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size, std::string stage, bool neighbor);
  std::string evalGlobalNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size, std::string stage, bool neighbor);
  std::string evalAllNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size, std::string stage, bool neighbor);

  std::string evalHorizonMargin(DensityCells cells, DensityRegion die, DensityRegion core, int32_t grid_size, std::string stage);
  std::string evalVerticalMargin(DensityCells cells, DensityRegion die, DensityRegion core, int32_t grid_size, std::string stage);
  std::string evalAllMargin(DensityCells cells, DensityRegion die, DensityRegion core, int32_t grid_size, std::string stage);

  void initIDB();
  void destroyIDB();
  void initIDBCells();
  void initIDBNets();
  void initIDBRegion();

  DensityCells getDensityCells();
  DensityPins getDensityPins();
  DensityNets getDensityNets();
  DensityRegion getDensityRegion();
  DensityRegion getDensityRegionCore();

  int32_t getRowHeight();

  std::map<int, int> patchPinDensity(DensityPins pins, std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>> patch_coords);
  std::map<int, double> patchCellDensity(DensityCells cells, std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>> patch_coords);
  std::map<int, double> patchNetDensity(DensityNets nets, std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>> patch_coords);
  std::map<int, int> patchMacroMargin(DensityCells cells, DensityRegion core, std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>> patch_coords);

  DensityValue calCellDensity(int bin_cnt_x = 256, int bin_cnt_y = 256, const std::string& save_path = "");
  DensityValue calPinDensity(int bin_cnt_x = 256, int bin_cnt_y = 256, const std::string& save_path = "");
  DensityValue calNetDensity(int bin_cnt_x = 256, int bin_cnt_y = 256, const std::string& save_path = "");

 private:
  static DensityEval* _density_eval;

  std::string evalDensity(DensityCells cells, DensityRegion region, int32_t grid_size, std::string cell_type, std::string output_filename);
  std::string evalPinDensity(DensityPins pins, DensityRegion region, int32_t grid_size, bool neighbor, std::string pin_type,
                             std::string output_filename);
  std::string evalNetDensity(DensityNets nets, DensityRegion region, int32_t grid_size, bool neighbor, std::string net_type,
                             std::string output_filename);
  std::string evalMargin(DensityCells cells, DensityRegion die, DensityRegion core, int32_t grid_size, std::string margin_type,
                         std::string output_filename);
  std::vector<MarginGrid> initMarginGrid(DensityRegion die, int32_t grid_size);
};

}  // namespace ieval