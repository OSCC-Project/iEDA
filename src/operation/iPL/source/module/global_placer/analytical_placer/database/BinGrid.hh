/*
 * @Author: S.J Chen
 * @Date: 2022-03-09 21:32:52
 * @LastEditTime: 2022-10-27 19:33:55
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/operator/global_placer/nesterov_place/database/BinGrid.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_OPERATOR_NESTEROV_PLACE_DATABASE_BIN_GRID_H
#define IPL_OPERATOR_NESTEROV_PLACE_DATABASE_BIN_GRID_H

#include <unordered_map>
#include <vector>

#include "GridManager.hh"
#include "NesInstance.hh"
#include "omp.h"

namespace ipl {

struct AreaInfo
{
  AreaInfo();

  void reset();

  int64_t macro_area;
  int64_t stdcell_area;
  int64_t filler_area;
};
inline AreaInfo::AreaInfo() : macro_area(int64_t(0)), stdcell_area(int64_t(0)), filler_area(int64_t(0))
{
}
inline void AreaInfo::reset()
{
  macro_area = 0;
  stdcell_area = 0;
  filler_area = 0;
}

class BinGrid
{
 public:
  BinGrid() = delete;
  explicit BinGrid(GridManager* grid_manager);
  BinGrid(const BinGrid&) = delete;
  BinGrid(BinGrid&&) = delete;
  ~BinGrid() = default;

  void updateBinGrid(std::vector<NesInstance*>& nInst_list, int32_t thread_num);
  int64_t obtainOverflowAreaWithoutFiller();

  std::vector<Grid*> obtainOccupiedBinList(NesInstance* nInst);  // TODO.
  std::vector<NesInstance*> obtainOccupiednInstList(Grid* bin);  // TODO.

 private:
  GridManager* _grid_manager;

  std::unordered_multimap<Grid*, NesInstance*> _bin_to_nInsts;
  std::unordered_map<Grid*, AreaInfo> _bin_to_area;
  void resetBinToArea();

  void addBinnInstConnection(Grid* bin, NesInstance* nInst);
  void addBinMacroAreaInfo(Grid* bin, int64_t macro_area);
  void addBinStdcellAreaInfo(Grid* bin, int64_t stdcell_area);
  void addBinFillerAreaInfo(Grid* bin, int64_t filler_area);
};
inline BinGrid::BinGrid(GridManager* grid_manager) : _grid_manager(grid_manager)
{
}

inline void BinGrid::resetBinToArea()
{
  for (auto& pair : _bin_to_area) {
    auto& area_info = pair.second;
    area_info.reset();
  }
}

inline void BinGrid::addBinnInstConnection(Grid* bin, NesInstance* nInst)
{
  _bin_to_nInsts.emplace(bin, nInst);
}

inline void BinGrid::updateBinGrid(std::vector<NesInstance*>& nInst_list, int32_t thread_num)
{
  _grid_manager->clearAllOccupiedArea();
  // _bin_to_nInsts.clear();
  resetBinToArea();

  for (auto* nInst : nInst_list) {
    if (nInst->isFixed()) {
      continue;
    }

    auto nInst_density_shape = std::move(nInst->get_density_shape());

    std::vector<Grid*> overlap_grid_list;
    _grid_manager->obtainOverlapGridList(overlap_grid_list, nInst_density_shape);

    for (auto* grid : overlap_grid_list) {
      // addBinnInstConnection(grid, nInst);

      int64_t overlap_area = _grid_manager->obtainOverlapArea(grid, nInst_density_shape);
      if (nInst->isMacro()) {
        int64_t macro_area = static_cast<int64_t>(overlap_area * nInst->get_density_scale() * grid->get_available_ratio());
        _bin_to_area[grid].macro_area += macro_area;
        grid->add_area(macro_area);
      } else if (nInst->isFiller()) {
        int64_t filler_area = static_cast<int64_t>(overlap_area * nInst->get_density_scale());
        _bin_to_area[grid].filler_area += filler_area;
        grid->add_area(filler_area);
      } else {
        int64_t stdcell_area = static_cast<int64_t>(overlap_area * nInst->get_density_scale());
        _bin_to_area[grid].stdcell_area += stdcell_area;
        grid->add_area(stdcell_area);
      }
    }
  }
}

inline int64_t BinGrid::obtainOverflowAreaWithoutFiller()
{
  int64_t overflow_area = 0;
  for (auto* row : _grid_manager->get_row_list()) {
    for (auto* grid : row->get_grid_list()) {
      auto occupy_it = _bin_to_area.find(grid);
      if (occupy_it != _bin_to_area.end()) {
        auto& area_info = occupy_it->second;
        int64_t relative_area = area_info.macro_area + area_info.stdcell_area;

        // bin target area.
        int64_t bin_area = static_cast<int64_t>(grid->get_width()) * static_cast<int64_t>(grid->get_height());
        int64_t target_area = static_cast<int64_t>(bin_area * grid->get_available_ratio());

        overflow_area += std::max(int64_t(0), relative_area - target_area);
      }
    }
  }

  return overflow_area;
}

}  // namespace ipl

#endif
