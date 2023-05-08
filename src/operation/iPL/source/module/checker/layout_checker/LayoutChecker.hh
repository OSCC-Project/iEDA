/*
 * @Author: S.J Chen
 * @Date: 2022-03-02 10:16:31
 * @LastEditTime: 2022-12-01 10:30:36
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/checker/LayoutChecker.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_CHECKER_LAYOUT_CHECKER_H
#define IPL_CHECKER_LAYOUT_CHECKER_H

#include <unordered_map>

#include "Log.hh"
#include "PlacerDB.hh"

namespace ipl {

class LayoutChecker
{
 public:
  LayoutChecker() = delete;
  LayoutChecker(PlacerDB* placer_db);
  LayoutChecker(const LayoutChecker&) = delete;
  LayoutChecker(LayoutChecker&&) = delete;
  ~LayoutChecker() = default;

  LayoutChecker& operator=(const LayoutChecker&) = delete;
  LayoutChecker& operator=(LayoutChecker&&) = delete;

  bool isAllPlacedInstInsideCore();
  bool isAllPlacedInstAlignRowSite();
  bool isAllPlacedInstAlignPower();
  bool isNoOverlapAmongInsts();

  std::vector<Instance*> obtainIllegalInstInsideCore();
  std::vector<Instance*> obtainIllegalInstAlignRowSite();
  std::vector<Instance*> obtainIllegalInstAlignPower();
  std::vector<std::vector<Instance*>> obtainOverlapInstClique();
  std::vector<Rectangle<int32_t>> obtainWhiteSiteList();

 private:
  PlacerDB* _placer_db;
  GridManager* _grid_manager;
  int32_t _row_height;
  int32_t _site_width;
  std::pair<int32_t, int32_t> _core_x_range;
  std::pair<int32_t, int32_t> _core_y_range;
  std::unordered_multimap<Instance*, Grid*> _inst_to_sites;
  std::unordered_multimap<Grid*, Instance*> _site_to_insts;

  bool checkInsideCore(Rectangle<int32_t> shape);
  bool checkAlignRowSite(Rectangle<int32_t> shape);
  bool checkAlignPower(Instance* inst);

  void updateSiteInstConnection();
  void addSiteInstConnection(Grid* site, Instance* inst);
  void clearSiteInstConnection();
  std::vector<Grid*> obtainOccupiedSiteList(Instance* inst);
  std::vector<Instance*> obtainOccupiedInstList(Grid* site);
  void connectInstSite(Instance* inst);

  Orient obtainLayoutRowOrient(GridRow* grid_row);
};

}  // namespace ipl

#endif