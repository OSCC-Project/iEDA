#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "IdbCellMaster.h"
#include "IdbDesign.h"
#include "IdbEnum.h"
#include "IdbInstance.h"
#include "IdbRow.h"
#include "ifp_enum.h"

using std::string;
using std::vector;
using std::pair;

namespace ifp {
class TapCellPlacer
{
 public:
  TapCellPlacer() = default;
  ~TapCellPlacer() = default;

  void insertTapCells(double distance, std::string tapcell_master);
  void insertEndCaps(std::string endcap_master);

 private:
  bool _init_macro = false;

  std::vector<std::pair<int32_t, std::pair<int32_t, int32_t>>> _filled_sites;

  int32_t _tapcell_start_index = 0;
  int32_t _tapcell_index = 0;

  // test func
  void initMacroLocation();
  int32_t find_macro_top_row(idb::IdbInstance* ins);
  int32_t find_macro_bottom_row(idb::IdbInstance* ins);
  int32_t find_rect_top_row(idb::IdbRect* rect);
  int32_t find_rect_bottom_row(idb::IdbRect* rect);
  // int32_t find_macro_overlap_site(idb::IdbInstance* ins);
  int32_t find_placement_blockage_overlap_site(idb::IdbBlockage* blk);
  int32_t find_index_in_row_fills(std::vector<pair<int32_t, vector<pair<int32_t, int32_t>>>> row_fills, int32_t lly);

  idb::IdbInstance* build_tapcell_instance(idb::IdbCellMaster* master, idb::IdbOrient orient, int32_t x, int32_t y, std::string prefix);
  std::map<std::pair<int32_t, int32_t>, vector<int32_t>> obtain_macro_outlines(idb::IdbRows* rows);
  std::pair<int32_t, int32_t> obtain_rows_min_max_x(idb::IdbRows* rows);
  int32_t make_site_location(int32_t x, int32_t site_width, int32_t diect, int32_t offset);
};
int obtain_pair_index(std::vector<std::pair<int32_t, std::vector<std::pair<int32_t, int32_t>>>> row_fills, int32_t y);
bool check_site_if_filled(int32_t x, int32_t width, idb::IdbOrient orient, std::vector<std::pair<int32_t, int32_t>> row_instances);
bool check_master_symmetry(idb::IdbCellMaster* master, idb::IdbOrient orient);
}  // namespace ifp