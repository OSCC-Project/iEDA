// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file
 * @author WenruiWu (lixq01@pcl.ac.cn)
 * @brief Filler;
 * @version 0.1
 * @date 2022-4-1
 **/

#ifndef IEDA_FILLER_H
#define IEDA_FILLER_H

#include <map>
#include <vector>

#include "PlacerDB.hh"
#include "config/FillerConfig.h"
#include "data/Rectangle.hh"

namespace ipl {
class MapFiller
{
 private:
  struct FillerSegment
  {
    int32_t l, r;
    // double  _now_sum; //total cells' width in seg
    bool operator<(const FillerSegment& b) const { return l < b.l; }
  };

  // placer db
  Design* _pl_design;
  const Layout* _pl_layout;

  // layout
  Rectangle<int32_t> _region;
  int32_t _row_count;
  int32_t _row_site_count;
  int32_t _site_width;
  int32_t _row_height;
  std::map<int32_t, Orient> _orient_map;

  // design
  std::vector<Rectangle<int32_t>> _blockage_list;
  std::vector<std::vector<FillerSegment>> _available_sites;
  std::vector<int32_t> _filler_count;
  std::vector<Cell*> _filler_master_list;
  std::vector<Instance*> _filler_inst_list;

  // config
  FillerConfig _filler_config;
  std::vector<std::vector<std::string>> _filler_group_list;
  int32_t _min_filler_width;

 public:
  explicit MapFiller(PlacerDB* placer_db, Config* config)
  {
    _pl_design = placer_db->get_design();
    _pl_layout = placer_db->get_layout();
    _filler_config = config->get_filler_config();
    _filler_group_list = _filler_config.get_filler_group_list();
    _min_filler_width = _filler_config.get_min_filler_width();
    _site_width = _pl_layout->get_site_width();
    _row_height = _pl_layout->get_row_height();
    for (auto row : _pl_layout->get_row_list())
      _orient_map[row->get_coordi().get_y()] = row->get_orient();
  }
  ~MapFiller() {}

  // main
  void mapFillerCell();
  // funtion
  void addFillerCell();
  void addFillerCellWithGroups();
  void addFillerCellWithoutGroups();
  void fixed_cell_assign();
  void writeBack();
  void clear();
  void init();
  void reset(Rectangle<int32_t> region, std::vector<Rectangle<int32_t>> blockage_list);
  bool isInstInside(Instance* inst, Rectangle<int32_t> rect);
  void add_filler_instance(int32_t inst_x, int32_t inst_y, std::string filler_name, Cell* filler_master);
  void findFillerMaster(std::vector<std::string> filler_name_list);
};
}  // namespace ipl
#endif