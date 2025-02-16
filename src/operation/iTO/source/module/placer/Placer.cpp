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
#include "Placer.h"

#include "IdbRow.h"
#include "builder.h"
#include "data_manager.h"
#include "idm.h"

#include <algorithm>

using namespace std;

namespace ito {

Placer* Placer::_instance = nullptr;
Placer::Placer()
{
  initRow();
}

Placer::~Placer()
{
}

Placer* Placer::get_instance()
{
  static std::mutex mt;
  if (_instance == nullptr) {
    std::lock_guard<std::mutex> lock(mt);
    if (_instance == nullptr) {
      _instance = new Placer();
    }
  }
  return _instance;
}

void Placer::destroy_instance()
{
  if (_instance != nullptr) {
    delete _instance;
    _instance = nullptr;
  }
}

void Placer::initRow()
{
  _row_space = toDmInst->init_placer();
  _row_height = toDmInst->get_site_height();
  _site_width = toDmInst->get_site_width();
}

/**
 * @brief Find a location that is closest to and can be placed near the desired location
 *
 * @param master_width
 * @param loc_x
 * @param loc_y
 * @return pair<int, int>
 */
std::pair<int, int> Placer::findNearestSpace(unsigned int master_width, int loc_x, int loc_y)
{
  // Which "rows" to search for
  int row_idx = (loc_y - toDmInst->get_core().get_y_min()) / _row_height;
  row_idx = std::max(0, row_idx);

  UsedSpace* best_opt = nullptr;
  int best_dist = INT_MAX;
  int update_loc_x = loc_x;
  int update_loc_y = loc_y;

  int find_row = 0;
  while (find_row < 40) {
    if (find_row > 0 && (find_row > INT_MAX / _row_height)) {
      break;
    }
    int current_height = find_row * _row_height;
    if (current_height >= best_dist) {
      break;
    }
  
    // 向上查找 (row_idx + find_row)
    int sum_up = row_idx + find_row;
    if (sum_up >= 0 && static_cast<size_t>(sum_up) < _row_space.size()) {  // 严格索引检查
      auto opt_up = findNearestRowLegalSpace(sum_up, master_width, loc_x, loc_y);
      if (opt_up.first && opt_up.second < best_dist) {
        best_opt = opt_up.first;
        best_dist = opt_up.second;
        update_loc_y = (sum_up * _row_height) + toDmInst->get_core().get_y_min();
      }
    }
  
    // 向下查找 (row_idx - find_row)
    if (find_row != 0) {
      int sum_down = row_idx - find_row;
      if (sum_down >= 0 && static_cast<size_t>(sum_down) < _row_space.size()) {  // 严格索引检查
        auto opt_down = findNearestRowLegalSpace(sum_down, master_width, loc_x, loc_y);
        if (opt_down.first && opt_down.second < best_dist) {
          best_opt = opt_down.first;
          best_dist = opt_down.second;
          update_loc_y = (sum_down * _row_height) + toDmInst->get_core().get_y_min();
        }
      }
    }
  
    find_row++;
  }

  if (best_opt) {
    update_loc_x = best_opt->begin();
    if (update_loc_x == -1 || update_loc_y == -1) {
      return make_pair(loc_x, loc_y);
    }
    return make_pair(update_loc_x, update_loc_y);
  }
  cout << "[Placer::findNearestSpace] Can't find suitable space to place the buffer "
          "in the specified search area."
       << endl;
  // If can't find suitable space to place the buffer, do not change
  return make_pair(update_loc_x, update_loc_y);
}

std::pair<UsedSpace*, int> Placer::findNearestSpace(std::vector<UsedSpace*> options, int begin, int end, int dis_margin)
{
  int best_dist = INT_MAX;
  UsedSpace* best_opt = nullptr;
  for (UsedSpace* opt : options) {
    if (opt->isOverlaps(begin, end)) {
      // best_opt = new UsedSpace(begin, end);
      // return make_pair(best_opt, 0 + dis_margin);
      int placed_x = placeAlignWithSite(UsedSpace(begin, end));
      int x_dis = abs(placed_x - begin);
      best_opt = new UsedSpace(placed_x, placed_x + (end - begin));
      return make_pair(best_opt, x_dis + dis_margin);
    }

    // since "begin - end" is always less than or equal to opt->length()
    assert(opt->length() >= end - begin);
    // int dis;
    if (begin < opt->begin()) {
      int dis = opt->begin() - begin;
      if (dis < best_dist) {
        best_dist = dis;
        best_opt = new UsedSpace(opt->begin(), end + dis);
      }
    } else {
      int dis = end - opt->end();
      if (dis < best_dist) {
        best_dist = dis;
        best_opt = new UsedSpace(begin - dis, opt->end());
      }
    }
  }
  return make_pair(best_opt, best_dist + dis_margin);
}

int Placer::placeAlignWithSite(UsedSpace option)
{
  int begin = option.begin();
  int core_min_x = toDmInst->get_core().get_x_min();
  int site_num = (begin - core_min_x) / _site_width;
  int placed_x1 = (site_num * _site_width) + core_min_x;
  int placed_x2 = ((site_num + 1) * _site_width) + core_min_x;
  int placed_x;
  if (abs(placed_x1 - begin) < abs(placed_x2 - begin)) {
    placed_x = placed_x1;
  } else {
    placed_x = placed_x2;
  }
  return placed_x;
}

IdbRow* Placer::findRow(int loc_y)
{
  IdbLayout* idb_layout = dmInst->get_idb_layout();
  IdbRows* rows = idb_layout->get_rows();

  for (auto row : rows->get_row_list()) {
    if (row->get_bounding_box()->get_low_y() == loc_y) {
      return row;
    }
  }
  return nullptr;
}

/**
 * @brief update the "Row" information after insert an instance
 *
 * @param master_width instance width
 * @param loc the location of inserted instance
 */
void Placer::updateRow(unsigned int master_width, int loc_x, int loc_y)
{
  // On which "Row" is the buffer inserted
  int row_idx = (loc_y - toDmInst->get_core().get_y_min()) / _row_height;
  _row_space[row_idx]->addUsedSpace(loc_x, loc_x + master_width);
}

std::pair<UsedSpace*, int> Placer::findNearestRowLegalSpace(int row_idx, unsigned int master_width, int loc_x, int loc_y)
{
  std::vector<UsedSpace*> alter_option = _row_space[row_idx]->searchFeasiblePlace(loc_x, loc_x + master_width, 3);
  int row_height1 = (row_idx * _row_height) + toDmInst->get_core().get_y_min();
  // The distance needed to move to the row
  int dis_margin1 = abs(row_height1 - loc_y);
  std::pair<UsedSpace*, int> opt = findNearestSpace(alter_option, loc_x, loc_x + master_width, dis_margin1);
  return opt;
}

}  // namespace ito
