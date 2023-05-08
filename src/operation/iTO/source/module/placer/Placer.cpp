#include "Placer.h"
#include "builder.h"

using namespace std;

namespace ito {
void Placer::initRow() {
  IdbLayout *idb_layout = _idb_builder->get_lef_service()->get_layout();
  IdbRect   *idb_rect = idb_layout->get_core()->get_bounding_box();
  _core = Rectangle(idb_rect->get_low_x(), idb_rect->get_high_x(), idb_rect->get_low_y(),
                    idb_rect->get_high_y());

  IdbRows *rows = idb_layout->get_rows();

  IdbRow  *first_row = rows->get_row_list()[0];
  IdbSite *first_site = first_row->get_site();
  int      site_width = first_site->get_width();
  int      site_height = first_site->get_height();

  _site_width = site_width;
  _row_height = site_height;
  unsigned row_count = rows->get_row_num();

  // init row spacing
  for (unsigned i = 0; i < row_count; i++) {
    RowSpacing *row_init = new RowSpacing(_core.get_x_min(), _core.get_x_max());
    _row_space.push_back(row_init);
  }
  _row_space.resize(row_count);

  /**
   * @brief Traverse over all instances and update the each row spacing
   */
  IdbDefService *idb_def_service = _idb_builder->get_def_service();
  IdbDesign     *idb_design = idb_def_service->get_design();

  // ito::GDSwriter::writeGDS(_idb_builder, "/home/wuhongxi/output/GDS/hold.gds");
  for (auto inst : idb_design->get_instance_list()->get_instance_list()) {
    // inst size
    int master_width = inst->get_cell_master()->get_width();
    int master_height = inst->get_cell_master()->get_height();

    // inst location
    int x_min = inst->get_coordinate()->get_x();
    int y_min = inst->get_coordinate()->get_y();
    // out of core
    if (!_core.overlaps(x_min, y_min)) {
      continue;
    }

    // which "Row" is it in
    int row_index = (y_min - _core.get_y_min()) / site_height;
    int occupied_row_num = master_height / site_height;
    occupied_row_num = (y_min - _core.get_y_min()) % site_height == 0
                           ? occupied_row_num
                           : occupied_row_num + 1;
    // the space occupied by the "master"
    int begin = x_min;
    int end = begin + master_width;

    // update row space
    for (int i = 0; i != occupied_row_num; i++) {
      _row_space[row_index + i]->addUsedSpace(begin, end);
    }
  }

  // Block obstacle
  for (auto block : idb_design->get_blockage_list()->get_blockage_list()) {
    // Block size
    if (!block->is_palcement_blockage()) {
      continue;
    }

    IdbRect *rect = block->get_rect_list()[0];

    // // Block location
    int x_min = rect->get_low_x();
    int y_min = rect->get_low_y();
    int x_max = rect->get_high_x();
    int y_max = rect->get_high_y();

    int master_height = y_max - y_min;

    // out of core
    if (!_core.overlaps(x_min, y_min)) {
      continue;
    }

    // which "Row" is it in
    int row_index = (y_min - _core.get_y_min()) / site_height;
    int occupied_row_num = int(master_height / site_height) + 1;
    occupied_row_num = (y_min - _core.get_y_min()) % site_height == 0
                           ? occupied_row_num
                           : occupied_row_num + 1;

    // the space occupied by the "blockage"
    int core_min_x = _core.get_x_min();
    int begin_site_num = (x_min - core_min_x) / _site_width;
    int end_site_num = int((x_max - core_min_x) / _site_width) + 1;
    x_min = (begin_site_num * _site_width) + core_min_x;
    x_max = (end_site_num * _site_width) + core_min_x;

    int begin = x_min > _core.get_x_min() + _site_width ? x_min - _site_width : x_min;
    int end = x_max < _core.get_x_max() - _site_width ? x_max + _site_width : x_max;

    // update row space
    for (int i = 0; i != occupied_row_num; i++) {
      if (row_index + i < (int)_row_space.size() - 1) {
        _row_space[row_index + i]->addUsedSpace(begin, end);
      }
    }
  }
}

/**
 * @brief Find a location that is closest to and can be placed near the desired location
 *
 * @param master_width
 * @param loc_x
 * @param loc_y
 * @return pair<int, int>
 */
pair<int, int> Placer::findNearestSpace(unsigned int master_width, int loc_x, int loc_y) {
  // Which "rows" to search for
  int row_idx = (loc_y - _core.get_y_min()) / _row_height;

  UsedSpace *best_opt = nullptr;
  int        best_dist = INT_MAX;
  int        update_loc_x = loc_x;
  int        update_loc_y = loc_y;

  int find_row = 0;
  while ((find_row * _row_height) < best_dist && find_row < 40) {

    if (row_idx + find_row < (int)_row_space.size() - 1) {
      auto opt_up =
          findNearestRowLegalSpace(row_idx + find_row, master_width, loc_x, loc_y);
      if (opt_up.first && opt_up.second < best_dist) {
        best_opt = opt_up.first;
        best_dist = opt_up.second;
        update_loc_y = ((row_idx + find_row) * _row_height) + _core.get_y_min();
      }
    }

    if (row_idx - find_row > 0 && find_row != 0) {
      auto opt_down =
          findNearestRowLegalSpace(row_idx - find_row, master_width, loc_x, loc_y);
      // Choose the option that moves the least distance
      if (opt_down.first && opt_down.second < best_dist) {
        best_opt = opt_down.first;
        best_dist = opt_down.second;
        update_loc_y = ((row_idx - find_row) * _row_height) + _core.get_y_min();
      }
    }
    find_row++;
  }

  if (best_opt) {
    update_loc_x = best_opt->begin();
    return make_pair(update_loc_x, update_loc_y);
  }
  cout << "[Placer::findNearestSpace] Can't find suitable space to place the buffer "
          "in the specified search area."
       << endl;
  // If can't find suitable space to place the buffer, do not change
  return make_pair(update_loc_x, update_loc_y);
}

pair<UsedSpace *, int> Placer::findNearestSpace(vector<UsedSpace *> options, int begin,
                                                int end, int dis_margin) {
  int        best_dist = INT_MAX;
  UsedSpace *best_opt = nullptr;
  for (UsedSpace *opt : options) {
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

int Placer::placeAlignWithSite(UsedSpace option) {
  int begin = option.begin();
  int core_min_x = _core.get_x_min();
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

IdbRow *Placer::findRow(int loc_y) {
  IdbLefService *idb_lef_service = _idb_builder->get_lef_service();
  IdbLayout     *idb_layout = idb_lef_service->get_layout();

  IdbRows *rows = idb_layout->get_rows();

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
void Placer::updateRow(unsigned int master_width, int loc_x, int loc_y) {
  // On which "Row" is the buffer inserted
  int row_idx = (loc_y - _core.get_y_min()) / _row_height;
  _row_space[row_idx]->addUsedSpace(loc_x, loc_x + master_width);
}

} // namespace ito
