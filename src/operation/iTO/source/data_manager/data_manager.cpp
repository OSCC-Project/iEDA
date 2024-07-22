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
#include "data_manager.h"

#include "Reporter.h"
#include "ToConfig.h"
#include "Utility.h"
#include "idm.h"

namespace ito {
ToDataManager* ToDataManager::_instance = nullptr;

ToDataManager* ToDataManager::get_instance()
{
  static std::mutex mt;
  if (_instance == nullptr) {
    std::lock_guard<std::mutex> lock(mt);
    if (_instance == nullptr) {
      _instance = new ToDataManager();
      _instance->initData();
    }
  }
  return _instance;
}

void ToDataManager::destroy_instance()
{
  if (_instance != nullptr) {
    delete _instance;
    _instance = nullptr;
  }
}

void ToDataManager::initData()
{
  /// db
  IdbLayout* idb_layout = dmInst->get_idb_layout();
  IdbDesign* idb_design = dmInst->get_idb_design();
  IdbCore* idb_core = idb_layout->get_core();
  IdbRect* idb_rect = idb_core->get_bounding_box();

  _dbu = idb_layout->get_units()->get_micron_dbu();
  _core = Rectangle(idb_rect->get_low_x(), idb_rect->get_high_x(), idb_rect->get_low_y(), idb_rect->get_high_y());
  _layout = new Layout(idb_design);
  _design_area = calculateDesignArea(_layout, _dbu);

  // log report
  std::string report_path = toConfig->get_report_file();
  toRptInst->init(report_path);
  toRptInst->reportTime(true);
  toRptInst->get_ofstream() << "Report file: " << report_path << endl << endl;
  toRptInst->get_ofstream().close();
}

bool ToDataManager::reachMaxArea()
{
  double max_core_utilization = toConfig->get_max_core_utilization();
  // initBlock();
  double core_area = calculateCoreArea(_core, _dbu);
  double max_area_utilize = core_area * max_core_utilization;
  return (_design_area > max_area_utilize);
}

int ToDataManager::get_site_width()
{
  IdbRows* rows = dmInst->get_idb_layout()->get_rows();

  return rows->get_row_list()[0]->get_site()->get_width();
}

int ToDataManager::get_site_height()
{
  IdbRows* rows = dmInst->get_idb_layout()->get_rows();

  return rows->get_row_list()[0]->get_site()->get_height();
}

std::vector<RowSpacing*> ToDataManager::init_placer()
{
  std::vector<RowSpacing*> row_space;
  IdbLayout* idb_layout = dmInst->get_idb_layout();
  auto idb_core = toDmInst->get_core();

  IdbRows* rows = idb_layout->get_rows();

  int site_width = get_site_width();
  int site_height = get_site_height();

  unsigned row_count = rows->get_row_num();

  // init row spacing
  for (unsigned i = 0; i < row_count; i++) {
    RowSpacing* row_init = new RowSpacing(idb_core.get_x_min(), idb_core.get_x_max());
    row_space.push_back(row_init);
  }
  row_space.resize(row_count);

  // Traverse over all instances and update the each row spacing
  IdbDesign* idb_design = dmInst->get_idb_design();

  for (auto inst : idb_design->get_instance_list()->get_instance_list()) {
    // inst size
    int master_width = inst->get_cell_master()->get_width();
    int master_height = inst->get_cell_master()->get_height();

    // inst location
    int x_min = inst->get_coordinate()->get_x();
    int y_min = inst->get_coordinate()->get_y();
    // out of core
    if (!idb_core.overlaps(x_min, y_min)) {
      continue;
    }

    // which "Row" is it in
    int row_index = (y_min - idb_core.get_y_min()) / site_height;
    int occupied_row_num = master_height / site_height;
    occupied_row_num = (y_min - idb_core.get_y_min()) % site_height == 0 ? occupied_row_num : occupied_row_num + 1;
    // the space occupied by the "master"
    int begin = x_min;
    int end = begin + master_width;

    // update row space
    for (int i = 0; i != occupied_row_num; i++) {
      row_space[row_index + i]->addUsedSpace(begin, end);
    }
  }

  // Block obstacle
  for (auto block : idb_design->get_blockage_list()->get_blockage_list()) {
    // Block size
    if (!block->is_palcement_blockage()) {
      continue;
    }

    IdbRect* rect = block->get_rect_list()[0];

    // // Block location
    int x_min = rect->get_low_x();
    int y_min = rect->get_low_y();
    int x_max = rect->get_high_x();
    int y_max = rect->get_high_y();

    int master_height = y_max - y_min;

    // out of core
    if (!idb_core.overlaps(x_min, y_min)) {
      continue;
    }

    // which "Row" is it in
    int row_index = (y_min - idb_core.get_y_min()) / site_height;
    int occupied_row_num = int(master_height / site_height) + 1;
    occupied_row_num = (y_min - idb_core.get_y_min()) % site_height == 0 ? occupied_row_num : occupied_row_num + 1;

    // the space occupied by the "blockage"
    int core_min_x = idb_core.get_x_min();
    int begin_site_num = (x_min - core_min_x) / site_width;
    int end_site_num = int((x_max - core_min_x) / site_width) + 1;
    x_min = (begin_site_num * site_width) + core_min_x;
    x_max = (end_site_num * site_width) + core_min_x;

    int begin = x_min > idb_core.get_x_min() + site_width ? x_min - site_width : x_min;
    int end = x_max < idb_core.get_x_max() - site_width ? x_max + site_width : x_max;

    // update row space
    for (int i = 0; i != occupied_row_num; i++) {
      if (row_index + i < (int) row_space.size() - 1) {
        row_space[row_index + i]->addUsedSpace(begin, end);
      }
    }
  }

  return row_space;
}

}  // namespace ito
