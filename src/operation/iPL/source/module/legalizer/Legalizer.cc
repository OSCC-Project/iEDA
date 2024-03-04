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

#include "Legalizer.hh"

#include "LGMethodCreator.hh"
#include "LGMethodInterface.hh"
#include "abacus/Abacus.hh"

namespace ipl {

void Legalizer::initLegalizer(Config* pl_config, PlacerDB* placer_db)
{
  initLGConfig(pl_config);
  initLGDatabase(placer_db);

  // change legalizer method
  ieda_solver::LGMethodCreator method_creator;
  _method = method_creator.createMethod();
  if (_method == nullptr) {
    LOG_ERROR << "Target legalizetion method has not been realized!";
  }
}

void Legalizer::initLGConfig(Config* pl_config)
{
  _config = pl_config->get_lg_config();
}

void Legalizer::initLGDatabase(PlacerDB* placer_db)
{
  _database._placer_db = placer_db;
  initLGLayout();
  updateInstanceList();
  initSegmentList();
}

void Legalizer::initLGLayout()
{
  const Layout* pl_layout = _database._placer_db->get_layout();

  auto core_shape = pl_layout->get_core_shape();
  int32_t row_height = pl_layout->get_row_height();
  int32_t row_num = std::floor(static_cast<double>(core_shape.get_height()) / row_height);

  // shift all element in core to (0,0)
  _database._shift_x = 0 - core_shape.get_ll_x();
  _database._shift_y = 0 - core_shape.get_ll_y();
  _database._lg_layout = new LGLayout(row_num, core_shape.get_ur_x() + _database._shift_x, core_shape.get_ur_y() + _database._shift_y);
  _database._lg_layout->set_dbu(pl_layout->get_database_unit());

  // arrange row to LGLayout _row_2d_list
  wrapRowList();

  // add LGLayout region list
  wrapRegionList();

  // add LGLayout cell list
  wrapCellList();
}

void Legalizer::wrapRowList()
{
  const Layout* pl_layout = _database._placer_db->get_layout();

  std::vector<std::vector<LGRow*>> row_2d_list;
  row_2d_list.resize(_database._lg_layout->get_row_num());
  for (auto* pl_row : pl_layout->get_row_list()) {
    auto* pl_site = pl_row->get_site();
    LGSite* row_site = new LGSite(pl_site->get_name());
    row_site->set_width(pl_site->get_site_width());
    row_site->set_height(pl_site->get_site_height());

    int32_t row_shift_x = pl_row->get_coordi().get_x() + _database._shift_x;
    int32_t row_shift_y = pl_row->get_coordi().get_y() + _database._shift_y;
    int32_t row_index = std::floor(static_cast<double>(row_shift_y) / pl_row->get_site_height());
    LGRow* row = new LGRow(pl_row->get_name(), row_site, pl_row->get_site_num());
    row->set_coordinate(row_shift_x, row_shift_y);
    row->set_orient(pl_site->get_orient());

    row_2d_list[row_index].push_back(row);
  }
  _database._lg_layout->set_row_2d_list(row_2d_list);

  // set row id
  int32_t row_cnt = 0;
  for (auto& row_vec : _database._lg_layout->get_row_2d_list()) {
    for (auto* row : row_vec) {
      row->set_index(row_cnt);
      row_cnt++;
    }
  }
}

void Legalizer::wrapRegionList()
{
  Design* pl_design = _database._placer_db->get_design();

  for (auto* pl_region : pl_design->get_region_list()) {
    LGRegion* region = new LGRegion(pl_region->get_name());
    if (pl_region->isFence()) {
      region->set_type(LGREGION_TYPE::kFence);
    }
    if (pl_region->isGuide()) {
      region->set_type(LGREGION_TYPE::kGuide);
    }
    for (auto boundary : pl_region->get_boundaries()) {
      int32_t llx = boundary.get_ll_x() + _database._shift_x;
      int32_t lly = boundary.get_ll_y() + _database._shift_y;
      int32_t urx = boundary.get_ur_x() + _database._shift_x;
      int32_t ury = boundary.get_ur_y() + _database._shift_y;
      region->add_shape(Rectangle<int32_t>(llx, lly, urx, ury));
    }
    _database._lg_layout->add_region(region);
  }
}

void Legalizer::wrapCellList()
{
  const Layout* pl_layout = _database._placer_db->get_layout();

  for (auto* pl_cell : pl_layout->get_cell_list()) {
    LGCell* cell = new LGCell(pl_cell->get_name());
    if (pl_cell->isMacro()) {
      cell->set_type(LGCELL_TYPE::kMacro);
    }
    if (pl_cell->isClockBuffer() || pl_cell->isFlipflop()) {
      cell->set_type(LGCELL_TYPE::kSequence);
    }
    if (pl_cell->isLogic() || pl_cell->isPhysicalFiller()) {
      cell->set_type(LGCELL_TYPE::kStdcell);
    }
    cell->set_width(pl_cell->get_width());
    cell->set_height(pl_cell->get_height());
    _database._lg_layout->add_cell(cell);
  }
}

void Legalizer::updateInstanceList()
{
  auto* pl_design = _database._placer_db->get_design();
  updateInstanceList(pl_design->get_instance_list());
}

void Legalizer::updateInstanceList(std::vector<Instance*> inst_list)
{
  checkMapping();

  int32_t changed_cnt = 0;
  for (auto* pl_inst : inst_list) {
    if (updateInstance(pl_inst)) {
      changed_cnt++;
    }
  }

  // set inst index
  int32_t inst_cnt = 0;
  for (auto* inst : _database.get_lgInstance_list()) {
    inst->set_index(inst_cnt);
    inst_cnt++;
  }

  // when changed_cnt reach a threshold, LG turn the default mode
  int32_t changed_threshold = _database._lgInstance_list.size() * 0.1;
  if (changed_cnt < changed_threshold) {
    _mode = LG_MODE::kIncremental;
  } else {
    _mode = LG_MODE::kComplete;
  }
}

bool Legalizer::updateInstance(Instance* pl_inst)
{
  auto* lg_inst = findLGInstance(pl_inst);
  if (!lg_inst) {
    lg_inst = new LGInstance(pl_inst->get_name());
    updateInstanceInfo(pl_inst, lg_inst);
    updateInstanceMapping(pl_inst, lg_inst);
    _target_inst_list.push_back(lg_inst);
    return true;
  } else {
    if (checkInstChanged(pl_inst, lg_inst)) {
      updateInstanceInfo(pl_inst, lg_inst);
      _target_inst_list.push_back(lg_inst);
      return true;
    }
  }

  return false;
}

bool Legalizer::updateInstance(std::string pl_inst_name)
{
  Design* design = _database._placer_db->get_design();
  auto* pl_inst = design->find_instance(pl_inst_name);
  return updateInstance(pl_inst);
}

bool Legalizer::checkMapping()
{
  bool flag = true;
  if (_database._instance_map.size() != _database._lgInstance_map.size()) {
    LOG_WARNING << "LG Instance Mapping is not equal!";
    flag = false;
  }
  return flag;
}

LGInstance* Legalizer::findLGInstance(Instance* pl_inst)
{
  LGInstance* lg_inst = nullptr;
  auto iter = _database._instance_map.find(pl_inst);
  if (iter != _database._instance_map.end()) {
    lg_inst = iter->second;
  }

  return lg_inst;
}

bool Legalizer::checkInstChanged(Instance* pl_inst, LGInstance* lg_inst)
{
  bool flag = false;

  // coordinate
  int32_t pl_inst_x = pl_inst->get_coordi().get_x();
  int32_t pl_inst_y = pl_inst->get_coordi().get_y();
  int32_t lg_inst_origin_x = lg_inst->get_coordi().get_x() - _database._shift_x;
  int32_t lg_inst_origin_y = lg_inst->get_coordi().get_y() - _database._shift_y;

  if ((pl_inst_x != lg_inst_origin_x) || (pl_inst_y != lg_inst_origin_y)) {
    flag = true;
  }

  return flag;
}

void Legalizer::updateInstanceInfo(Instance* pl_inst, LGInstance* lg_inst)
{
  LGCell* inst_master = nullptr;
  if (pl_inst->get_cell_master()) {
    inst_master = _database._lg_layout->find_cell(pl_inst->get_cell_master()->get_name());
  }
  lg_inst->set_master(inst_master);

  int32_t movement_weight = pl_inst->get_pins().size();
  if (movement_weight > 0) {
    // if(inst_master){
    //   if(inst_master->get_type() == LGCELL_TYPE::kSequence){
    //   movement_weight *= 2;
    //   }
    // }
    lg_inst->set_weight(movement_weight);
  }

  // set lg_inst shape with shift x/y and right padding
  int32_t site_width = _database._lg_layout->get_site_width();
  int32_t inst_lx = pl_inst->get_shape().get_ll_x() + _database._shift_x;
  int32_t inst_ly = pl_inst->get_shape().get_ll_y() + _database._shift_y;
  int32_t inst_ux = pl_inst->get_shape().get_ur_x() + _database._shift_x + _config.get_global_padding() * site_width;
  int32_t inst_uy = pl_inst->get_shape().get_ur_y() + _database._shift_y;
  lg_inst->set_shape(Rectangle<int32_t>(inst_lx, inst_ly, inst_ux, inst_uy));

  lg_inst->set_orient(pl_inst->get_orient());

  // set lg_inst state
  if (pl_inst->isUnPlaced()) {
    lg_inst->set_state(LGINSTANCE_STATE::kUnPlaced);
  } else if (pl_inst->isPlaced()) {
    lg_inst->set_state(LGINSTANCE_STATE::kPlaced);
  } else if (pl_inst->isFixed()) {
    lg_inst->set_state(LGINSTANCE_STATE::kFixed);
  }

  //   // set lg_inst region
  //   auto* pl_inst_region = pl_inst->get_belong_region();
  //   if (pl_inst_region) {
  //     auto* lg_inst_region = _database._lg_layout->find_region(pl_inst_region->get_name());
  //     if (lg_inst_region) {
  //       lg_inst->set_belong_region(lg_inst_region);
  //       lg_inst_region->add_inst(lg_inst);
  //     } else {
  //       LOG_WARNING << "Region : " << pl_inst_region->get_name() << " has not been initialized!";
  //     }
  //   }
}

void Legalizer::updateInstanceMapping(Instance* pl_inst, LGInstance* lg_inst)
{
  _database._lgInstance_list.push_back(lg_inst);
  _database._instance_map.emplace(pl_inst, lg_inst);
  _database._lgInstance_map.emplace(lg_inst, pl_inst);
}

void Legalizer::initSegmentList()
{
  auto* lg_layout = _database._lg_layout;
  Utility utility;

  int32_t core_width = lg_layout->get_max_x();
  int32_t core_height = lg_layout->get_max_y();
  int32_t row_height = lg_layout->get_row_height();
  int32_t site_width = lg_layout->get_site_width();
  int32_t site_count_x = static_cast<double>(core_width) / site_width;
  int32_t site_count_y = lg_layout->get_row_num();

  enum SiteInfo
  {
    kEmpty,
    kOccupied
  };
  std::vector<SiteInfo> site_grid(site_count_x * site_count_y, SiteInfo::kOccupied);

  // Deal with fragmented row case and add left global padding
  for (int32_t i = 0; i < lg_layout->get_row_num(); i++) {
    for (auto* row : lg_layout->get_row_2d_list()[i]) {
      int32_t row_min_x = row->get_coordinate().get_x();
      int32_t row_max_x = row_min_x + row->get_site_num() * site_width;
      int32_t row_min_y = row->get_coordinate().get_y();
      int32_t row_max_y = row_min_y + row_height;
      std::pair<int32_t, int32_t> pair_x = utility.obtainMinMaxIdx(0, site_width, row_min_x, row_max_x);
      std::pair<int32_t, int32_t> pair_y = utility.obtainMinMaxIdx(0, row_height, row_min_y, row_max_y);

      // In order to ensure the left padding of instances
      pair_x.first = pair_x.first + _config.get_global_padding();

      for (int32_t j = pair_x.first; j < pair_x.second; j++) {
        for (int32_t k = pair_y.first; k < pair_y.second; k++) {
          LOG_FATAL_IF((k * site_count_x + j) >= static_cast<int32_t>(site_grid.size()))
              << "Row : " << row->get_name() << " is out of core boundary.";
          site_grid[k * site_count_x + j] = kEmpty;
        }
      }
    }
  }

  // Deal with fence region
  for (auto* region : lg_layout->get_region_list()) {
    if (region->get_type() == LGREGION_TYPE::kFence) {
      for (auto rect : region->get_shape_list()) {
        std::pair<int32_t, int32_t> pair_x = utility.obtainMinMaxIdx(0, site_width, rect.get_ll_x(), rect.get_ur_x());
        std::pair<int32_t, int32_t> pair_y = utility.obtainMinMaxIdx(0, row_height, rect.get_ll_y(), rect.get_ur_y());

        // In order to ensure the left padding of instances
        pair_x.second = pair_x.second + _config.get_global_padding();

        for (int32_t i = pair_x.first; i < pair_x.second; i++) {
          for (int32_t j = pair_y.first; j < pair_y.second; j++) {
            LOG_FATAL_IF((j * site_count_x + i) >= static_cast<int32_t>(site_grid.size()))
                << "Region : " << region->get_name() << " is out of core boundary.";
            site_grid[j * site_count_x + i] = kOccupied;
          }
        }
      }
    }
  }

  // Deal with fixed instances
  for (auto* inst : _database._lgInstance_list) {
    if (inst->get_state() == LGINSTANCE_STATE::kFixed) {
      int32_t rect_llx = (inst->get_coordi().get_x() > 0 ? inst->get_coordi().get_x() : 0);
      int32_t rect_lly = (inst->get_coordi().get_y() > 0 ? inst->get_coordi().get_y() : 0);
      int32_t rect_urx = (inst->get_shape().get_ur_x() < core_width ? inst->get_shape().get_ur_x() : core_width);
      int32_t rect_ury = (inst->get_shape().get_ur_y() < core_height ? inst->get_shape().get_ur_y() : core_height);
      if ((rect_llx > core_width) || (rect_lly > core_height) || (rect_urx < 0) || (rect_ury < 0)) {
        continue;
      }

      std::pair<int32_t, int32_t> pair_x = utility.obtainMinMaxIdx(0, site_width, rect_llx, rect_urx);
      std::pair<int32_t, int32_t> pair_y = utility.obtainMinMaxIdx(0, row_height, rect_lly, rect_ury);

      for (int32_t i = pair_x.first; i < pair_x.second; i++) {
        for (int32_t j = pair_y.first; j < pair_y.second; j++) {
          site_grid[j * site_count_x + i] = kOccupied;
        }
      }
    }
  }

  // Add LGLayout segment_2d_list
  std::vector<std::vector<LGInterval*>> segment_2d_list;
  segment_2d_list.resize(_database._lg_layout->get_row_num());
  for (int32_t j = 0; j < site_count_y; j++) {
    int32_t segment_cnt = 0;
    for (int32_t i = 0; i < site_count_x; i++) {
      if (site_grid[j * site_count_x + i] == kEmpty) {
        int32_t start_x = i;
        while (i < site_count_x && site_grid[j * site_count_x + i] == kEmpty) {
          ++i;
        }
        int32_t end_x = i;

        int32_t min_x = site_width * start_x;
        int32_t max_x = site_width * end_x;
        LGInterval* segment = new LGInterval(std::to_string(j) + "_" + std::to_string(segment_cnt++), min_x, max_x);

        // search belong row
        for (auto* row : lg_layout->get_row_2d_list()[j]) {
          int32_t row_lx = row->get_coordinate().get_x();
          int32_t row_ux = row_lx + row->get_site_num() * site_width;

          if (min_x >= row_lx && max_x <= row_ux) {
            segment->set_belong_row(row);
          }
        }
        segment_2d_list[j].push_back(segment);
      }
    }
  }
  _database._lg_layout->set_interval_2d_list(segment_2d_list);

  // set interval index
  int32_t interval_cnt = 0;
  for (auto& interval_vec : _database._lg_layout->get_interval_2d_list()) {
    for (auto* interval : interval_vec) {
      interval->set_index(interval_cnt);
      interval_cnt++;
    }
  }
}

bool Legalizer::runLegalize()
{
  LOG_INFO << "-----------------Start Legalization-----------------";
  ieda::Stats lg_status;

  bool is_succeed = true;
  _method->initDataRequirement(&_config, &_database);
  is_succeed = _method->runLegalization();
  if (is_succeed) {
    alignInstanceOrient();
    LOG_INFO << "Total Movement: " << calTotalMovement();

    this->notifyPLMovementInfo();

    writebackPlacerDB();
    _target_inst_list.clear();
  }

  PlacerDBInst.updateTopoManager();
  PlacerDBInst.updateGridManager();

  double time_delta = lg_status.elapsedRunTime();
  LOG_INFO << "Legalization Total Time Elapsed: " << time_delta << "s";
  LOG_INFO << "-----------------Finish Legalization-----------------";

  return is_succeed;
}

bool Legalizer::runIncrLegalize()
{
  LOG_INFO << "-----------------Start Incrmental Legalization-----------------";
  ieda::Stats incr_lg_status;

  _mode = LG_MODE::kIncremental; // tmp for incremental placement

  bool is_succeed = true;
  if (!_method->isInitialized()) {
    LOG_WARNING << "Clusters have not been initialized! Turn to Complete Legalization.";
    _method->initDataRequirement(&_config, &_database);
    is_succeed = _method->runLegalization();
  } else {
    if (_mode == LG_MODE::kComplete) {
      LOG_WARNING << "Too many instances changed, start legalization of complete mode";

      is_succeed = _method->runLegalization();
    } else if (_mode == LG_MODE::kIncremental) {
      _method->specifyTargetInstList(_target_inst_list);
      is_succeed = _method->runIncrLegalization();
    }
  }
  if (is_succeed) {
    alignInstanceOrient();
    LOG_INFO << "Total Movement: " << calTotalMovement();
    writebackPlacerDB();
    _target_inst_list.clear();
  }

  PlacerDBInst.updateTopoManager();
  // PlacerDBInst.updateGridManager();

  double time_delta = incr_lg_status.elapsedRunTime();
  LOG_INFO << "Incrmental Legalization Total Time Elapsed: " << time_delta << "s";
  LOG_INFO << "-----------------Finish Incrmental Legalization-----------------";

  return is_succeed;
}

bool Legalizer::runRollback(bool clear_but_not_rollback){
  bool is_succeed = _method->runRollback(clear_but_not_rollback);
  if(is_succeed){
    alignInstanceOrient();
    writebackPlacerDB();
  }
  PlacerDBInst.updateTopoManager();

  return is_succeed;
}

void Legalizer::alignInstanceOrient()
{
  int32_t row_height = _database.get_lg_layout()->get_row_height();
  for (auto* inst : _database._lgInstance_list) {
    if (inst->get_state() == LGINSTANCE_STATE::kFixed) {
      continue;
    }

    int32_t row_idx = inst->get_coordi().get_y() / row_height;
    auto* inst_row = _database._lg_layout->get_row_2d_list()[row_idx][0];
    inst->set_orient(inst_row->get_row_orient());
    inst->set_state(LGINSTANCE_STATE::kPlaced);
  }
}

int64_t Legalizer::calTotalMovement()
{
  int64_t sum_movement = 0;
  for (auto pair : _database._instance_map) {
    sum_movement += std::abs(pair.first->get_coordi().get_x() - (pair.second->get_coordi().get_x() - _database._shift_x));
    sum_movement += std::abs(pair.first->get_coordi().get_y() - (pair.second->get_coordi().get_y() - _database._shift_y));
  }
  return sum_movement;
}

int64_t Legalizer::calMaxMovement()
{
  int64_t max_movement = 0;
  for(auto pair : _database._instance_map){
    int64_t cur_movement = std::abs(pair.first->get_coordi().get_x() - (pair.second->get_coordi().get_x() - _database._shift_x))
                    + std::abs(pair.first->get_coordi().get_y() - (pair.second->get_coordi().get_y() - _database._shift_y));
    cur_movement > max_movement ? max_movement = cur_movement : cur_movement;
  }
  return max_movement;
}

void Legalizer::notifyPLMovementInfo()
{
  PlacerDBInst.lg_total_movement = calTotalMovement();
  PlacerDBInst.lg_max_movement = calMaxMovement();
}

void Legalizer::writebackPlacerDB()
{
  for (auto* lg_inst : _database._lgInstance_list) {
    if (lg_inst->get_state() == LGINSTANCE_STATE::kFixed) {
      continue;
    }

    int32_t inst_lx = lg_inst->get_coordi().get_x() - _database._shift_x;
    int32_t inst_ly = lg_inst->get_coordi().get_y() - _database._shift_y;

    auto it = _database._lgInstance_map.find(lg_inst);
    if (it != _database._lgInstance_map.end()) {
      auto* pl_inst = it->second;
      pl_inst->set_orient(lg_inst->get_orient());
      pl_inst->update_coordi(inst_lx, inst_ly);
    }
  }
}

Legalizer& Legalizer::getInst()
{
  if (!_s_lg_instance) {
    _s_lg_instance = new Legalizer();
  }
  return *_s_lg_instance;
}

void Legalizer::destoryInst()
{
  if (_s_lg_instance) {
    delete _s_lg_instance;
    _s_lg_instance = nullptr;
  }
}

Legalizer::~Legalizer()
{
  delete _method;
}

// private
Legalizer* Legalizer::_s_lg_instance = nullptr;

}  // namespace ipl