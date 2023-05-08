#include "AbacusLegalizer.hh"

#include "Log.hh"
#include "usage/usage.hh"
#include "utility/Utility.hh"

namespace ipl {

AbacusLegalizer& AbacusLegalizer::getInst()
{
  if (!_abacus_lg_instance) {
    _abacus_lg_instance = new AbacusLegalizer();
  }
  return *_abacus_lg_instance;
}

void AbacusLegalizer::destoryInst()
{
  if (_abacus_lg_instance) {
    delete _abacus_lg_instance;
  }
}

AbacusLegalizer::~AbacusLegalizer()
{
}

void AbacusLegalizer::initAbacusLegalizer(Config* pl_config, PlacerDB* placer_db)
{
  initLGConfig(pl_config);
  initLGDatabase(placer_db);
  _row_height = _database._lg_layout->get_row_height();
  _site_width = _database._lg_layout->get_site_width();
}

void AbacusLegalizer::initLGConfig(Config* pl_config)
{
  _config = pl_config->get_lg_config();
}

void AbacusLegalizer::initLGDatabase(PlacerDB* placer_db)
{
  _database._placer_db = placer_db;
  initLGLayout();
  updateInstanceList();
  initSegmentList();
}

void AbacusLegalizer::initLGLayout()
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

void AbacusLegalizer::wrapRowList()
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
}

void AbacusLegalizer::wrapRegionList()
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

void AbacusLegalizer::wrapCellList()
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

void AbacusLegalizer::updateInstanceList()
{
  _database.resetClusterInfo();
  auto* pl_design = _database._placer_db->get_design();
  updateInstanceList(pl_design->get_instance_list());
}

void AbacusLegalizer::updateInstanceList(std::vector<Instance*> inst_list)
{
  checkMapping();

  int32_t changed_cnt = 0;
  for (auto* pl_inst : inst_list) {
    auto* lg_inst = findLGInstance(pl_inst);
    if (!lg_inst) {
      lg_inst = new LGInstance(pl_inst->get_name());
      updateInstanceInfo(pl_inst, lg_inst);
      updateInstanceMapping(pl_inst, lg_inst);
      _target_inst_list.push_back(lg_inst);
      changed_cnt++;
    } else {
      if (checkInstChanged(pl_inst, lg_inst)) {
        updateInstanceInfo(pl_inst, lg_inst);
        _target_inst_list.push_back(lg_inst);
        changed_cnt++;
      }
    }
  }

  // when changed_cnt reach a threshold, LG turn the default mode
  int32_t changed_threshold = _database._lgInstance_list.size() * 0.1;
  if (changed_cnt < changed_threshold) {
    _mode = LG_MODE::kIncremental;
  } else {
    _mode = LG_MODE::kComplete;
  }
}

bool AbacusLegalizer::checkMapping()
{
  bool flag = true;
  if (_database._instance_map.size() != _database._lgInstance_map.size()) {
    LOG_WARNING << "LG Instance Mapping is not equal!";
    flag = false;
  }
  return flag;
}

LGInstance* AbacusLegalizer::findLGInstance(Instance* pl_inst)
{
  LGInstance* lg_inst = nullptr;
  auto iter = _database._instance_map.find(pl_inst);
  if (iter != _database._instance_map.end()) {
    lg_inst = iter->second;
  }

  return lg_inst;
}

bool AbacusLegalizer::checkInstChanged(Instance* pl_inst, LGInstance* lg_inst)
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

void AbacusLegalizer::updateInstanceInfo(Instance* pl_inst, LGInstance* lg_inst)
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

  // set lg_inst region
  auto* pl_inst_region = pl_inst->get_belong_region();
  if (pl_inst_region) {
    auto* lg_inst_region = _database._lg_layout->find_region(pl_inst_region->get_name());
    if (lg_inst_region) {
      lg_inst->set_belong_region(lg_inst_region);
      lg_inst_region->add_inst(lg_inst);
    } else {
      LOG_WARNING << "Region : " << pl_inst_region->get_name() << " has not been initialized!";
    }
  }
}

void AbacusLegalizer::updateInstanceMapping(Instance* pl_inst, LGInstance* lg_inst)
{
  _database._lgInstance_list.push_back(lg_inst);
  _database._instance_map.emplace(pl_inst, lg_inst);
  _database._lgInstance_map.emplace(lg_inst, pl_inst);
}

void AbacusLegalizer::initSegmentList()
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
}

bool AbacusLegalizer::runLegalize()
{
  LOG_INFO << "-----------------Start Legalization-----------------";
  ieda::Stats lg_status;

  bool is_succeed = true;
  is_succeed = this->runCompleteMode();
  if (is_succeed) {
    alignInstanceOrient(_database._lgInstance_list);
    LOG_INFO << "Total Movement: " << calTotalMovement();
    writebackPlacerDB(_database._lgInstance_list);
    _target_inst_list.clear();
  }

  PlacerDBInst.updateTopoManager();
  PlacerDBInst.updateGridManager();

  double time_delta = lg_status.elapsedRunTime();
  LOG_INFO << "Legalization Total Time Elapsed: " << time_delta << "s";
  LOG_INFO << "-----------------Finish Legalization-----------------";

  return is_succeed;
}

bool AbacusLegalizer::runIncrLegalize()
{
  LOG_INFO << "-----------------Start Incrmental Legalization-----------------";
  ieda::Stats incr_lg_status;

  bool is_succeed = true;
  if (_mode == LG_MODE::kComplete) {
    LOG_WARNING << "Too many instances changed, start legalization of complete mode";
    is_succeed = this->runCompleteMode();
    if (is_succeed) {
      alignInstanceOrient(_database._lgInstance_list);
      LOG_INFO << "Total Movement: " << calTotalMovement();
      writebackPlacerDB(_database._lgInstance_list);
      _target_inst_list.clear();
    }
  } else if (_mode == LG_MODE::kIncremental) {
    is_succeed = this->runIncrementalMode();
    if (is_succeed) {
      alignInstanceOrient(_target_inst_list);
      LOG_INFO << "Total Movement: " << calTotalMovement();
      writebackPlacerDB(_database._lgInstance_list);
      _target_inst_list.clear();
    }
  }

  PlacerDBInst.updateTopoManager();
  PlacerDBInst.updateGridManager();

  double time_delta = incr_lg_status.elapsedRunTime();
  LOG_INFO << "Incrmental Legalization Total Time Elapsed: " << time_delta << "s";
  LOG_INFO << "-----------------Finish Incrmental Legalization-----------------";

  return is_succeed;
}

bool AbacusLegalizer::runCompleteMode()
{
  // Sort all movable instances
  std::vector<LGInstance*> movable_inst_list;
  pickAndSortMovableInstList(movable_inst_list);

  int32_t inst_id = 0;
  for (auto* inst : movable_inst_list) {
    int32_t best_row = INT32_MAX;
    int32_t best_cost = INT32_MAX;
    for (int32_t row_idx = 0; row_idx < _database._lg_layout->get_row_num(); row_idx++) {
      int32_t cost = placeRow(inst, row_idx, true);

      if (cost < best_cost) {
        best_cost = cost;
        best_row = row_idx;
      }
    }

    if (best_row == INT32_MAX) {
      LOG_ERROR << "Instance: " << inst->get_name() << "Cannot find a row for placement";
      return false;
    }

    placeRow(inst, best_row, false);

    inst_id++;
    if (inst_id % 100000 == 0) {
      LOG_INFO << "Place Instance : " << inst_id;
    }
  }

  return true;
}

bool AbacusLegalizer::runIncrementalMode()
{
  int32_t row_range_num = 5;

  for (auto* inst : _target_inst_list) {
    int32_t row_height = _database._lg_layout->get_row_height();
    int32_t row_idx = inst->get_coordi().get_y() / row_height;
    int32_t max_row_idx
        = (row_idx + row_range_num > _database._lg_layout->get_row_num()) ? _database._lg_layout->get_row_num() : row_idx + row_range_num;
    int32_t min_row_idx = (row_idx - row_range_num < 0) ? 0 : row_idx - row_range_num;

    int32_t best_row = INT32_MAX;
    int32_t best_cost = INT32_MAX;
    for (int32_t row_idx = min_row_idx; row_idx < max_row_idx; row_idx++) {
      int32_t cost = placeRow(inst, row_idx, true);
      if (cost < best_cost) {
        best_cost = cost;
        best_row = row_idx;
      }
    }
    placeRow(inst, best_row, false);
  }

  return true;
}

void AbacusLegalizer::pickAndSortMovableInstList(std::vector<LGInstance*>& movable_inst_list)
{
  for (auto* inst : _database._lgInstance_list) {
    if (inst->get_state() == LGINSTANCE_STATE::kFixed) {
      continue;
    }
    movable_inst_list.push_back(inst);
  }

  std::sort(movable_inst_list.begin(), movable_inst_list.end(),
            [](LGInstance* l_inst, LGInstance* r_inst) { return (l_inst->get_coordi().get_x() < r_inst->get_coordi().get_x()); });
}

int32_t AbacusLegalizer::placeRow(LGInstance* inst, int32_t row_idx, bool is_trial)
{
  Rectangle<int32_t> inst_shape = std::move(inst->get_shape());

  // Determine clusters and their optimal positions x_c(c):
  std::vector<LGInterval*> interval_list = _database._lg_layout->get_interval_2d_list()[row_idx];

  // Select the nearest interval for the instance
  int32_t interval_idx = searchNearestIntervalIndex(interval_list, inst_shape);
  if (interval_idx == INT32_MAX) {
    // LOG_WARNING << "Instance is not overlap with interval!";
    return INT32_MAX;
  }

  if (interval_list[interval_idx]->get_remain_length() < inst_shape.get_width()) {
    // Select the most recent non-full interval
    int32_t origin_idx = interval_idx;
    interval_idx = searchRemainSpaceSegIndex(interval_list, inst_shape, origin_idx);
    if (interval_idx == origin_idx) {
      // LOG_INFO << "Row : " << row_idx << " has no room to place.";
      return INT32_MAX;
    }
  }

  // Arrange inst into interval
  auto* target_interval = interval_list[interval_idx];
  LGCluster target_cluster = std::move(arrangeInstIntoIntervalCluster(inst, target_interval));

  // Calculate cost
  int32_t movement_cost = 0;
  int32_t coordi_x = target_cluster.get_min_x();
  int32_t inst_movement_x = INT32_MAX;
  for (auto* target_inst : target_cluster.get_inst_list()) {
    int32_t origin_x = target_inst->get_coordi().get_x();
    if (target_inst == inst) {
      inst_movement_x = std::abs(coordi_x - origin_x);
    } else {
      movement_cost += std::abs(coordi_x - origin_x);
    }
    coordi_x += target_inst->get_shape().get_width();
  }
  // Add Inst Coordi Movement Cost
  int32_t inst_movement_y = std::abs(row_idx * _row_height - inst_shape.get_ll_y());
  int32_t inst_displacement = inst_movement_x + inst_movement_y;
  movement_cost += inst_displacement;

  // Penalize violations of maximum movement constraints
  if (inst_displacement > _config.get_max_displacement()) {
    movement_cost += _database._lg_layout->get_max_x();
  }

  // Replace cluster
  if (!is_trial) {
    replaceClusterInfo(target_cluster);
    target_interval->updateRemainLength(-(inst->get_shape().get_width()));
  }

  return movement_cost;
}

int32_t AbacusLegalizer::searchNearestIntervalIndex(std::vector<LGInterval*>& segment_list, Rectangle<int32_t>& inst_shape)
{
  if (segment_list.size() == 1) {
    return 0;
  }

  int32_t prev_distance = INT32_MAX;
  int32_t segment_idx = INT32_MAX;
  for (size_t i = 0; i < segment_list.size(); i++) {
    int32_t cur_distance
        = calDistanceWithBox(inst_shape.get_ll_x(), inst_shape.get_ur_x(), segment_list[i]->get_min_x(), segment_list[i]->get_max_x());
    if (cur_distance > prev_distance) {
      segment_idx = i - 1;
      break;
    }
    if (cur_distance == 0) {
      segment_idx = i;
      break;
    }

    prev_distance = cur_distance;
  }

  return segment_idx;
}

int32_t AbacusLegalizer::searchRemainSpaceSegIndex(std::vector<LGInterval*>& segment_list, Rectangle<int32_t>& inst_shape,
                                                   int32_t origin_index)
{
  int32_t segment_idx = origin_index;
  // int32_t max_range = segment_list.size() - 1;
  int32_t max_range = 2;
  int32_t range = 1;
  while (range <= max_range) {
    int32_t r_idx = segment_idx;
    if (r_idx + range < static_cast<int32_t>(segment_list.size())) {
      r_idx += range;
      if (segment_list[r_idx]->get_remain_length() >= inst_shape.get_width()) {
        segment_idx = r_idx;
        break;
      }
    }
    int32_t l_idx = segment_idx;
    if (l_idx - range >= 0) {
      l_idx -= range;
      if (segment_list[l_idx]->get_remain_length() >= inst_shape.get_width()) {
        segment_idx = l_idx;
        break;
      }
    }
    range++;
  }
  return segment_idx;
}

LGCluster AbacusLegalizer::arrangeInstIntoIntervalCluster(LGInstance* inst, LGInterval* interval)
{
  auto inst_shape = std::move(inst->get_shape());
  LGCluster record_cluster;
  auto* cur_cluster = interval->get_cluster_root();
  auto* last_cluster = cur_cluster;
  bool is_collapse = false;

  if (cur_cluster && (inst_shape.get_ur_x() < cur_cluster->get_min_x())) {
    // should insert in the cluster
    record_cluster = *cur_cluster;
    record_cluster.insertInstance(inst);
    legalizeCluster(record_cluster);
    is_collapse = true;
  } else {
    while (cur_cluster) {
      if (checkOverlapWithBox(inst_shape.get_ll_x(), inst_shape.get_ur_x(), cur_cluster->get_min_x(), cur_cluster->get_max_x())) {
        record_cluster = *cur_cluster;
        record_cluster.insertInstance(inst);
        legalizeCluster(record_cluster);
        is_collapse = true;
        break;
      }
      auto* back_cluster = cur_cluster->get_back_cluster();
      if (back_cluster) {
        // tmp fix bug.
        if (inst_shape.get_ll_x() >= cur_cluster->get_max_x() && inst_shape.get_ur_x() <= back_cluster->get_min_x()) {
          record_cluster = *cur_cluster;
          record_cluster.insertInstance(inst);
          legalizeCluster(record_cluster);
          is_collapse = true;
          break;
        }

        last_cluster = back_cluster;
      }
      cur_cluster = back_cluster;
    }
  }

  if (!is_collapse) {
    // Create new cluster
    record_cluster = LGCluster(inst->get_name());
    record_cluster.add_inst(inst);
    record_cluster.updateAbacusInfo(inst);
    record_cluster.set_belong_interval(interval);
    record_cluster.set_front_cluster(last_cluster);
    legalizeCluster(record_cluster);
  }

  return record_cluster;
}

void AbacusLegalizer::arrangeClusterMinXCoordi(LGCluster& cluster)
{
  int32_t cluster_x = (cluster.get_weight_q() / cluster.get_weight_e());
  cluster_x = (cluster_x / _site_width) * _site_width;

  int32_t boundary_min_x = cluster.get_belong_interval()->get_min_x();
  int32_t boundary_max_x = cluster.get_belong_interval()->get_max_x();
  cluster_x < boundary_min_x ? cluster_x = boundary_min_x : cluster_x;
  cluster_x + cluster.get_total_width() > boundary_max_x ? cluster_x = boundary_max_x - cluster.get_total_width() : cluster_x;

  if (cluster_x < boundary_min_x) {
    LOG_WARNING << "Cluster width is out of interval capcity";
  }

  cluster.set_min_x(cluster_x);
}

void AbacusLegalizer::legalizeCluster(LGCluster& cluster)
{
  arrangeClusterMinXCoordi(cluster);
  int32_t cur_min_x, front_max_x, back_min_x;
  cur_min_x = cluster.get_min_x();
  front_max_x = obtainFrontMaxX(cluster);
  while (cur_min_x < front_max_x) {
    LGCluster front_cluster = *(cluster.get_front_cluster());
    front_cluster.appendCluster(cluster);
    front_cluster.set_back_cluster(cluster.get_back_cluster());
    arrangeClusterMinXCoordi(front_cluster);
    cur_min_x = front_cluster.get_min_x();
    front_max_x = obtainFrontMaxX(front_cluster);
    cluster = front_cluster;
  }

  cur_min_x = cluster.get_min_x();
  back_min_x = obtainBackMinX(cluster);
  while (cur_min_x + cluster.get_total_width() > back_min_x) {
    auto* back_cluster = cluster.get_back_cluster();
    cluster.appendCluster(*back_cluster);
    cluster.set_back_cluster(back_cluster->get_back_cluster());
    arrangeClusterMinXCoordi(cluster);
    cur_min_x = cluster.get_min_x();
    back_min_x = obtainBackMinX(cluster);
  }
}

int32_t AbacusLegalizer::obtainFrontMaxX(LGCluster& cluster)
{
  int32_t front_max_x = cluster.get_belong_interval()->get_min_x();
  if (cluster.get_front_cluster()) {
    front_max_x = cluster.get_front_cluster()->get_max_x();
  }
  return front_max_x;
}

int32_t AbacusLegalizer::obtainBackMinX(LGCluster& cluster)
{
  int32_t back_min_x = cluster.get_belong_interval()->get_max_x();
  if (cluster.get_back_cluster()) {
    back_min_x = cluster.get_back_cluster()->get_min_x();
  }
  return back_min_x;
}

void AbacusLegalizer::replaceClusterInfo(LGCluster& modify_cluster)
{
  auto* origin_interval = modify_cluster.get_belong_interval();
  int32_t coordi_y = origin_interval->get_belong_row()->get_coordinate().get_y();

  auto* cluster_ptr = _database.findCluster(modify_cluster.get_name());
  if (!cluster_ptr) {
    LGCluster* new_cluster = new LGCluster(std::move(modify_cluster));
    auto inst_list = new_cluster->get_inst_list();
    if (inst_list.size() > 1 || inst_list.size() == 0) {
      LOG_WARNING << "Cluster Inst is not correctly set";
    }
    inst_list[0]->set_belong_cluster(new_cluster);
    inst_list[0]->updateCoordi(new_cluster->get_min_x(), coordi_y);

    _database.insertCluster(new_cluster->get_name(), new_cluster);

    if (!origin_interval->get_cluster_root()) {
      origin_interval->set_cluster_root(new_cluster);
    }

    if (new_cluster->get_front_cluster()) {
      new_cluster->get_front_cluster()->set_back_cluster(new_cluster);
    }
    return;
  }

  auto& origin_cluster = *(cluster_ptr);
  auto* origin_root = origin_interval->get_cluster_root();

  // may be collapsing with front or back cluster
  LGCluster* front_origin = origin_cluster.get_front_cluster();
  LGCluster* front_modify = modify_cluster.get_front_cluster();
  while (front_origin != front_modify) {
    if (front_origin == origin_root) {
      origin_interval->set_cluster_root(&origin_cluster);
    }

    std::string delete_cluster = front_origin->get_name();
    front_origin = front_origin->get_front_cluster();
    if (front_origin) {
      front_origin->set_back_cluster(&origin_cluster);
    }
    _database.deleteCluster(delete_cluster);
  }
  LGCluster* back_origin = origin_cluster.get_back_cluster();
  LGCluster* back_modify = modify_cluster.get_back_cluster();
  while (back_origin != back_modify) {
    std::string delete_cluster = back_origin->get_name();

    back_origin = back_origin->get_back_cluster();
    if (back_origin) {
      back_origin->set_front_cluster(&origin_cluster);
    }
    _database.deleteCluster(delete_cluster);
  }

  // update all inst info
  origin_cluster = std::move(modify_cluster);
  int32_t coordi_x = origin_cluster.get_min_x();
  for (auto* inst : origin_cluster.get_inst_list()) {
    inst->set_belong_cluster(&origin_cluster);
    inst->updateCoordi(coordi_x, coordi_y);
    coordi_x += inst->get_shape().get_width();
  }
}

int32_t AbacusLegalizer::calDistanceWithBox(int32_t min_x, int32_t max_x, int32_t box_min_x, int32_t box_max_x)
{
  if (max_x >= box_min_x && min_x <= box_max_x) {
    return 0;
  } else if (min_x > box_max_x) {
    return (min_x - box_max_x);
  } else if (max_x < box_min_x) {
    return (box_min_x - max_x);
  } else {
    return INT32_MAX;
  }
}

bool AbacusLegalizer::checkOverlapWithBox(int32_t min_x, int32_t max_x, int32_t box_min_x, int32_t box_max_x)
{
  if (max_x > box_min_x && min_x < box_max_x) {
    return true;
  } else {
    return false;
  }
}

void AbacusLegalizer::alignInstanceOrient(std::vector<LGInstance*> inst_list)
{
  for (auto* inst : inst_list) {
    if (inst->get_state() == LGINSTANCE_STATE::kFixed) {
      continue;
    }

    auto* inst_row = inst->get_belong_cluster()->get_belong_interval()->get_belong_row();
    inst->set_orient(inst_row->get_row_orient());
    inst->set_state(LGINSTANCE_STATE::kPlaced);
  }
}

void AbacusLegalizer::writebackPlacerDB(std::vector<LGInstance*> inst_list)
{
  for (auto* lg_inst : inst_list) {
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

int32_t AbacusLegalizer::calTotalMovement()
{
  int32_t sum_movement = 0;
  for (auto pair : _database._instance_map) {
    sum_movement += std::abs(pair.first->get_coordi().get_x() - (pair.second->get_coordi().get_x() - _database._shift_x));
    sum_movement += std::abs(pair.first->get_coordi().get_y() - (pair.second->get_coordi().get_y() - _database._shift_y));
  }
  return sum_movement;
}

// private
AbacusLegalizer* AbacusLegalizer::_abacus_lg_instance = nullptr;

}  // namespace ipl