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
#include "DetailPlacer.hh"

#include "module/evaluator/density/Density.hh"
#include "module/evaluator/wirelength/HPWirelength.hh"
#ifdef ENABLE_AI
#include "module/evaluator/wirelength/AIWirelength.hh"
#endif
#include "operation/BinOpt.hh"
#include "operation/InstanceSwap.hh"
#include "operation/LocalReorder.hh"
#include "operation/NFSpread.hh"
#include "operation/RowOpt.hh"
#include "usage/usage.hh"
#include "utility/Utility.hh"

namespace ipl {

DetailPlacer::DetailPlacer(Config* pl_config, PlacerDB* placer_db)
{
  initDPConfig(pl_config);
  _config.set_grid_cnt_x(pl_config->get_nes_config().get_bin_cnt_x());
  _config.set_grid_cnt_y(pl_config->get_nes_config().get_bin_cnt_y());

  initDPDatabase(placer_db);
  _operator.initDPOperator(&_database, &_config);

#ifdef ENABLE_AI
  // Initialize AI wirelength evaluator
  _ai_wirelength_evaluator = std::make_unique<AIWirelength>(_operator.get_topo_manager());
  _use_ai_wirelength = false;
#endif
}

DetailPlacer::~DetailPlacer()
{
}

void DetailPlacer::initDPConfig(Config* pl_config)
{
  _config = pl_config->get_dp_config();
}

void DetailPlacer::initDPDatabase(PlacerDB* placer_db)
{
  _database._placer_db = placer_db;
  initDPLayout();
  initDPDesign();
  initIntervalList();
}

void DetailPlacer::initDPLayout()
{
  const Layout* pl_layout = _database._placer_db->get_layout();

  auto core_shape = pl_layout->get_core_shape();
  int32_t row_height = pl_layout->get_row_height();
  int32_t row_num = std::floor(static_cast<double>(core_shape.get_height()) / row_height);

  // shift all element in core to (0,0)
  _database._shift_x = 0 - core_shape.get_ll_x();
  _database._shift_y = 0 - core_shape.get_ll_y();
  _database._layout = new DPLayout(row_num, core_shape.get_ur_x() + _database._shift_x, core_shape.get_ur_y() + _database._shift_y);
  _database._layout->set_dbu(pl_layout->get_database_unit());

  // arrange row to DPLayout _row_2d_list
  wrapRowList();

  // add DPLayout region list
  wrapRegionList();

  // add DPLayout cell list
  wrapCellList();
}

void DetailPlacer::wrapRowList()
{
  const Layout* pl_layout = _database._placer_db->get_layout();

  std::vector<std::vector<DPRow*>> row_2d_list;
  row_2d_list.resize(_database._layout->get_row_num());
  for (auto* pl_row : pl_layout->get_row_list()) {
    auto* pl_site = pl_row->get_site();
    DPSite* row_site = new DPSite(pl_site->get_name());
    row_site->set_width(pl_site->get_site_width());
    row_site->set_height(pl_site->get_site_height());

    int32_t row_shift_x = pl_row->get_coordi().get_x() + _database._shift_x;
    int32_t row_shift_y = pl_row->get_coordi().get_y() + _database._shift_y;
    int32_t row_index = std::floor(static_cast<double>(row_shift_y) / pl_row->get_site_height());
    DPRow* row = new DPRow(pl_row->get_name(), row_site, pl_row->get_site_num());
    row->set_coordinate(row_shift_x, row_shift_y);
    row->set_orient(std::move(pl_site->get_orient()));

    Rectangle<int64_t> rect(row_shift_x, row_shift_y, row_shift_x + pl_row->get_site_num() * row_site->get_width(),
                            row_shift_y + row_site->get_height());
    row->set_bound(rect);

    row_2d_list.at(row_index).push_back(row);
  }
  auto* dp_site = row_2d_list.at(0).at(0)->get_site();
  _database._layout->set_row_height(dp_site->get_height());
  _database._layout->set_site_width(dp_site->get_width());
  _database._layout->set_row_2d_list(row_2d_list);
}

void DetailPlacer::wrapRegionList()
{
  Design* pl_design = _database._placer_db->get_design();

  for (auto* pl_region : pl_design->get_region_list()) {
    DPRegion* region = new DPRegion(pl_region->get_name());
    if (pl_region->isFence()) {
      region->set_type(DPREGION_TYPE::kFence);
    }
    if (pl_region->isGuide()) {
      region->set_type(DPREGION_TYPE::kGuide);
    }
    for (auto boundary : pl_region->get_boundaries()) {
      int32_t llx = boundary.get_ll_x() + _database._shift_x;
      int32_t lly = boundary.get_ll_y() + _database._shift_y;
      int32_t urx = boundary.get_ur_x() + _database._shift_x;
      int32_t ury = boundary.get_ur_y() + _database._shift_y;
      region->add_shape(Rectangle<int32_t>(llx, lly, urx, ury));
    }
    _database._layout->add_region(region);
  }
}

void DetailPlacer::wrapCellList()
{
  const Layout* pl_layout = _database._placer_db->get_layout();

  for (auto* pl_cell : pl_layout->get_cell_list()) {
    DPCell* cell = new DPCell(pl_cell->get_name());
    if (pl_cell->isMacro()) {
      cell->set_type(DPCELL_TYPE::kMacro);
    }
    if (pl_cell->isClockBuffer() || pl_cell->isFlipflop()) {
      cell->set_type(DPCELL_TYPE::kSequence);
    }
    if (pl_cell->isLogic() || pl_cell->isPhysicalFiller()) {
      cell->set_type(DPCELL_TYPE::kStdcell);
    }
    cell->set_width(pl_cell->get_width());
    cell->set_height(pl_cell->get_height());
    _database._layout->add_cell(cell);
  }
}

void DetailPlacer::initDPDesign()
{
  _database._design = new DPDesign();
  wrapInstanceList();
  wrapNetList();
  correctOutsidePinCoordi();
  updateInstanceList();
}

void DetailPlacer::wrapInstanceList()
{
  auto* pl_design = _database._placer_db->get_design();
  auto* dp_design = _database._design;
  for (auto* pl_inst : pl_design->get_instance_list()) {
    DPInstance* dp_inst = wrapInstance(pl_inst);
    dp_design->add_instance(dp_inst);
    dp_design->connectInst(dp_inst, pl_inst);
  }
}

DPInstance* DetailPlacer::wrapInstance(Instance* pl_inst)
{
  DPInstance* dp_inst = new DPInstance(pl_inst->get_name());

  if (pl_inst->get_cell_master()) {
    std::string cell_name = pl_inst->get_cell_master()->get_name();
    auto* dp_cell = _database._layout->find_cell(cell_name);
    dp_inst->set_master(dp_cell);
  }

  // set dp_inst shape with shift x/y and right padding
  int32_t site_width = _database._layout->get_site_width();
  int32_t inst_lx = pl_inst->get_shape().get_ll_x() + _database._shift_x;
  int32_t inst_ly = pl_inst->get_shape().get_ll_y() + _database._shift_y;
  int32_t inst_ux = pl_inst->get_shape().get_ur_x() + _database._shift_x + _config.get_global_padding() * site_width;
  int32_t inst_uy = pl_inst->get_shape().get_ur_y() + _database._shift_y;
  dp_inst->set_shape(Rectangle<int32_t>(inst_lx, inst_ly, inst_ux, inst_uy));

  dp_inst->set_orient(pl_inst->get_orient());

  // set dp_inst state
  if (pl_inst->isUnPlaced()) {
    dp_inst->set_state(DPINSTANCE_STATE::kUnPlaced);
  } else if (pl_inst->isPlaced()) {
    dp_inst->set_state(DPINSTANCE_STATE::kPlaced);
  } else if (pl_inst->isFixed()) {
    dp_inst->set_state(DPINSTANCE_STATE::kFixed);
  }

  // set dp_inst reigon
  auto* pl_inst_region = pl_inst->get_belong_region();
  if (pl_inst_region) {
    auto* dp_inst_region = _database._layout->find_region(pl_inst_region->get_name());
    if (dp_inst_region) {
      dp_inst->set_belong_region(dp_inst_region);
      dp_inst_region->add_inst(dp_inst);
    } else {
      LOG_WARNING << "Region: " << pl_inst_region->get_name() << " has not been initialized!";
    }
  }

  dp_inst->set_weight(pl_inst->get_pins().size());

  return dp_inst;
}

void DetailPlacer::wrapNetList()
{
  auto* pl_design = _database._placer_db->get_design();
  auto* dp_design = _database._design;
  for (auto* pl_net : pl_design->get_net_list()) {
    DPNet* dp_net = wrapNet(pl_net);
    dp_design->add_net(dp_net);
  }
}

DPNet* DetailPlacer::wrapNet(Net* pl_net)
{
  DPNet* dp_net = new DPNet(pl_net->get_name());

  if (pl_net->isClockNet()) {
    dp_net->set_net_type(DPNET_TYPE::kClock);
  } else if (pl_net->isSignalNet()) {
    dp_net->set_net_type(DPNET_TYPE::kSignal);
  }

  if (pl_net->isDontCareNet()) {
    dp_net->set_net_state(DPNET_STATE::kDontCare);
  } else {
    dp_net->set_net_state(DPNET_STATE::kNormal);
  }

  dp_net->set_netweight(pl_net->get_net_weight());

  auto* pl_driver_pin = pl_net->get_driver_pin();
  if (pl_driver_pin) {
    DPPin* driver_pin = wrapPin(pl_driver_pin);
    driver_pin->set_net(dp_net);
    dp_net->set_driver_pin(driver_pin);
    dp_net->add_pin(driver_pin);
    _database._design->add_pin(driver_pin);
  }

  const auto& pl_pin_list = pl_net->get_sink_pins();
  for (size_t i = 0; i < pl_pin_list.size(); i++) {
    DPPin* dp_pin = wrapPin(pl_pin_list[i]);
    dp_pin->set_internal_id(i);
    dp_pin->set_net(dp_net);
    dp_net->add_pin(dp_pin);
    _database._design->add_pin(dp_pin);
  }

  return dp_net;
}

DPPin* DetailPlacer::wrapPin(Pin* pl_pin)
{
  DPPin* dp_pin = new DPPin(pl_pin->get_name());

  const auto& pin_coordi = pl_pin->get_center_coordi();
  const auto& pin_offset_coordi = pl_pin->get_offset_coordi();

  int32_t x_coordi = pin_coordi.get_x() + _database._shift_x;
  int32_t y_coordi = pin_coordi.get_y() + _database._shift_y;

  dp_pin->set_x_coordi(x_coordi);
  dp_pin->set_y_coordi(y_coordi);

  // offset compared to cell master
  dp_pin->set_offset_x(pin_offset_coordi.get_x());
  dp_pin->set_offset_y(pin_offset_coordi.get_y());

  auto* pl_pin_inst = pl_pin->get_instance();
  if (pl_pin_inst) {
    DPInstance* pin_inst = _database._design->find_instance(pl_pin_inst->get_name());
    if (pin_inst) {
      dp_pin->set_instance(pin_inst);
      pin_inst->add_pin(dp_pin);
    } else {
      LOG_WARNING << "Instance: " << pl_pin_inst->get_name() << " has not been initialized!";
    }
  }
  return dp_pin;
}

void DetailPlacer::updateInstanceList()
{
  for (auto* inst : _database.get_design()->get_inst_list()) {
    if (inst->get_state() == DPINSTANCE_STATE::kFixed) {
      continue;
    }
    auto coordi = std::move(inst->get_coordi());
    inst->updateCoordi(coordi.get_x(), coordi.get_y());
  }
}

void DetailPlacer::correctOutsidePinCoordi()
{
  int32_t core_max_x = _database.get_layout()->get_max_x();
  int32_t core_max_y = _database.get_layout()->get_max_y();
  Rectangle<int32_t> core_shape(0, 0, core_max_x, core_max_y);

  for (auto* net : _database.get_design()->get_net_list()) {
    auto bounding_box = std::move(net->obtainBoundingBox());
    if (_operator.checkInNest(bounding_box, core_shape)) {
      continue;
    }

    auto overlap_box = std::move(_operator.obtainOverlapRectangle(bounding_box, core_shape));
    int32_t overlap_wl = overlap_box.get_half_perimeter();
    if (overlap_wl != 0) {
      _database._outside_wl += (bounding_box.get_half_perimeter() - overlap_wl);

      for (auto* pin : net->get_pins()) {
        int32_t pin_x = pin->get_x_coordi();
        int32_t pin_y = pin->get_y_coordi();

        if (pin_x < 0) {
          pin_x = 0;
        }
        if (pin_y < 0) {
          pin_y = 0;
        }
        if (pin_x > core_max_x) {
          pin_x = core_max_x;
        }
        if (pin_y > core_max_y) {
          pin_y = core_max_y;
        }

        pin->set_x_coordi(pin_x);
        pin->set_y_coordi(pin_y);
      }
    }
  }
}

void DetailPlacer::initIntervalList()
{
  auto* layout = _database._layout;
  Utility utility;

  int32_t core_width = layout->get_max_x();
  int32_t core_height = layout->get_max_y();
  int32_t row_height = layout->get_row_height();
  int32_t site_width = layout->get_site_width();
  int32_t site_count_x = static_cast<double>(core_width) / site_width;
  int32_t site_count_y = layout->get_row_num();

  enum SiteInfo
  {
    kEmpty,
    kOccupied
  };
  std::vector<SiteInfo> site_grid(site_count_x * site_count_y, SiteInfo::kOccupied);

  // Deal with fragmented row case and add left global padding
  for (int32_t i = 0; i < layout->get_row_num(); i++) {
    for (auto* row : layout->get_row_2d_list().at(i)) {
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
          site_grid.at(k * site_count_x + j) = kEmpty;
        }
      }
    }
  }

  // Deal with fence region
  for (auto* region : layout->get_region_list()) {
    if (region->get_type() == DPREGION_TYPE::kFence) {
      for (auto rect : region->get_shape_list()) {
        std::pair<int32_t, int32_t> pair_x = utility.obtainMinMaxIdx(0, site_width, rect.get_ll_x(), rect.get_ur_x());
        std::pair<int32_t, int32_t> pair_y = utility.obtainMinMaxIdx(0, row_height, rect.get_ll_y(), rect.get_ur_y());

        // In order to ensure the left padding of instances
        pair_x.second = pair_x.second + _config.get_global_padding();

        for (int32_t i = pair_x.first; i < pair_x.second; i++) {
          for (int32_t j = pair_y.first; j < pair_y.second; j++) {
            LOG_FATAL_IF((j * site_count_x + i) >= static_cast<int32_t>(site_grid.size()))
                << "Region : " << region->get_name() << " is out of core boundary.";
            site_grid.at(j * site_count_x + i) = kOccupied;
          }
        }
      }
    }
  }

  // Deal with fixed instances
  for (auto* inst : _database._design->get_inst_list()) {
    if (inst->get_state() == DPINSTANCE_STATE::kFixed) {
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
          site_grid.at(j * site_count_x + i) = kOccupied;
        }
      }
    }
  }

  // Add DPLayout interval_2d_list
  std::vector<std::vector<DPInterval*>> interval_2d_list;
  interval_2d_list.resize(_database._layout->get_row_num());
  for (int32_t j = 0; j < site_count_y; j++) {
    int32_t interval_cnt = 0;
    for (int32_t i = 0; i < site_count_x; i++) {
      if (site_grid.at(j * site_count_x + i) == kEmpty) {
        int32_t start_x = i;
        while (i < site_count_x && site_grid.at(j * site_count_x + i) == kEmpty) {
          ++i;
        }
        int32_t end_x = i;

        int32_t min_x = site_width * start_x;
        int32_t max_x = site_width * end_x;
        DPInterval* interval = new DPInterval(std::to_string(j) + "_" + std::to_string(interval_cnt++), min_x, max_x);

        // search belong row
        for (auto* row : layout->get_row_2d_list().at(j)) {
          int32_t row_lx = row->get_coordinate().get_x();
          int32_t row_ux = row_lx + row->get_site_num() * site_width;

          if (min_x >= row_lx && max_x <= row_ux) {
            interval->set_belong_row(row);
          }
        }
        interval_2d_list.at(j).push_back(interval);
      }
    }
  }
  _database._layout->set_interval_2d_list(interval_2d_list);
}

bool DetailPlacer::checkIsLegal()
{
  return true;
}

void DetailPlacer::runDetailPlace()
{
  LOG_INFO << "-----------------Start Detail Placement-----------------";
  ieda::Stats dp_status;

  LOG_INFO << "Execution Origin Instance Shift: ";
  RowOpt row_opt(&_config, &_database, &_operator);
  row_opt.runRowOpt();
  _operator.updateTopoManager();
  LOG_INFO << "After RowOpt HPWL: " << calTotalHPWL();
  // _operator.updateGridManager();
  // LOG_INFO << "After Origin Peak Bin Density: " << calPeakBinDensity();

  double threshold = 0.005;

  double improve_ratio = threshold;  // NOLINT
  int64_t front_hpwl = calTotalHPWL();
  int64_t update_hpwl = front_hpwl;  // NOLINT
  int32_t swap_iter = 0;
  do {
    LOG_INFO << "Execution Swap Iteration: " << swap_iter;

    InstanceSwap swap_opt(&_config, &_database, &_operator);
    swap_opt.runGlobalSwap();
    _operator.updateTopoManager();
    LOG_INFO << "---After Global Swap HPWL: " << calTotalHPWL();
    // _operator.updateGridManager();
    // LOG_INFO << "---After Global Swap Peak Density: " << calPeakBinDensity();

    swap_opt.runVerticalSwap();
    _operator.updateTopoManager();
    LOG_INFO << "---After Vertical Swap HPWL: " << calTotalHPWL();
    // _operator.updateGridManager();
    // LOG_INFO << "---After Vertical Swap Peak Density: " << calPeakBinDensity();

    LocalReorder reorder_opt(&_config, &_database, &_operator);
    reorder_opt.runLocalReorder();
    _operator.updateTopoManager();
    LOG_INFO << "---After Local Reorder HPWL: " << calTotalHPWL();
    // _operator.updateGridManager();
    // LOG_INFO << "---After Local Reorder Peak Density: " << calPeakBinDensity();

    update_hpwl = calTotalHPWL();
    improve_ratio = static_cast<double>(front_hpwl - update_hpwl) / front_hpwl;

    // BinOpt bin_opt(&_config, &_database, &_operator);
    // bin_opt.runBinOpt();
    // _operator.updateTopoManager();
    // LOG_INFO << "---After Bin Opt HPWL: " << calTotalHPWL();
    // _operator.updateGridManager();
    // LOG_INFO << "---After Bin Opt Peak Density: " << calPeakBinDensity();

    RowOpt row_opt_test(&_config, &_database, &_operator);
    row_opt_test.runRowOpt();
    _operator.updateTopoManager();
    LOG_INFO << "---After Row Opt HPWL: " << calTotalHPWL();
    // _operator.updateGridManager();
    // LOG_INFO << "After Row Opt Peak Density: " << calPeakBinDensity();

    update_hpwl = calTotalHPWL();
    front_hpwl = update_hpwl;
    ++swap_iter;
  } while (improve_ratio > threshold && swap_iter < 10);

  int32_t shift_iter = 0;
  do {
    LOG_INFO << "Execution Final Instance Shift Iteration: " << shift_iter;

    RowOpt row_opt2(&_config, &_database, &_operator);
    row_opt2.runRowOpt();
    _operator.updateTopoManager();

    update_hpwl = calTotalHPWL();
    improve_ratio = static_cast<double>(front_hpwl - update_hpwl) / front_hpwl;
    front_hpwl = update_hpwl;

    LOG_INFO << "---After RowOpt HPWL: " << update_hpwl;
    // _operator.updateGridManager();
    // LOG_INFO << "Iteration: " << shift_iter << " Row Opt Peak Density: " << calPeakBinDensity();
    ++shift_iter;
  } while (improve_ratio > threshold && shift_iter < 10);

  notifyPLPlaceDensity();

  _database._design->writeBackToPL(_database._shift_x, _database._shift_y);
  _database._placer_db->updateTopoManager();
  _database._placer_db->updateGridManager();

  double time_delta = dp_status.elapsedRunTime();
  LOG_INFO << "Detail Plaement Total Time Elapsed: " << time_delta << "s";
  LOG_INFO << "-----------------Finish Detail Placement-----------------";
}

void DetailPlacer::runDetailPlaceNFS()
{
  LOG_INFO << "-----------------Start Network Flow Cell Spreading-----------------";
  ieda::Stats dp_status;

  NFSpread nfspread_opt(&_config, &_database, &_operator);
  nfspread_opt.runNFSpread();
  _operator.updateTopoManager();

  _database._design->writeBackToPL(_database._shift_x, _database._shift_y);
  _database._placer_db->updateTopoManager();
  _database._placer_db->updateGridManager();

  double time_delta = dp_status.elapsedRunTime();
  LOG_INFO << "Detail Plaement Total Time Elapsed: " << time_delta << "s";
  LOG_INFO << "-----------------Finish Network Flow Cell Spreading-----------------";
}

void DetailPlacer::notifyPLPlaceDensity()
{
  auto* grid_manager = _operator.get_grid_manager();
  PlacerDBInst.place_density[2] = grid_manager->obtainAvgGridDensity();
}

int64_t DetailPlacer::calTotalHPWL()
{
#ifdef ENABLE_AI
  if (_use_ai_wirelength && _ai_wirelength_evaluator && _ai_wirelength_evaluator->isModelLoaded()) {
    LOG_INFO << "Calculate Total Wirelength using AI model.";
    return calTotalAIWirelength() + _database._outside_wl;
  } else {
#endif
    HPWirelength hpwl_eval(_operator.get_topo_manager());
    return hpwl_eval.obtainTotalWirelength() + _database._outside_wl;
#ifdef ENABLE_AI
  }
#endif
}

#ifdef ENABLE_AI
bool DetailPlacer::loadAIWirelengthModel(const std::string& model_path)
{
  if (_ai_wirelength_evaluator) {
    bool success = _ai_wirelength_evaluator->loadModel(model_path);
    if (success) {
      LOG_INFO << "Successfully loaded AI wirelength model: " << model_path;
    } else {
      LOG_ERROR << "Failed to load AI wirelength model: " << model_path;
    }
    return success;
  }
  return false;
}

bool DetailPlacer::loadAIWirelengthNormalizationParams(const std::string& params_path)
{
  if (_ai_wirelength_evaluator) {
    bool success = _ai_wirelength_evaluator->loadNormalizationParams(params_path);
    if (success) {
      LOG_INFO << "Successfully loaded AI wirelength normalization parameters: " << params_path;
    } else {
      LOG_ERROR << "Failed to load AI wirelength normalization parameters: " << params_path;
    }
    return success;
  }
  return false;
}

void DetailPlacer::setUseAIWirelength(bool use_ai)
{
  _use_ai_wirelength = use_ai;
  if (_use_ai_wirelength) {
    if (!_ai_wirelength_evaluator || !_ai_wirelength_evaluator->isModelLoaded()) {
      LOG_WARNING << "AI wirelength model not loaded, falling back to HPWL";
      _use_ai_wirelength = false;
    }
  }
  LOG_INFO << "AI wirelength prediction " << (_use_ai_wirelength ? "enabled" : "disabled");
}

int64_t DetailPlacer::calTotalAIWirelength()
{
  if (_ai_wirelength_evaluator && _ai_wirelength_evaluator->isModelLoaded()) {
    return _ai_wirelength_evaluator->obtainTotalWirelength();
  }
  return 0;
}
#endif

float DetailPlacer::calPeakBinDensity()
{
  Density density_eval(_operator.get_grid_manager());
  return density_eval.obtainPeakBinDensity();
}

void DetailPlacer::clearClusterInfo()
{
  _database.get_design()->clearClusterInfo();
}

void DetailPlacer::alignInstanceOrient()
{
  for (auto* inst : _database.get_design()->get_inst_list()) {
    if (inst->get_state() == DPINSTANCE_STATE::kFixed) {
      continue;
    }

    auto* inst_row = inst->get_belong_cluster()->get_belong_interval()->get_belong_row();
    inst->set_orient(inst_row->get_row_orient());
  }
}

}  // namespace ipl