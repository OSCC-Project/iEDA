/*
 * @FilePath: init_idb.cpp
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-09-02 15:47:09
 * @Description:
 */
#include "init_idb.h"

#include "idm.h"

namespace ieval {

InitIDB* InitIDB::_init_idb = nullptr;

InitIDB::InitIDB()
{
}

InitIDB::~InitIDB()
{
}

InitIDB* InitIDB::getInst()
{
  if (_init_idb == nullptr) {
    _init_idb = new InitIDB();
  }
  return _init_idb;
}

void InitIDB::destroyInst()
{
  if (_init_idb != nullptr) {
    delete _init_idb;
    _init_idb = nullptr;
  }
}

void InitIDB::initPointSets()
{
  auto* idb_builder = dmInst->get_idb_builder();
  idb::IdbDesign* idb_design = idb_builder->get_def_service()->get_design();

  for (auto* idb_net : idb_design->get_net_list()->get_net_list()) {
    std::vector<std::pair<int32_t, int32_t>> point_set;
    std::string net_name = idb_net->get_net_name();

    auto* idb_driving_pin = idb_net->get_driving_pin();
    if (idb_driving_pin) {
      int32_t x = idb_driving_pin->get_average_coordinate()->get_x();
      int32_t y = idb_driving_pin->get_average_coordinate()->get_y();
      point_set.emplace_back(std::make_pair(x, y));
    }

    for (auto* idb_load_pin : idb_net->get_load_pins()) {
      int32_t x = idb_load_pin->get_average_coordinate()->get_x();
      int32_t y = idb_load_pin->get_average_coordinate()->get_y();
      point_set.emplace_back(std::make_pair(x, y));
    }

    _name_point_set[net_name] = point_set;
    _point_sets.emplace_back(point_set);
  }
}

int32_t InitIDB::getDesignUnit()
{
  return dmInst->get_idb_layout()->get_units()->get_micron_dbu();
}

int32_t InitIDB::getRowHeight()
{
  return dmInst->get_idb_layout()->get_rows()->get_row_height();
}

void InitIDB::initCongestionDB()
{
  auto* idb_builder = dmInst->get_idb_builder();
  idb::IdbDesign* idb_design = idb_builder->get_def_service()->get_design();
  idb::IdbLayout* idb_layout = idb_builder->get_def_service()->get_layout();

  // Init CongestionRegion
  idb::IdbRect* die_bbox = idb_layout->get_die()->get_bounding_box();
  _congestion_region.lx = die_bbox->get_low_x();
  _congestion_region.ly = die_bbox->get_low_y();
  _congestion_region.ux = die_bbox->get_high_x();
  _congestion_region.uy = die_bbox->get_high_y();

  // Init CongestionNets
  for (size_t i = 0; i < idb_design->get_net_list()->get_net_list().size(); i++) {
    auto* idb_net = idb_design->get_net_list()->get_net_list()[i];
    std::string net_name = idb_net->get_net_name();
    CongestionNet net;
    net.name = net_name;

    auto* idb_driving_pin = idb_net->get_driving_pin();
    if (idb_driving_pin) {
      CongestionPin pin;
      pin.lx = idb_driving_pin->get_average_coordinate()->get_x();
      pin.ly = idb_driving_pin->get_average_coordinate()->get_y();
      net.pins.emplace_back(pin);
    }
    for (auto* idb_load_pin : idb_net->get_load_pins()) {
      CongestionPin pin;
      pin.lx = idb_load_pin->get_average_coordinate()->get_x();
      pin.ly = idb_load_pin->get_average_coordinate()->get_y();
      net.pins.emplace_back(pin);
    }
    _congestion_nets.emplace_back(net);
  }
}

void InitIDB::initDensityDB()
{
  initDensityDBRegion();
  initDensityDBCells();
  initDensityDBNets();
  _density_db_initialized = true;
}

void InitIDB::initDensityDBRegion()
{
  if (_density_db_initialized) {
    return;
  }
  auto* idb_builder = dmInst->get_idb_builder();
  idb::IdbLayout* idb_layout = idb_builder->get_def_service()->get_layout();

  idb::IdbRect* die_bbox = idb_layout->get_die()->get_bounding_box();
  _density_region.lx = die_bbox->get_low_x();
  _density_region.ly = die_bbox->get_low_y();
  _density_region.ux = die_bbox->get_high_x();
  _density_region.uy = die_bbox->get_high_y();

  idb::IdbRect* core_bbox = idb_layout->get_core()->get_bounding_box();
  _density_region_core.lx = core_bbox->get_low_x();
  _density_region_core.ly = core_bbox->get_low_y();
  _density_region_core.ux = core_bbox->get_high_x();
  _density_region_core.uy = core_bbox->get_high_y();
}

int32_t InitIDB::getDieHeight()
{
  return dmInst->get_idb_builder()->get_def_service()->get_layout()->get_die()->get_bounding_box()->get_height();  
}

int32_t InitIDB::getDieWidth()
{
  return dmInst->get_idb_builder()->get_def_service()->get_layout()->get_die()->get_bounding_box()->get_width();
}

void InitIDB::initDensityDBCells()
{
  if (_density_db_initialized) {
    return;
  }
  auto* idb_builder = dmInst->get_idb_builder();
  idb::IdbDesign* idb_design = idb_builder->get_def_service()->get_design();
  idb::IdbLayout* idb_layout = idb_builder->get_def_service()->get_layout();
  int id = 0;

  idb::IdbRect* core_bbox = idb_layout->get_core()->get_bounding_box();
  int32_t core_lx = core_bbox->get_low_x();
  int32_t core_ly = core_bbox->get_low_y();
  int32_t core_ux = core_bbox->get_high_x();
  int32_t core_uy = core_bbox->get_high_y();

  int32_t row_height = idb_layout->get_rows()->get_row_list()[0]->get_site()->get_height();

  for (auto* idb_inst : idb_design->get_instance_list()->get_instance_list()) {
    auto cell_bbox = idb_inst->get_bounding_box();

    DensityCell cell;
    cell.id = id++;
    cell.lx = cell_bbox->get_low_x();
    cell.ly = cell_bbox->get_low_y();
    cell.width = cell_bbox->get_width();
    cell.height = cell_bbox->get_height();

    bool is_all_in_core = true;
    if (cell.lx < core_lx || cell.ly < core_ly || cell.lx + cell.width > core_ux || cell.ly + cell.height > core_uy) {
      is_all_in_core = false;
    }
    auto inst_status = idb_inst->get_status();
    if (inst_status == IdbPlacementStatus::kFixed && is_all_in_core && cell.height > row_height) {
      cell.type = "macro";
    } else if (inst_status == IdbPlacementStatus::kPlaced && is_all_in_core) {
      cell.type = "stdcell";
    }

    for (auto* inst_pin : idb_inst->get_pin_list()->get_pin_list()) {
      DensityPin pin;
      pin.type = cell.type;
      pin.lx = inst_pin->get_average_coordinate()->get_x();
      pin.ly = inst_pin->get_average_coordinate()->get_y();
      _density_pins.emplace_back(pin);
    }

    _density_cells.emplace_back(cell);
  }
}

void InitIDB::initDensityDBNets()
{
  if (_density_db_initialized) {
    return;
  }
  auto* idb_builder = dmInst->get_idb_builder();
  idb::IdbDesign* idb_design = idb_builder->get_def_service()->get_design();
  int id = 0;

  for (size_t i = 0; i < idb_design->get_net_list()->get_net_list().size(); i++) {
    auto* idb_net = idb_design->get_net_list()->get_net_list()[i];
    DensityNet net;
    int32_t min_lx = INT32_MAX;
    int32_t min_ly = INT32_MAX;
    int32_t max_ux = INT32_MIN;
    int32_t max_uy = INT32_MIN;

    auto* idb_driving_pin = idb_net->get_driving_pin();
    if (idb_driving_pin) {
      min_lx = idb_driving_pin->get_average_coordinate()->get_x();
      min_ly = idb_driving_pin->get_average_coordinate()->get_y();
      max_ux = idb_driving_pin->get_average_coordinate()->get_x();
      max_uy = idb_driving_pin->get_average_coordinate()->get_y();
    }
    for (auto* idb_load_pin : idb_net->get_load_pins()) {
      min_lx = std::min(min_lx, idb_load_pin->get_average_coordinate()->get_x());
      min_ly = std::min(min_ly, idb_load_pin->get_average_coordinate()->get_y());
      max_ux = std::max(max_ux, idb_load_pin->get_average_coordinate()->get_x());
      max_uy = std::max(max_uy, idb_load_pin->get_average_coordinate()->get_y());
    }
    
    // 安全检查：如果net没有任何pin或坐标无效，跳过此net
    if (min_lx == INT32_MAX || min_ly == INT32_MAX || 
        max_ux == INT32_MIN || max_uy == INT32_MIN ||
        min_lx > max_ux || min_ly > max_uy) {
      continue;
    }
    
    net.lx = min_lx;
    net.ly = min_ly;
    net.ux = max_ux;
    net.uy = max_uy;
    net.id = id++;
    _density_nets.emplace_back(net);
  }
}

}  // namespace ieval
