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

void InitIDB::initCongestionDB()
{
  auto* idb_builder = dmInst->get_idb_builder();
  idb::IdbDesign* idb_design = idb_builder->get_def_service()->get_design();
  idb::IdbLayout* idb_layout = idb_builder->get_def_service()->get_layout();

  // Init CongestionRegion
  idb::IdbRect* die_bbox = idb_layout->get_die()->get_bounding_box();
  int32_t lx = die_bbox->get_low_x();
  int32_t ly = die_bbox->get_low_y();
  int32_t width = die_bbox->get_width();
  int32_t height = die_bbox->get_height();
  _region.lx = lx;
  _region.ly = ly;
  _region.ux = lx + width;
  _region.uy = ly + height;

  // Init CongestionNets
  for (size_t i = 0; i < idb_design->get_net_list()->get_net_list().size(); i++) {
    auto* idb_net = idb_design->get_net_list()->get_net_list()[i];
    CongestionNet net;

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
    _nets.emplace_back(net);
  }
}

}  // namespace ieval
