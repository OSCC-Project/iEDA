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

}  // namespace ieval
