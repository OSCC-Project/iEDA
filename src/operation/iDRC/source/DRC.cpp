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
#include "DRC.h"

#include <ctime>

#include "CornerFillSpacingCheck.hpp"
#include "CutEolSpacingCheck.hpp"
#include "CutSpacingCheck.hpp"
#include "DrcConfig.h"
#include "DrcConfigurator.h"
#include "DrcConflictGraph.h"
#include "DrcDesign.h"
#include "DrcIDBWrapper.h"
#include "EOLSpacingCheck.hpp"
#include "EnclosedAreaCheck.h"
#include "EnclosureCheck.hpp"
// #include "IDRWrapper.h"
#include "JogSpacingCheck.hpp"
#include "MinStepCheck.hpp"
#include "MultiPatterning.h"
#include "NotchSpacingCheck.hpp"
#include "RegionQuery.h"
#include "RoutingAreaCheck.h"
#include "RoutingSpacingCheck.h"
#include "RoutingWidthCheck.h"
#include "SpotParser.h"
#include "Tech.h"
#include "idm.h"

namespace idrc {
DRC* DRC::_drc_instance = nullptr;

DRC& DRC::getInst()
{
  if (_drc_instance == nullptr) {
    _drc_instance = new DRC();
  }
  return *_drc_instance;
}

void DRC::destroyInst()
{
  if (_drc_instance != nullptr) {
    delete _drc_instance;
    _drc_instance = nullptr;
  }
}

DRC::DRC()
{
  _config = new DrcConfig();
  _drc_design = new DrcDesign();
  _tech = new Tech();
  _conflict_graph = new DrcConflictGraph();
  _multi_patterning = new MultiPatterning();
  _region_query = new RegionQuery(_tech);
}

// /**
//  * @brief 读取DEF文件的模式下获得短路违规的数目
//  *
//  * @return int 短路违规的数目
//  */
// int DRC::getShortViolationNum()
// {
//   return _routing_sapcing_check->get_short_violation_num();
// }

// /**
//  * @brief 读取DEF文件的模式下获得间距违规的数目
//  *
//  * @return int 间距违规的数目
//  */
// int DRC::getSpacingViolationNum()
// {
//   return _routing_sapcing_check->get_spacing_violation_num();
// }

// /**
//  * @brief 读取DEF文件的模式下获得宽度违规的数目
//  *
//  * @return int 宽度违规的数目
//  */
// int DRC::getWidthViolationNum()
// {
//   return _routing_width_check->get_width_violation_num();
// }

// /**
//  * @brief 读取DEF文件的模式下获得面积违规的数目
//  *
//  * @return int 面积违规的数目
//  */
// int DRC::getAreaViolationNum()
// {
//   return _routing_area_check->get_area_violation_num();
// }

// /**
//  * @brief 读取DEF文件的模式下获得包围面积违规的数目
//  *
//  * @return int 包围面积违规的数目
//  */
// int DRC::getEnclosedAreaViolationNum()
// {
//   return _enclosed_area_check->get_enclosed_area_violation_num();
// }

// /**
//  * @brief 返回存储短路违规信息的Spot列表
//  *
//  * @return std::map<int, std::vector<DrcSpot>>&
//  */
// std::map<int, std::vector<DrcSpot>>& DRC::getShortSpotList()
// {
//   return _routing_sapcing_check->get_routing_layer_to_short_spots_list();
// }

// /**
//  * @brief 返回存储间距违规信息的Spot列表
//  *
//  * @return std::map<int, std::vector<DrcSpot>>&
//  */
// std::map<int, std::vector<DrcSpot>>& DRC::getSpacingSpotList()
// {
//   return _routing_sapcing_check->get_routing_layer_to_spacing_spots_list();
// }

// /**
//  * @brief 返回存储宽度违规信息的Spot列表
//  *
//  * @return std::map<int, std::vector<DrcSpot>>&
//  */
// std::map<int, std::vector<DrcSpot>>& DRC::getWidthSpotList()
// {
//   return _routing_width_check->get_routing_layer_to_spots_map();
// }

// /**
//  * @brief 返回存储面积违规信息的Spot列表
//  *
//  * @return std::map<int, std::vector<DrcSpot>>&
//  */
// std::map<int, std::vector<DrcSpot>>& DRC::getAreaSpotList()
// {
//   return _routing_area_check->get_routing_layer_to_spots_map();
// }

// /**
//  * @brief 返回存储孔洞面积违规信息的Spot列表
//  *
//  * @return std::map<int, std::vector<DrcSpot>>&
//  */
// std::map<int, std::vector<DrcSpot>>& DRC::getEnclosedAreaSpotList()
// {
//   return _enclosed_area_check->get_routing_layer_to_spots_map();
// }

// /**
//  * @brief 通过配置文件的方式初始化Tech数据（工艺设计规则数据）
//  *
//  * @param drc_config_path 配置文件路径
//  */
// void DRC::initTechFromIDB(std::string& drc_config_path)
// {
//   DrcConfigurator* configurator = new DrcConfigurator();
//   configurator->set(_config, drc_config_path);
//   delete configurator;

//   _idb_wrapper = new DrcIDBWrapper(_config, _tech, _drc_design, _region_query);
//   _idb_wrapper->initTech();  //传入IdbBuilder？
// }

/**
 * @brief 通过iDB_Builder的指针的方式初始化Tech数据（工艺规则数据）
 *
 * @param idb_builder idb_builder指针
 */
void DRC::initTechFromIDB(idb::IdbBuilder* idb_builder)
{
  delete _idb_wrapper;
  _idb_wrapper = new DrcIDBWrapper(_config, _tech, _drc_design, _region_query);
  _idb_wrapper->initTech(idb_builder);  //传入IdbBuilder？
}

/**
 * @brief 通过配置文件的方式初始化iDRC，包括Tech数据，DrcDesign数据，RegionQuery模块数据
 *
 * @param drc_config_path 配置文件路径
 */
void DRC::initDRC(std::string& drc_config_path, idb::IdbBuilder* idb_builder)
{
  DrcConfigurator* configurator = new DrcConfigurator();
  configurator->set(_config, drc_config_path);
  delete configurator;

  _idb_wrapper = new DrcIDBWrapper(_config, _tech, _drc_design, _region_query);
  _idb_wrapper->input(idb_builder);  //传入IdbBuilder？
}

//通过Data Manager获取数据
void DRC::initDRC()
{
  // _tech = new Tech();
  // _drc_design = new DrcDesign();
  // _region_query = new RegionQuery();
  if (dmInst->get_idb_builder()) {
    _idb_wrapper = new DrcIDBWrapper(_tech, _drc_design, dmInst->get_idb_builder(), _region_query);
    _idb_wrapper->wrapTech();
    _idb_wrapper->wrapDesign();
  } else {
    std::cout << "Error: idb builder is null" << std::endl;
    exit(1);
  }
}

void DRC::initDesign(std::map<std::string, std::any> config_map)
{
  std::string def_path = std::any_cast<std::string>(config_map.find("-def_path")->second);
  dmInst->readDef(def_path);
  if (dmInst->get_idb_builder()) {
    _drc_design = new DrcDesign();
    _region_query = new RegionQuery();
    _idb_wrapper = new DrcIDBWrapper(_tech, _drc_design, dmInst->get_idb_builder(), _region_query);
    _idb_wrapper->wrapDesign();
  } else {
    std::cout << "Error: idb builder is null" << std::endl;
    exit(1);
  }
}

/**
 * @brief 初始化各个设计规则检查模块，为进行设计规则检查做准备
 *
 */
void DRC::initCheckModule()
{
  _jog_spacing_check = new JogSpacingCheck(_tech, _region_query);
  _notch_spacing_check = new NotchSpacingCheck(_tech, _region_query);
  _min_step_check = new MinStepCheck(_tech, _region_query);
  _corner_fill_spacing_check = new CornerFillSpacingCheck(_tech, _region_query);
  _cut_eol_spacing_check = new CutEolSpacingCheck(_tech, _region_query);
  _routing_sapcing_check = new RoutingSpacingCheck(_tech, _region_query);
  _eol_spacing_check = new EOLSpacingCheck(_tech, _region_query);
  _routing_area_check = new RoutingAreaCheck(_tech, _region_query);
  _routing_width_check = new RoutingWidthCheck(_tech, _region_query);
  _enclosed_area_check = new EnclosedAreaCheck(_tech, _region_query);
  _cut_spacing_check = new CutSpacingCheck(_tech, _region_query);
  _enclosure_check = new EnclosureCheck(_tech, _region_query);
  _spot_parser = SpotParser::getInstance(_config, _tech);
}

/**
 * @brief 更新当前各个设计规则检查模块的过程数据与存储结果，以备下一轮设计规则检查
 *
 */
void DRC::update()
{
  _region_query->clear_layer_to_routing_rects_tree_map();
  clearRoutingShapesInDrcNetList();
  _routing_sapcing_check->reset();
  _routing_area_check->reset();
  _routing_width_check->reset();
  _enclosed_area_check->reset();
}

/**
 * @brief 遍历每一条Net运行各个设计规则检查模块对每一条Net进行设计规则检查
 *
 */
void DRC::run()
{
  int index = 0;
  for (auto& drc_net : _drc_design->get_drc_net_list()) {
    if (index++ % 1000 == 0) {
      std::cout << "-" << std::flush;
    }
    if (index++ % 100000 == 0) {
      std::cout << std::endl;
    }
    _routing_sapcing_check->checkRoutingSpacing(drc_net);

    _routing_width_check->checkRoutingWidth(drc_net);

    _routing_area_check->checkArea(drc_net);

    _enclosed_area_check->checkEnclosedArea(drc_net);

    _cut_spacing_check->checkCutSpacing(drc_net);

    _enclosure_check->checkEnclosure(drc_net);

    _eol_spacing_check->checkEOLSpacing(drc_net);

    _notch_spacing_check->checkNotchSpacing(drc_net);

    _min_step_check->checkMinStep(drc_net);

    _corner_fill_spacing_check->checkCornerFillSpacing(drc_net);

    _cut_eol_spacing_check->checkCutEolSpacing(drc_net);
    // cout << "CutEol" << timeEnd - timeStart << std::endl;
    _jog_spacing_check->checkJogSpacing(drc_net);
  }
  // if (_conflict_graph != nullptr) {
  // }
}

// /**
//  * @brief 检查目标Net是否存在设计违规（需要重新实现）
//  *
//  * @param netId
//  */
// void DRC::checkTargetNet(int netId)
// {
//   // DrcNet* targrt_net = _idr_wrapper->get_drc_net(netId);
//   // _routing_sapcing_check->checkRoutingSpacing(targrt_net);
//   // _routing_width_check->checkRoutingWidth(targrt_net);
//   // _routing_area_check->checkRoutingArea(targrt_net);
//   // _enclosed_area_check->checkEnclosedArea(targrt_net);
// }

/**
 * @brief 以文件的形式报告每个模块的设计规则检查结果
 *
 */
void DRC::report()
{
  // _spot_parser->reportSpacingViolation(_routing_sapcing_check);
  // _spot_parser->reportShortViolation(_routing_sapcing_check);
  // _spot_parser->reportAreaViolation(_routing_area_check);
  // _spot_parser->reportWidthViolation(_routing_width_check);
  // _spot_parser->reportEnclosedAreaViolation(_enclosed_area_check);
  // _spot_parser->reportCutSpacingViolation(_cut_spacing_check);
  // _spot_parser->reportEnclosureViolation(_enclosure_check);
  // _spot_parser->reportEOLSpacingViolation(_eol_spacing_check);
  // _spot_parser->reportEnd2EndSpacingViolation(_eol_spacing_check);
  std::map<std::string, int> viotype_to_nums_map;
  _region_query->getRegionReport(viotype_to_nums_map);
  for (auto& [name, nums] : viotype_to_nums_map) {
    std::cout << name << "          " << nums << std::endl;
  }
}

std::map<std::string, int> DRC::getDrcResult()
{
  std::map<std::string, int> viotype_to_nums_map;
  _region_query->getRegionReport(viotype_to_nums_map);
  return viotype_to_nums_map;
}

std::map<std::string, std::vector<DrcViolationSpot*>> DRC::getDrcDetailResult()
{
  std::map<std::string, std::vector<DrcViolationSpot*>> vio_map;
  _region_query->getRegionDetailReport(vio_map);
  return vio_map;
}

/**
 * @brief 与iRT的交互接口
 *
 * @param layer_to_rects_rtree_map iRT的区域绕线结果，包含多层的导体图形信息
 * @return std::vector<std::pair<DrcRect*, DrcRect*>> 存在间距违规或短路违规的矩形对
 */
// std::vector<std::pair<DrcRect*, DrcRect*>> DRC::checkiRTResult(const LayerNameToRTreeMap& layer_to_rects_rtree_map)
// {
//   update();
//   _routing_sapcing_check->switchToiRTMode();
//   _routing_sapcing_check->checkRoutingSpacing(layer_to_rects_rtree_map);
//   return _routing_sapcing_check->get_violation_rect_pair_list();
// }

////////////////////////////////////////
////////////////////////////////////////
/////////////////////////private function
void DRC::clearRoutingShapesInDrcNetList()
{
  for (auto& drc_net : _drc_design->get_drc_net_list()) {
    drc_net->clear_layer_to_routing_rects_map();
    drc_net->clear_layer_to_merge_polygon_list();
    drc_net->clear_layer_to_routing_polygon_set();
  }
}

// void DRC::checkMultipatterning(int check_colorable_num)
// {
//   _multi_patterning->set_conflict_graph(_conflict_graph);
//   _multi_patterning->checkMultiPatterning(check_colorable_num);
// }

// ////////////////////////////////////////////
// //////debug
// void DRC::printRTree()
// {
//   _region_query->printRoutingRectsRTree();
//   _region_query->printFixedRectsRTree();
// }

}  // namespace idrc