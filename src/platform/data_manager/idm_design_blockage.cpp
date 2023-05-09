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
/**
 * @File Name: dm_design_blockage.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-07-19
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "idm.h"

namespace idm {
/**
 * @brief add placement blockage
 *
 * @param llx
 * @param lly
 * @param urx
 * @param ury
 * @return IdbPlacementBlockage*
 */
IdbPlacementBlockage* DataManager::addPlacementBlockage(int32_t llx, int32_t lly, int32_t urx, int32_t ury)
{
  auto idb_core = _layout->get_core();
  auto blockage_list = _design->get_blockage_list();
  IdbPlacementBlockage* pl_blockage = blockage_list->add_blockage_placement();

  IdbRect* core = idb_core->get_bounding_box();
  if (core->isIntersection(IdbRect(llx, lly, urx, ury))) {
    std::vector<int32_t> rect_x, rect_y;
    rect_x.push_back(llx);
    rect_x.push_back(urx);
    rect_x.push_back(core->get_low_x());
    rect_x.push_back(core->get_high_x());

    rect_y.push_back(lly);
    rect_y.push_back(ury);
    rect_y.push_back(core->get_low_y());
    rect_y.push_back(core->get_high_y());

    sort(rect_x.begin(), rect_x.end(), [](int32_t a, int32_t b) { return a < b; });
    sort(rect_y.begin(), rect_y.end(), [](int32_t a, int32_t b) { return a < b; });

    pl_blockage->add_rect(rect_x[1], rect_y[1], rect_x[2], rect_y[2]);
  } else {
    pl_blockage->add_rect(llx, lly, urx, ury);
  }

  return pl_blockage;
}
/**
 * @brief add placement halo
 *
 * @param instance_name
 * @param distance_top
 * @param distance_bottom
 * @param distance_left
 * @param distance_right
 */
void DataManager::addPlacementHalo(const string& instance_name, int32_t distance_top, int32_t distance_bottom, int32_t distance_left,
                                   int32_t distance_right)
{
  auto inst_list = _design->get_instance_list();

  if (instance_name != "" && instance_name != "all") {
    auto instance = inst_list->find_instance(instance_name);
    if (instance == nullptr) {
      std::cout << "NO instance: " << instance_name << std::endl;
      return;
    }
    if (instance->is_unplaced()) {
      std::cout << "This instance " << instance_name << " is not placed!" << std::endl;
      return;
    }
    instance->set_bounding_box();
    int blk_llx = instance->get_bounding_box()->get_low_x() - distance_left;
    int blk_urx = instance->get_bounding_box()->get_high_x() + distance_right;
    int blk_lly = instance->get_bounding_box()->get_low_y() - distance_bottom;
    int blk_ury = instance->get_bounding_box()->get_high_y() + distance_top;
    addPlacementBlockage(blk_llx, blk_lly, blk_urx, blk_ury);
  } else if (instance_name == "all") {
    if (inst_list->get_instance_list().size() == 0) {
      std::cout << "NO macro in this design." << std::endl;
      return;
    }
    for (auto instance : inst_list->get_instance_list()) {
      if (instance->is_unplaced()) {
        std::cout << "This instance " << instance_name << " is not placed!" << std::endl;
        continue;
      }
      instance->set_bounding_box();
      int blk_llx = instance->get_bounding_box()->get_low_x() - distance_left;
      int blk_urx = instance->get_bounding_box()->get_high_x() + distance_right;
      int blk_lly = instance->get_bounding_box()->get_low_y() - distance_bottom;
      int blk_ury = instance->get_bounding_box()->get_high_y() + distance_top;
      addPlacementBlockage(blk_llx, blk_lly, blk_urx, blk_ury);
    }
  }
}
/**
 * @brief remove blockage for except pg net
 *
 */
void DataManager::removeBlockageExceptPGNet()
{
  auto blockage_list = _design->get_blockage_list();
  if (blockage_list != nullptr) {
    blockage_list->removeExceptPgNetBlockageList();
  }
}

void DataManager::clearBlockage(string type)
{
  auto blockage_list = _design->get_blockage_list();
  if (blockage_list != nullptr) {
    if (type == "routing") {
      blockage_list->clearRoutingBlockage();
    } else if (type == "placement") {
      blockage_list->clearPlacementBlockage();
    } else {
      blockage_list->reset();
    }
  }
}
/**
 * @brief add routing blockage
 *
 * @param llx
 * @param lly
 * @param urx
 * @param ury
 * @param layers
 * @param is_except_pgnet
 */
void DataManager::addRoutingBlockage(int32_t llx, int32_t lly, int32_t urx, int32_t ury, const std::vector<std::string>& layers,
                                     const bool& is_except_pgnet)
{
  auto blockage_list = _design->get_blockage_list();
  auto layer_list = _layout->get_layers();

  for (std::string layer : layers) {
    IdbRoutingBlockage* rt_blockage = blockage_list->add_blockage_routing(layer);
    rt_blockage->set_layer(layer_list->find_layer(layer));
    rt_blockage->add_rect(llx, lly, urx, ury);
    rt_blockage->set_except_pgnet(is_except_pgnet);
  }
}
/**
 * @brief add routing halo
 *
 * @param instance_name
 * @param layers
 * @param distance_top
 * @param distance_bottom
 * @param distance_left
 * @param distance_right
 * @param is_except_pgnet
 */
void DataManager::addRoutingHalo(const string& instance_name, const std::vector<std::string>& layers, int32_t distance_top,
                                 int32_t distance_bottom, int32_t distance_left, int32_t distance_right, const bool& is_except_pgnet)
{
  auto idb_inst_list = _design->get_instance_list();

  if (instance_name != "" && instance_name != "all") {
    IdbInstance* instance = idb_inst_list->find_instance(instance_name);
    if (instance == nullptr) {
      std::cout << "NO instance: " << instance_name << std::endl;
      return;
    }
    if (instance->is_unplaced()) {
      std::cout << "This instance " << instance_name << " is not placed!" << std::endl;
      return;
    }
    instance->set_bounding_box();
    int blk_llx = instance->get_bounding_box()->get_low_x() - distance_left;
    int blk_urx = instance->get_bounding_box()->get_high_x() + distance_right;
    int blk_lly = instance->get_bounding_box()->get_low_y() - distance_bottom;
    int blk_ury = instance->get_bounding_box()->get_high_y() + distance_top;
    addRoutingBlockage(blk_llx, blk_lly, blk_urx, blk_ury, layers, is_except_pgnet);
  } else if (instance_name == "all") {
    if (idb_inst_list->get_instance_list().size() == 0) {
      std::cout << "NO macro in this design." << std::endl;
      return;
    }
    for (IdbInstance* instance : idb_inst_list->get_instance_list()) {
      if (instance->is_unplaced()) {
        std::cout << "This instance " << instance_name << " is not placed!" << std::endl;
        continue;
      }
      instance->set_bounding_box();
      int blk_llx = instance->get_bounding_box()->get_low_x() - distance_left;
      int blk_urx = instance->get_bounding_box()->get_high_x() + distance_right;
      int blk_lly = instance->get_bounding_box()->get_low_y() - distance_bottom;
      int blk_ury = instance->get_bounding_box()->get_high_y() + distance_top;
      addRoutingBlockage(blk_llx, blk_lly, blk_urx, blk_ury, layers, is_except_pgnet);
    }
  }
}

}  // namespace idm
