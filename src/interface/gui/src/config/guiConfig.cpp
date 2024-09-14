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
#include "guiConfig.h"

namespace igui {
  GuiConfig* GuiConfig::_instance = nullptr;

  GuiConfig::GuiConfig() {
    _instance_tree.set_node_name("Instance");
    _instance_tree.add_option("Standard Cell", GuiOptionValue::kOn);
    _instance_tree.add_option("IO Cell");
    _instance_tree.add_option("Block");
    _instance_tree.add_option("Pad");
    // _instance_tree.add_option("Filler");
    // _instance_tree.add_option("EndCap");

    _net_tree.set_node_name("Net");
    _net_tree.add_option("Signal");
    _net_tree.add_option("Clock");
    _net_tree.add_option("Power");
    _net_tree.add_option("Ground");

    _clock_tree.set_node_name("Clock Tree");
    _clock_tree.add_option("Show");

    _specialnet_tree.set_node_name("Special Net");
    _specialnet_tree.add_option("Power");
    _specialnet_tree.add_option("Ground");

    _trackgrid_tree.set_node_name("Track Grid");
    _trackgrid_tree.add_option("Prefer");
    _trackgrid_tree.add_option("NonPrefer");

    _layer_tree.set_node_name("Layer");

    _shape_tree.set_node_name("Shape");
    _shape_tree.add_option("Instance Pin");
    _shape_tree.add_option("Instance Obs");
    _shape_tree.add_option("Instance Pdn");
    _shape_tree.add_option("IO Pin");

    _drc_tree.set_node_name("DRC");
    _drc_tree.add_option("Cut EOL Spacing");
    _drc_tree.add_option("Cut Spacing");
    _drc_tree.add_option("Cut Enclosure");
    _drc_tree.add_option("EndOfLine Spacing");
    _drc_tree.add_option("Metal Short");
    _drc_tree.add_option("ParallelRunLength Spacing");
    _drc_tree.add_option("Notch Spacing");
    _drc_tree.add_option("MinStep");
    _drc_tree.add_option("Minimum Area");
  }

  void GuiConfig::UpdateLayerTree(vector<string> layer_name_list) {
    _layer_tree.clear();
    for (auto layer_name : layer_name_list) {
      _layer_tree.add_option(layer_name);
    }
  }

  GuiTreeNode* GuiConfig::findTreeNodeByNodeName(string node_name) {
    if (node_name == "Layer") {
      return &_layer_tree;
    } else if (node_name == "Instance") {
      return &_instance_tree;
    } else if (node_name == "Net") {
      return &_net_tree;
    } else if (node_name == "Special Net") {
      return &_specialnet_tree;
    } else if (node_name == "Track Grid") {
      return &_trackgrid_tree;
    } else if (node_name == "Shape") {
      return &_shape_tree;
    } else if (node_name == "DRC") {
      return &_drc_tree;
    } else if (node_name == "Clock Tree") {
      return &_clock_tree;
    } else {
      return nullptr;
    }
  }

  /**
   * @Brief :
   * @param  value true or false
   * @param  node_name node name selected
   * @param  parent_name if empty, save all the data of node_name
   */
  void GuiConfig::saveData(bool value, string node_name, string parent_name) {
    if (parent_name.empty()) {
      /// save all the data under node
      GuiTreeNode* tree_node = findTreeNodeByNodeName(node_name);
      if (tree_node != nullptr) {
        tree_node->set_option_list(value);
      }
    } else {
      /// find parent node
      GuiTreeNode* parent_node = findTreeNodeByNodeName(parent_name);
      if (parent_node != nullptr) {
        parent_node->set_option(node_name, value);
      }
    }
  }

}  // namespace igui