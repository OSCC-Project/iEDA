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
#pragma once

#include <string>
#include <vector>

#include "guiConfigTree.h"

#define guiConfig (igui::GuiConfig::getInstance())

namespace igui {
  using namespace std;

  class GuiConfig {
   public:
    static GuiConfig* getInstance() {
      if (!_instance) {
        _instance = new GuiConfig;
      }
      return _instance;
    }

    /// getter
    GuiTreeNode& get_layer_tree() { return _layer_tree; }
    GuiTreeNode& get_instance_tree() { return _instance_tree; }
    GuiTreeNode& get_net_tree() { return _net_tree; }
    GuiTreeNode& get_specialnet_tree() { return _specialnet_tree; }
    GuiTreeNode& get_trackgrid_tree() { return _trackgrid_tree; }
    GuiTreeNode& get_shape_tree() { return _shape_tree; }
    GuiTreeNode& get_drc_tree() { return _drc_tree; }
    GuiTreeNode& get_clock_tree() { return _clock_tree; }

    /// setter
    void UpdateLayerTree(vector<string> layer_name_list);
    void enableInstance(bool b_enable = true) { _instance_tree.set_option_list(b_enable); }
    void enableNet(bool b_enable = true) { _net_tree.set_option_list(b_enable); }
    void enableClockTree(bool b_enable = true) { _clock_tree.set_option_list(b_enable); }

    /// operator
    GuiTreeNode* findTreeNodeByNodeName(string node_name);
    void saveData(bool value, string node_name, string parent_name = "");

    bool isLayerVisible(int32_t z_order) { return _layer_tree.isChecked(z_order); }

   private:
    static GuiConfig* _instance;

    GuiConfig();
    ~GuiConfig() = default;

    /// config
    GuiTreeNode _layer_tree;
    GuiTreeNode _instance_tree;
    GuiTreeNode _net_tree;
    GuiTreeNode _specialnet_tree;
    GuiTreeNode _trackgrid_tree;
    GuiTreeNode _shape_tree;
    GuiTreeNode _drc_tree;
    GuiTreeNode _clock_tree;
  };

}  // namespace igui
