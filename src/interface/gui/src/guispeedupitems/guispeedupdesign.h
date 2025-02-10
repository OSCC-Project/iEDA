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
 * @file GuiSpeedupDesign.h
 * @author Yell

 * @brief
 * @version 0.1
 * @date 2021-11-29(V0.1)
 *
 *
 *
 */

#ifndef GUI_SPEEDUP_DESIGN
#define GUI_SPEEDUP_DESIGN

#include "IdbLayer.h"
#include "IdbLayout.h"
#include "guispeedupclocktree.h"
#include "guispeedupdrc.h"
#include "guispeedupgrid.h"
#include "guispeedupinstance.h"
#include "guispeedupvia.h"
#include "guispeedupwire.h"

class GuiSpeedupDesign {
 public:
  GuiSpeedupDesign(GuiGraphicsScene* scene, DbSetupType type = DbSetupType::kChip) {
    _scene = scene;
    _type  = type;
    if (_type == DbSetupType::kClockTree) {
      _clock_list = new GuiSpeedupClockTreeItemList(scene);
    } else {
      _instance_list                    = new GuiSpeedupInstanceList(scene);
      _signal_via_container             = new GuiSeedupViaContainer(scene);
      _clock_via_container              = new GuiSeedupViaContainer(scene);
      _power_via_container              = new GuiSeedupViaContainer(scene);
      _ground_via_container             = new GuiSeedupViaContainer(scene);
      _power_container                  = new GuiSpeedupWireContainer(scene, GuiSpeedupItemType::kPdnPower);
      _ground_container                 = new GuiSpeedupWireContainer(scene, GuiSpeedupItemType::kPdnGround);
      _signal_grid_container            = new GuiSpeedupWireContainer(scene, GuiSpeedupItemType::kSignal);
      _signal_panel_prefer_container    = new GuiSpeedupWireContainer(scene, GuiSpeedupItemType::kSignal);
      _signal_panel_nonprefer_container = new GuiSpeedupWireContainer(scene, GuiSpeedupItemType::kSignal);
      _clock_grid_container             = new GuiSpeedupWireContainer(scene, GuiSpeedupItemType::kSignalClock);
      _clock_panel_prefer_container     = new GuiSpeedupWireContainer(scene, GuiSpeedupItemType::kSignalClock);
      _clock_panel_nonprefer_container  = new GuiSpeedupWireContainer(scene, GuiSpeedupItemType::kSignalClock);
      _track_grid_prefer_container      = new GuiSeedupGridContainer(scene, GuiSpeedupItemType::kTrackGrid);
      _track_grid_nonprefer_container   = new GuiSeedupGridContainer(scene, GuiSpeedupItemType::kTrackGrid);
      _gcell_x_list                     = new GuiSpeedupGridList(scene, GuiSpeedupItemType::kGCellGrid);
      _gcell_y_list                     = new GuiSpeedupGridList(scene, GuiSpeedupItemType::kGCellGrid);

      _cut_eol_container       = new GuiSpeedupDrcContainer(scene, GuiSpeedupItemType::kDrcCutEOL);
      _cut_spacing_container   = new GuiSpeedupDrcContainer(scene, GuiSpeedupItemType::kDrcCutSpacing);
      _cut_enclosure_container = new GuiSpeedupDrcContainer(scene, GuiSpeedupItemType::kDrcCutEnclosure);
      _eol_container           = new GuiSpeedupDrcContainer(scene, GuiSpeedupItemType::kDrcEOL);
      _metal_short_container   = new GuiSpeedupDrcContainer(scene, GuiSpeedupItemType::kDrcMetalShort);
      _prl_container           = new GuiSpeedupDrcContainer(scene, GuiSpeedupItemType::kDrcPRL);
      _notch_container         = new GuiSpeedupDrcContainer(scene, GuiSpeedupItemType::kDrcNotchSpacing);
      _min_step_container      = new GuiSpeedupDrcContainer(scene, GuiSpeedupItemType::kDrcMinStep);
      _min_area_container      = new GuiSpeedupDrcContainer(scene, GuiSpeedupItemType::kDrcMinArea);
    }
  }
  ~GuiSpeedupDesign() = default;
  /// getter
  GuiSpeedupInstanceList* get_instance_list() { return _instance_list; }

  GuiSeedupViaContainer* get_signal_via_container() { return _signal_via_container; }
  GuiSeedupViaContainer* get_clock_via_container() { return _clock_via_container; }
  GuiSeedupViaContainer* get_power_via_container() { return _power_via_container; }
  GuiSeedupViaContainer* get_ground_via_container() { return _ground_via_container; }

  GuiSpeedupWireContainer* get_power_container() { return _power_container; }
  GuiSpeedupWireContainer* get_ground_container() { return _ground_container; }
  GuiSpeedupWireContainer* get_net_grid_container() { return _signal_grid_container; }
  GuiSpeedupWireContainer* get_net_panel_prefer_container() { return _signal_panel_prefer_container; }
  GuiSpeedupWireContainer* get_net_panel_nonprefer_container() { return _signal_panel_nonprefer_container; }
  GuiSpeedupWireContainer* get_clock_grid_container() { return _clock_grid_container; }
  GuiSpeedupWireContainer* get_clock_panel_prefer_container() { return _clock_panel_prefer_container; }
  GuiSpeedupWireContainer* get_clock_panel_nonprefer_container() { return _clock_panel_nonprefer_container; }
  GuiSeedupGridContainer* get_track_grid_prefer_container() { return _track_grid_prefer_container; }
  GuiSeedupGridContainer* get_track_grid_nonprefer_container() { return _track_grid_nonprefer_container; }

  GuiSpeedupDrcContainer* get_drc_container(std::string drc_rule) {
    if (drc_rule == "Cut EOL Spacing") {
      return _cut_eol_container;
    }

    if (drc_rule == "Cut Spacing") {
      return _cut_spacing_container;
    }

    if (drc_rule == "Cut Enclosure") {
      return _cut_enclosure_container;
    }

    if (drc_rule == "EndOfLine Spacing") {
      return _eol_container;
    }

    if (drc_rule == "Metal Short") {
      return _metal_short_container;
    }

    if (drc_rule == "ParallelRunLength Spacing") {
      return _prl_container;
    }

    if (drc_rule == "Notch Spacing") {
      return _notch_container;
    }

    if (drc_rule == "MinStep") {
      return _min_step_container;
    }

    if (drc_rule == "Minimum Area") {
      return _min_area_container;
    }

    return nullptr;
  }

  GuiSpeedupClockTreeItemList* get_clock_list() { return _clock_list; }

  /// operator
  void set_idb_layout(IdbLayout* layout) { _layout = layout; }
  void init(QRectF boundingbox);
  void initViaContainer(QRectF boundingbox);
  void initPdnContainer(QRectF boundingbox, GuiSpeedupWireContainer* pdn_container, GuiSpeedupItemType type);
  void initNetContainer(QRectF boundingbox);
  void initNetGridContainer(QRectF boundingbox);
  void initNetPanelPreferContainer(QRectF boundingbox);
  void initNetPanelNonPreferContainer(QRectF boundingbox);
  void initClockContainer(QRectF boundingbox);
  void initClockGridContainer(QRectF boundingbox);
  void initClockPanelPreferContainer(QRectF boundingbox);
  void initClockPanelNonPreferContainer(QRectF boundingbox);
  void initTrackGridContainer(QRectF boundingbox);
  void initDrcContainer(QRectF boundingbox);

  void finishCreateItem();
  void clear();

  void update(std::string node_name, std::string parent_name = "");

  /// update
  void addUpdateItem(GuiSpeedupItem* item) { _update_list.insert(item); }
  void clearUpdateItemList() { _update_list.clear(); }
  void update() {
    for (auto item : _update_list) {
      item->update();
    }
  }

 private:
  IdbLayout* _layout = nullptr;
  GuiGraphicsScene* _scene;
  DbSetupType _type;

  /// update list
  std::set<GuiSpeedupItem*> _update_list;

  GuiSpeedupInstanceList* _instance_list;
  GuiSpeedupWireContainer* _power_container;

  GuiSpeedupWireContainer* _ground_container;

  GuiSpeedupWireContainer* _signal_grid_container;
  GuiSpeedupWireContainer* _signal_panel_prefer_container;
  GuiSpeedupWireContainer* _signal_panel_nonprefer_container;

  GuiSpeedupWireContainer* _clock_grid_container;
  GuiSpeedupWireContainer* _clock_panel_prefer_container;
  GuiSpeedupWireContainer* _clock_panel_nonprefer_container;

  GuiSeedupViaContainer* _power_via_container;
  GuiSeedupViaContainer* _ground_via_container;
  GuiSeedupViaContainer* _signal_via_container;
  GuiSeedupViaContainer* _clock_via_container;

  GuiSeedupGridContainer* _track_grid_prefer_container;
  GuiSeedupGridContainer* _track_grid_nonprefer_container;
  GuiSpeedupGridList* _gcell_x_list;
  GuiSpeedupGridList* _gcell_y_list;

  GuiSpeedupDrcContainer* _cut_eol_container;
  GuiSpeedupDrcContainer* _cut_spacing_container;
  GuiSpeedupDrcContainer* _cut_enclosure_container;
  GuiSpeedupDrcContainer* _eol_container;
  GuiSpeedupDrcContainer* _metal_short_container;
  GuiSpeedupDrcContainer* _prl_container;
  GuiSpeedupDrcContainer* _notch_container;
  GuiSpeedupDrcContainer* _min_step_container;
  GuiSpeedupDrcContainer* _min_area_container;

  GuiSpeedupClockTreeItemList* _clock_list;
};

#endif  // GUI_SPEEDUP_DESIGN
