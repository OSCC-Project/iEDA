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
#include "guispeedupdesign.h"

#include <QString>

#include "guiConfig.h"

void GuiSpeedupDesign::clear() {
  if (_type == DbSetupType::kClockTree) {
    _clock_list->clear();
  } else {
    _instance_list->clear();
    _power_container->clear();
    _ground_container->clear();
    _signal_grid_container->clear();
    _signal_panel_prefer_container->clear();
    _signal_panel_nonprefer_container->clear();
    _clock_grid_container->clear();
    _clock_panel_prefer_container->clear();
    _clock_panel_nonprefer_container->clear();

    _power_via_container->clear();
    _ground_via_container->clear();
    _signal_via_container->clear();
    _clock_via_container->clear();

    _track_grid_prefer_container->clear();
    _track_grid_nonprefer_container->clear();
    _gcell_x_list->clear();
    _gcell_y_list->clear();

    _cut_eol_container->clear();
    _cut_spacing_container->clear();
    _cut_enclosure_container->clear();
    _eol_container->clear();
    _metal_short_container->clear();
    _prl_container->clear();
    _notch_container->clear();
    _min_step_container->clear();
    _min_area_container->clear();
  }
}

void GuiSpeedupDesign::init(QRectF boundingbox) {
  if (_type == DbSetupType::kClockTree) {
  } else {
    QRectF new_box = boundingbox;
    new_box.setLeft(boundingbox.left() - 1);
    new_box.setRight(boundingbox.right() + 1);
    new_box.setTop(boundingbox.top() - 1);
    new_box.setBottom(boundingbox.bottom() + 1);
    _instance_list->init(new_box);
    initViaContainer(new_box);
    initPdnContainer(new_box, _power_container, GuiSpeedupItemType::kPdnPower);
    initPdnContainer(new_box, _ground_container, GuiSpeedupItemType::kPdnGround);
    initNetContainer(new_box);
    initClockContainer(new_box);
    initTrackGridContainer(new_box);
    initDrcContainer(new_box);
  }
}

void GuiSpeedupDesign::initViaContainer(QRectF boundingbox) {
  auto initViaList = [](GuiSpeedupViaList* via_list, IdbLayer* cut_layer, QRectF boundingbox) -> void {
    QColor color = attributeInst->getLayerColor(cut_layer->get_name());
    via_list->set_color(color);
    via_list->set_layer_name(QString::fromStdString(cut_layer->get_name()));
    via_list->set_order(cut_layer->get_order());
    via_list->init(boundingbox);
  };

  for (IdbLayer* cut_layer : _layout->get_layers()->get_cut_layers()) {
    initViaList(_signal_via_container->addViaList(), cut_layer, boundingbox);
    initViaList(_clock_via_container->addViaList(), cut_layer, boundingbox);
    initViaList(_power_via_container->addViaList(), cut_layer, boundingbox);
    initViaList(_ground_via_container->addViaList(), cut_layer, boundingbox);
  }
}

void GuiSpeedupDesign::initPdnContainer(QRectF boundingbox, GuiSpeedupWireContainer* pdn_container,
                                        GuiSpeedupItemType type) {
  for (IdbLayer* layer : _layout->get_layers()->get_routing_layers()) {
    IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(layer);
    GuiSpeedupWireList* wire_list  = pdn_container->addWireList(type);
    QColor color                   = attributeInst->getLayerColor(routing_layer->get_name());
    wire_list->set_color(color);
    wire_list->set_layer_name(QString::fromStdString(routing_layer->get_name()));
    wire_list->set_order(routing_layer->get_order());

    wire_list->initPanel(boundingbox, routing_layer->get_direction());
  }
}

void GuiSpeedupDesign::initNetContainer(QRectF boundingbox) {
  /// init grid
  initNetGridContainer(boundingbox);
  /// init panel
  initNetPanelPreferContainer(boundingbox);
  initNetPanelNonPreferContainer(boundingbox);
}

void GuiSpeedupDesign::initNetGridContainer(QRectF boundingbox) {
  /// init grid
  for (IdbLayer* layer : _layout->get_layers()->get_routing_layers()) {
    IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(layer);
    GuiSpeedupWireList* wire_list  = _signal_grid_container->addWireList(GuiSpeedupItemType::kSignal);
    QColor color                   = attributeInst->getLayerColor(routing_layer->get_name());
    wire_list->set_color(color);
    wire_list->set_layer_name(QString::fromStdString(routing_layer->get_name()));
    wire_list->set_order(routing_layer->get_order());
    wire_list->init(boundingbox, routing_layer->get_direction());
  }
}

void GuiSpeedupDesign::initNetPanelPreferContainer(QRectF boundingbox) {
  /// init grid
  for (IdbLayer* layer : _layout->get_layers()->get_routing_layers()) {
    IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(layer);
    GuiSpeedupWireList* wire_list  = _signal_panel_prefer_container->addWireList(GuiSpeedupItemType::kSignal);
    QColor color                   = attributeInst->getLayerColor(routing_layer->get_name());
    wire_list->set_color(color);
    wire_list->set_layer_name(QString::fromStdString(routing_layer->get_name()));
    wire_list->set_order(routing_layer->get_order());
    wire_list->initPanel(boundingbox, routing_layer->get_direction());
  }
}

void GuiSpeedupDesign::initNetPanelNonPreferContainer(QRectF boundingbox) {
  /// init grid
  for (IdbLayer* layer : _layout->get_layers()->get_routing_layers()) {
    IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(layer);
    GuiSpeedupWireList* wire_list  = _signal_panel_nonprefer_container->addWireList(GuiSpeedupItemType::kSignal);
    QColor color                   = attributeInst->getLayerColor(routing_layer->get_name());
    wire_list->set_color(color);
    wire_list->set_layer_name(QString::fromStdString(routing_layer->get_name()));
    wire_list->set_order(routing_layer->get_order());
    wire_list->initPanel(boundingbox, routing_layer->get_nonprefer_direction());
  }
}

void GuiSpeedupDesign::initClockContainer(QRectF boundingbox) {
  /// init grid
  initClockGridContainer(boundingbox);
  /// init panel
  initClockPanelPreferContainer(boundingbox);
  initClockPanelNonPreferContainer(boundingbox);
}

void GuiSpeedupDesign::initClockGridContainer(QRectF boundingbox) {
  /// init grid
  for (IdbLayer* layer : _layout->get_layers()->get_routing_layers()) {
    IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(layer);
    GuiSpeedupWireList* wire_list  = _clock_grid_container->addWireList(GuiSpeedupItemType::kSignalClock);
    QColor color                   = attributeInst->getLayerColor(routing_layer->get_name());
    wire_list->set_color(color);
    wire_list->set_layer_name(QString::fromStdString(routing_layer->get_name()));
    wire_list->set_order(routing_layer->get_order());
    wire_list->init(boundingbox, routing_layer->get_direction());
  }
}

void GuiSpeedupDesign::initClockPanelPreferContainer(QRectF boundingbox) {
  /// init grid
  for (IdbLayer* layer : _layout->get_layers()->get_routing_layers()) {
    IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(layer);
    GuiSpeedupWireList* wire_list  = _clock_panel_prefer_container->addWireList(GuiSpeedupItemType::kSignalClock);
    QColor color                   = attributeInst->getLayerColor(routing_layer->get_name());
    wire_list->set_color(color);
    wire_list->set_layer_name(QString::fromStdString(routing_layer->get_name()));
    wire_list->set_order(routing_layer->get_order());
    wire_list->initPanel(boundingbox, routing_layer->get_direction());
  }
}

void GuiSpeedupDesign::initClockPanelNonPreferContainer(QRectF boundingbox) {
  /// init grid
  for (IdbLayer* layer : _layout->get_layers()->get_routing_layers()) {
    IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(layer);
    GuiSpeedupWireList* wire_list  = _clock_panel_nonprefer_container->addWireList(GuiSpeedupItemType::kSignalClock);
    QColor color                   = attributeInst->getLayerColor(routing_layer->get_name());
    wire_list->set_color(color);
    wire_list->set_layer_name(QString::fromStdString(routing_layer->get_name()));
    wire_list->set_order(routing_layer->get_order());
    wire_list->initPanel(boundingbox, routing_layer->get_nonprefer_direction());
  }
}

void GuiSpeedupDesign::initTrackGridContainer(QRectF boundingbox) {
  /// init prefer grid
  for (IdbLayer* layer : _layout->get_layers()->get_routing_layers()) {
    IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(layer);
    GuiSpeedupGridList* grid_list  = _track_grid_prefer_container->addGridList(GuiSpeedupItemType::kTrackGrid);
    QColor color                   = attributeInst->getLayerColor(routing_layer->get_name());
    grid_list->set_color(color);
    grid_list->set_layer_name(QString::fromStdString(routing_layer->get_name()));
    grid_list->set_order(routing_layer->get_order());
    grid_list->initPanel(boundingbox, routing_layer->get_direction());
  }

  /// init nonprefer grid
  for (IdbLayer* layer : _layout->get_layers()->get_routing_layers()) {
    IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(layer);
    GuiSpeedupGridList* grid_list  = _track_grid_nonprefer_container->addGridList(GuiSpeedupItemType::kTrackGrid);
    QColor color                   = attributeInst->getLayerColor(routing_layer->get_name());
    grid_list->set_color(color);
    grid_list->set_layer_name(QString::fromStdString(routing_layer->get_name()));
    grid_list->set_order(routing_layer->get_order());
    grid_list->initPanel(boundingbox, routing_layer->get_nonprefer_direction());
  }
}

void GuiSpeedupDesign::initDrcContainer(QRectF boundingbox) {
  auto drc_opt_list = guiConfig->get_drc_tree().get_option_list();
  for (auto opt : drc_opt_list) {
    auto container = get_drc_container(opt._name);

    auto layer_list = _layout->get_layers()->get_layers();

    for (auto layer : layer_list) {
      GuiSpeedupDrcList* drc_list = container->addDrcList(container->get_type());
      drc_list->set_layer_name(QString::fromStdString(layer->get_name()));
      drc_list->set_order(layer->get_order());
      drc_list->init(boundingbox, layer->get_order());
    }
  }
}

void GuiSpeedupDesign::finishCreateItem() {
  if (_type == DbSetupType::kClockTree) {
    return;
  }

  _instance_list->finishCreateItem();
  _power_container->finishCreateItem();
  _ground_container->finishCreateItem();
  _signal_grid_container->finishCreateItem();
  _signal_panel_prefer_container->finishCreateItem();
  _signal_panel_nonprefer_container->finishCreateItem();
  _signal_via_container->finishCreateItem();
  _clock_via_container->finishCreateItem();
  _power_via_container->finishCreateItem();
  _ground_via_container->finishCreateItem();
  _track_grid_prefer_container->finishCreateItem();
  _track_grid_nonprefer_container->finishCreateItem();

  _cut_eol_container->finishCreateItem();
  _cut_spacing_container->finishCreateItem();
  _cut_enclosure_container->finishCreateItem();
  _eol_container->finishCreateItem();
  _metal_short_container->finishCreateItem();
  _prl_container->finishCreateItem();
  _notch_container->finishCreateItem();
  _min_step_container->finishCreateItem();
  _min_area_container->finishCreateItem();
}

void GuiSpeedupDesign::update(std::string node_name, std::string parent_name) {
  if (_type == DbSetupType::kClockTree) {
    return;
  }

  if (node_name == "Shape") {
    _instance_list->update();
    return;
  }

  if (node_name == "Instance") {
    _instance_list->update();
    return;
  }

  if (node_name == "DRC") {
    _cut_eol_container->update();
    _cut_spacing_container->update();
    _cut_enclosure_container->update();
    _eol_container->update();
    _metal_short_container->update();
    _prl_container->update();
    _notch_container->update();
    _min_step_container->update();
    _min_area_container->update();
    return;
  }

  if (parent_name == "DRC" && node_name != "DRC") {
    auto container = get_drc_container(node_name);
    container->update();
    return;
  }

  if (parent_name == "Clock Tree" || node_name == "Clock Tree") {
    _scene->update();
    return;
  }

  if (node_name == "Net") {
    _signal_grid_container->update();
    _signal_panel_prefer_container->update();
    _signal_panel_nonprefer_container->update();
    _clock_grid_container->update();
    _clock_panel_prefer_container->update();
    _clock_panel_nonprefer_container->update();
    _signal_via_container->update();
    _clock_via_container->update();
    _power_via_container->update();
    _ground_via_container->update();
    return;
  }

  if (parent_name == "Net" && node_name == "Signal") {
    _signal_grid_container->update();
    _signal_panel_prefer_container->update();
    _signal_panel_nonprefer_container->update();
    _signal_via_container->update();
    return;
  }

  if (parent_name == "Net" && node_name == "Clock") {
    _clock_grid_container->update();
    _clock_panel_prefer_container->update();
    _clock_panel_nonprefer_container->update();
    _clock_via_container->update();
    return;
  }

  if (node_name == "Special Net") {
    _power_container->update();
    _ground_container->update();
    _power_via_container->update();
    _ground_via_container->update();
    return;
  }

  if (parent_name == "Special Net" && node_name == "Power") {
    _power_container->update();
    _power_via_container->update();
    return;
  }

  if (parent_name == "Special Net" && node_name == "Ground") {
    _ground_container->update();
    _ground_via_container->update();
    return;
  }

  if (node_name == "Track Grid") {
    _track_grid_prefer_container->update();
    _track_grid_nonprefer_container->update();
    return;
  }

  if (parent_name == "Track Grid" && node_name == "Prefer") {
    _track_grid_prefer_container->update();
    return;
  }

  if (parent_name == "Track Grid" && node_name == "NonPrefer") {
    _track_grid_nonprefer_container->update();
    return;
  }

  if (node_name == "Layer") {
    _scene->update();
    return;
  }

  if (parent_name == "Layer") {
    {
      auto container = _signal_grid_container->findWireList(node_name);
      if (container != nullptr) {
        container->update();
      }
    }

    {
      auto container = _signal_panel_prefer_container->findWireList(node_name);
      if (container != nullptr) {
        container->update();
      }
    }

    {
      auto container = _signal_panel_nonprefer_container->findWireList(node_name);
      if (container != nullptr) {
        container->update();
      }
    }

    {
      auto container = _clock_grid_container->findWireList(node_name);
      if (container != nullptr) {
        container->update();
      }
    }

    {
      auto container = _clock_panel_prefer_container->findWireList(node_name);
      if (container != nullptr) {
        container->update();
      }
    }

    {
      auto container = _clock_panel_nonprefer_container->findWireList(node_name);
      if (container != nullptr) {
        container->update();
      }
    }

    {
      auto container = _power_container->findWireList(node_name);
      if (container != nullptr) {
        container->update();
      }
    }

    {
      auto container = _ground_container->findWireList(node_name);
      if (container != nullptr) {
        container->update();
      }
    }

    {
      auto grid_container = _track_grid_prefer_container->findGridList(node_name);
      if (grid_container != nullptr) {
        grid_container->update();
      }
    }

    {
      auto grid_container = _track_grid_nonprefer_container->findGridList(node_name);
      if (grid_container != nullptr) {
        grid_container->update();
      }
    }

    {
      auto container = _signal_via_container->findViaList(node_name);
      if (container != nullptr) {
        container->update();
      }
    }

    {
      auto container = _clock_via_container->findViaList(node_name);
      if (container != nullptr) {
        container->update();
      }
    }

    {
      auto container = _power_via_container->findViaList(node_name);
      if (container != nullptr) {
        container->update();
      }
    }

    {
      auto container = _ground_via_container->findViaList(node_name);
      if (container != nullptr) {
        container->update();
      }
    }
  }
}