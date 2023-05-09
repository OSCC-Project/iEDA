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
#include "guispeedupwire.h"

#include "guiConfig.h"
#include "guiConfigTree.h"

void GuiSpeedupWire::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) {
  Q_UNUSED(widget);

  //   const qreal lod = option->levelOfDetailFromTransform(painter->worldTransform());

  GuiSpeedupItem::paint(painter, option, widget);
}

void GuiSpeedupWire::paintScaleTop(QPainter* painter, qreal lod) {
  if (is_port()) {
    return;
  }

  QPen pen = _pen;
  pen.setWidthF(0);
  painter->setPen(pen);
  painter->setBrush(Qt::BrushStyle::NoBrush);

  auto& rect_list = get_rect_list();
  for (size_t i = 0; i < get_rect_number(); ++i) {
    /// horizontal
    if (rect_list[i].width() > rect_list[i].height()) {
      painter->drawLine(rect_list[i].left(), rect_list[i].center().y(), rect_list[i].right(), rect_list[i].center().y());
    } else {
      /// vertical
      painter->drawLine(rect_list[i].center().x(), rect_list[i].top(), rect_list[i].center().x(), rect_list[i].bottom());
    }
  }
}

void GuiSpeedupWire::paintScale_1st(QPainter* painter, qreal lod) {
  if (is_port()) {
    return;
  }

  QPen pen = _pen;
  //   pen.setWidthF(pen.widthF() / lod);
  pen.setWidthF(0);
  painter->setPen(pen);
  painter->setBrush(Qt::BrushStyle::NoBrush);
  auto& rect_list = get_rect_list();
  for (size_t i = 0; i < rect_list.size(); i++) {
    painter->drawRect(rect_list[i]);
  }
}

void GuiSpeedupWire::paintScale_2nd(QPainter* painter, qreal lod) {
  QPen pen = _pen;
  pen.setWidthF(0);
  painter->setPen(pen);
  painter->setBrush(Qt::BrushStyle::NoBrush);

  auto& rect_list = get_rect_list();
  for (size_t i = 0; i < rect_list.size(); i++) {
    painter->drawRect(rect_list[i]);
  }
}

void GuiSpeedupWire::paintScale_3rd(QPainter* painter, qreal lod) {
  QPen pen = _pen;
  pen.setWidthF(pen.widthF() / lod);
  pen.setWidthF(0);
  painter->setPen(pen);

  QBrush brush = _brush;
  brush.setTransform(painter->worldTransform().inverted());
  painter->setBrush(brush);

  auto& rect_list = get_rect_list();
  for (size_t i = 0; i < rect_list.size(); i++) {
    painter->drawRect(rect_list[i]);
  }
}

bool GuiSpeedupWire::is_visible() {
  if (get_type() == GuiSpeedupItemType::kNet) {
    igui::GuiTreeNode& tree_node = guiConfig->get_net_tree();
    return guiConfig->isLayerVisible((int32_t)zValue()) && (tree_node.isChecked("Signal") || tree_node.isChecked("Clock") ||
                                                            tree_node.isChecked("Power") || tree_node.isChecked("Ground"));
  }

  if (get_type() == GuiSpeedupItemType::kSignal) {
    igui::GuiTreeNode& tree_node = guiConfig->get_net_tree();
    return tree_node.isChecked("Signal") && guiConfig->isLayerVisible((int32_t)zValue());
  }

  if (get_type() == GuiSpeedupItemType::kSignalClock) {
    igui::GuiTreeNode& tree_node = guiConfig->get_net_tree();
    return tree_node.isChecked("Clock") && guiConfig->isLayerVisible((int32_t)zValue());
  }

  if (get_type() == GuiSpeedupItemType::kSignalPower) {
    igui::GuiTreeNode& tree_node = guiConfig->get_net_tree();
    return tree_node.isChecked("Power") && guiConfig->isLayerVisible((int32_t)zValue());
  }

  if (get_type() == GuiSpeedupItemType::kSignalGround) {
    igui::GuiTreeNode& tree_node = guiConfig->get_net_tree();
    return tree_node.isChecked("Ground") && guiConfig->isLayerVisible((int32_t)zValue());
  }

  if (get_type() == GuiSpeedupItemType::kPdnPower) {
    igui::GuiTreeNode& tree_node = guiConfig->get_specialnet_tree();
    return tree_node.isChecked("Power") && guiConfig->isLayerVisible((int32_t)zValue());
  }

  if (get_type() == GuiSpeedupItemType::kPdnGround) {
    igui::GuiTreeNode& tree_node = guiConfig->get_specialnet_tree();
    return tree_node.isChecked("Ground") && guiConfig->isLayerVisible((int32_t)zValue());
  }

  /// if pin shape in instance
  if (get_type() == GuiSpeedupItemType::kInstStandarCell) {
    igui::GuiTreeNode& tree_node_inst      = guiConfig->get_instance_tree();
    igui::GuiTreeNode& tree_node_shape_pin = guiConfig->get_shape_tree();
    return (tree_node_inst.isChecked("Standard Cell") && tree_node_shape_pin.isChecked("Instance Pin") &&
            guiConfig->isLayerVisible((int32_t)zValue()));
  }

  if (get_type() == GuiSpeedupItemType::kInstIoCell) {
    igui::GuiTreeNode& tree_node = guiConfig->get_instance_tree();
    return tree_node.isChecked("IO Cell");
  }

  if (get_type() == GuiSpeedupItemType::kInstBlock) {
    igui::GuiTreeNode& tree_node = guiConfig->get_instance_tree();
    return tree_node.isChecked("Block");
  }

  if (get_type() == GuiSpeedupItemType::kInstPad) {
    igui::GuiTreeNode& tree_node = guiConfig->get_instance_tree();
    return tree_node.isChecked("Pad");
  }

  return true;
}

bool GuiSpeedupWire::is_pdn() {
  if (get_type() == GuiSpeedupItemType::kPdn || get_type() == GuiSpeedupItemType::kPdnPower ||
      get_type() == GuiSpeedupItemType::kPdnGround) {
    return true;
  }

  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GuiSpeedupWireList::init(QRectF boundingbox, IdbLayerDirection direction) {
  //   qreal step_x = direction == IdbLayerDirection::kHorizontal ? boundingbox.width() / GUI_NET_GRID_PREFER
  //                                                              : boundingbox.width() / GUI_NET_GRID_NONPREFER;
  //   qreal step_y = direction == IdbLayerDirection::kHorizontal ? boundingbox.height() / GUI_NET_GRID_NONPREFER
  //                                                              : boundingbox.height() / GUI_NET_GRID_PREFER;

  //   std::cout << " step_x = " << step_x << " step_y =  " << step_y << std::endl;

  //   qreal y_start = boundingbox.top();
  //   while (y_start <= boundingbox.bottom()) {
  //     qreal x_start = boundingbox.left();
  //     while (x_start <= boundingbox.right()) {
  //       GuiSpeedupWire* item_new = addItem();
  //       item_new->set_bounding_rect(QRectF(x_start, y_start, step_x, step_y));
  //       x_start += step_x;
  //     }

  //     y_start += step_y;
  //   }
  qreal step_x     = direction == IdbLayerDirection::kHorizontal ? boundingbox.width() / GUI_NET_GRID_PREFER
                                                                 : boundingbox.width() / GUI_NET_GRID_NONPREFER;
  qreal step_y     = direction == IdbLayerDirection::kHorizontal ? boundingbox.height() / GUI_NET_GRID_NONPREFER
                                                                 : boundingbox.height() / GUI_NET_GRID_PREFER;
  int32_t number_x = direction == IdbLayerDirection::kHorizontal ? GUI_NET_GRID_PREFER + 1 : GUI_NET_GRID_NONPREFER + 1;
  int32_t number_y = direction == IdbLayerDirection::kHorizontal ? GUI_NET_GRID_NONPREFER + 1 : GUI_NET_GRID_PREFER + 1;
  qreal x_start    = boundingbox.left();
  qreal y_start    = boundingbox.top();
  for (int y = 0; y < number_y; ++y) {
    for (int x = 0; x < number_x; ++x) {
      GuiSpeedupWire* item_new = addCurrentItem();
      item_new->set_bounding_rect(QRectF(x_start + x * step_x, y_start + y * step_y, step_x, step_y));
    }
  }

  set_number(number_x, number_y);
  set_step(step_x, step_y);
}

void GuiSpeedupWireList::initPanel(QRectF boundingbox, IdbLayerDirection direction) {
  qreal step_x = boundingbox.width() / GUI_PDN_GRID_COL;
  qreal step_y = boundingbox.height() / GUI_PDN_GRID_ROW;

  int32_t number_x = GUI_PDN_GRID_COL + 1;
  int32_t number_y = GUI_PDN_GRID_ROW + 1;
  qreal x_start    = boundingbox.left();
  qreal y_start    = boundingbox.top();
  if (direction == IdbLayerDirection::kHorizontal) {
    /// horizontal
    // qreal y_start = boundingbox.top();
    // while (y_start <= boundingbox.bottom()) {
    //   GuiSpeedupWire* item_new = addItem();
    //   item_new->set_bounding_rect(QRectF(boundingbox.left(), y_start, boundingbox.width(), step_y));
    //   y_start += step_y;
    // }

    for (int y = 0; y < number_y; ++y) {
      GuiSpeedupWire* item_new = addCurrentItem();
      item_new->set_bounding_rect(QRectF(x_start, y_start + y * step_y, boundingbox.width(), step_y));
    }
    set_step(boundingbox.width(), step_y);
    set_number(1, number_y);
  } else {
    /// vertical
    // qreal x_start = boundingbox.left();
    // while (x_start <= boundingbox.right()) {
    //   GuiSpeedupWire* item_new = addItem();
    //   item_new->set_bounding_rect(QRectF(x_start, boundingbox.top(), step_x, boundingbox.height()));
    //   item_new->setZValue(_z_order);
    //   x_start += step_x;
    // }

    for (int x = 0; x < number_x; ++x) {
      GuiSpeedupWire* item_new = addCurrentItem();
      item_new->set_bounding_rect(QRectF(x_start + x * step_x, y_start, step_x, boundingbox.height()));
    }
    set_step(step_x, boundingbox.height());
    set_number(number_x, 1);
  }
  //   qreal step_x     = boundingbox.width() / GUI_PDN_GRID_ROW;
  //   qreal step_y     = boundingbox.height() / GUI_PDN_GRID_COL;
  //   int32_t number_x = direction == IdbLayerDirection::kHorizontal ? GUI_NET_GRID_PREFER + 1 : GUI_NET_GRID_NONPREFER + 1;
  //   int32_t number_y = direction == IdbLayerDirection::kHorizontal ? GUI_NET_GRID_NONPREFER + 1 : GUI_NET_GRID_PREFER + 1;
  //   qreal x_start    = boundingbox.left();
  //   qreal y_start    = boundingbox.top();
  //   for (int y = 0; y < number_y; ++y) {
  //     for (int x = 0; x < number_x; ++x) {
  //       GuiSpeedupWire* item_new = addItem();
  //       item_new->set_bounding_rect(QRectF(x_start + x * step_x, y_start + y * step_y, step_x, step_y));
  //     }
  //   }

  GuiSpeedupItemList::initPanel(boundingbox, direction);
}
