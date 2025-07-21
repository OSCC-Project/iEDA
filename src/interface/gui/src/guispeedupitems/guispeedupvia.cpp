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
#include "guispeedupvia.h"

#include "guiConfig.h"
#include "guiConfigTree.h"

void GuiSpeedupVia::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) {
  Q_UNUSED(widget);
  if (!is_visible()) {
    return;
  }
  //   const qreal lod = option->levelOfDetailFromTransform(painter->worldTransform());

  GuiSpeedupItem::paint(painter, option, widget);
}

void GuiSpeedupVia::paintScaleTop(QPainter* painter, qreal lod) {
  /// do not paint
}

void GuiSpeedupVia::paintScale_1st(QPainter* painter, qreal lod) {
  QPen pen = _pen;
  pen.setWidthF(pen.widthF() / lod);
  pen.setWidthF(0);
  painter->setPen(pen);

  QBrush brush = _brush;
  brush.setTransform(painter->worldTransform().inverted());
  //   painter->setBrush(brush);
  painter->setBrush(Qt::NoBrush);

  // #pragma omp parallel for
  for (auto& rect : get_rect_list()) {
    painter->drawPoint(rect.center());
  }
}

void GuiSpeedupVia::paintScale_2nd(QPainter* painter, qreal lod) {
  QPen pen = _pen;
  pen.setWidthF(pen.widthF() / lod);
  pen.setWidthF(0);
  painter->setPen(pen);
  painter->setBrush(Qt::NoBrush);

  // #pragma omp parallel for
  for (auto& rect : get_rect_list()) {
    painter->drawRect(rect);
    painter->drawLine(rect.bottomLeft(), rect.topRight());
    painter->drawLine(rect.topLeft(), rect.bottomRight());
  }
}

void GuiSpeedupVia::paintScale_3rd(QPainter* painter, qreal lod) {
  QPen pen = _pen;
  pen.setWidthF(pen.widthF() / lod);
  pen.setWidthF(0);
  painter->setPen(pen);

  QBrush brush = _brush;
  brush.setTransform(painter->worldTransform().inverted());
  painter->setBrush(brush);

  // #pragma omp parallel for
  for (auto& rect : get_rect_list()) {
    painter->drawRect(rect);
    painter->drawLine(rect.bottomLeft(), rect.topRight());
    painter->drawLine(rect.topLeft(), rect.bottomRight());
  }
}

bool GuiSpeedupVia::is_visible() {
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

  return true;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GuiSpeedupViaList::init(QRectF boundingbox) {
  //   qreal step_x = boundingbox.width() / GUI_VIA_GRID_COL;
  //   qreal step_y = boundingbox.height() / GUI_VIA_GRID_ROW;

  //   qreal y_start = boundingbox.top();
  //   while (y_start <= boundingbox.bottom()) {
  //     qreal x_start = boundingbox.left();
  //     while (x_start <= boundingbox.right()) {
  //       GuiSpeedupVia* item_new = addItem();
  //       item_new->set_bounding_rect(QRectF(x_start, y_start, step_x, step_y));
  //       x_start += step_x;
  //     }

  //     y_start += step_y;
  //   }

  qreal step_x     = boundingbox.width() / GUI_VIA_GRID_COL;
  qreal step_y     = boundingbox.height() / GUI_VIA_GRID_ROW;
  int32_t number_x = GUI_VIA_GRID_COL + 1;
  int32_t number_y = GUI_VIA_GRID_ROW + 1;
  qreal x_start    = boundingbox.left();
  qreal y_start    = boundingbox.top();
  for (int y = 0; y < number_y; ++y) {
    for (int x = 0; x < number_x; ++x) {
      GuiSpeedupVia* item_new = addCurrentItem();
      item_new->set_bounding_rect(QRectF(x_start + x * step_x, y_start + y * step_y, step_x, step_y));
    }
  }

  set_number(number_x, number_y);
  set_step(step_x, step_y);
}
