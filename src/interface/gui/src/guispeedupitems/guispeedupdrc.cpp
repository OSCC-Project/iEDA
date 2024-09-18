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
#include "guispeedupdrc.h"

#include "guiConfig.h"
#include "guiConfigTree.h"
#include "omp.h"

void GuiSpeedupDrc::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) {
  Q_UNUSED(widget);

  GuiSpeedupItem::paint(painter, option, widget);
}

void GuiSpeedupDrc::paintScaleTop(QPainter* painter, qreal lod) {
  QPen pen = _pen;
  //   pen.setWidthF(pen.widthF() / lod);
  pen.setWidthF(0);
  painter->setPen(pen);
  // foreach (QRectF rect, _rect_list) { painter->drawPoint(rect.center()); }

  //   QBrush brush = _brush;
  //   brush.setTransform(painter->worldTransform().inverted());
  //   painter->setBrush(brush);

  // #pragma omp parallel for
  for (auto& rect : get_rect_list()) {
    painter->drawPoint(rect.center());
  }
}

void GuiSpeedupDrc::paintScale_1st(QPainter* painter, qreal lod) {
  /// draw instance as points
  QPen pen = _pen;
  //   pen.setWidthF(pen.widthF() / lod);
  pen.setWidthF(0);
  painter->setPen(pen);
  // foreach (QRectF rect, _rect_list) { painter->drawPoint(rect.center()); }

  QBrush brush = _brush;
  brush.setTransform(painter->worldTransform().inverted());
  painter->setBrush(Qt::NoBrush);

  // #pragma omp parallel for
  for (auto& rect : get_rect_list()) {
    painter->drawRect(rect);
  }
}

void GuiSpeedupDrc::paintScale_2nd(QPainter* painter, qreal lod) {
  QPen pen = _pen;
  pen.setWidthF(pen.widthF() / lod);
  painter->setPen(pen);

  /// draw instance as rect without pin shape
  QBrush brush = _brush;
  brush.setTransform(painter->worldTransform().inverted());
  painter->setBrush(brush);

  // #pragma omp parallel for
  for (auto& rect : get_rect_list()) {
    painter->drawRect(rect);
  }
}

void GuiSpeedupDrc::paintScale_3rd(QPainter* painter, qreal lod) {
  QPen pen = _pen;
  pen.setWidthF(pen.widthF() / lod);
  painter->setPen(pen);

  /// draw instance as rect with pin shape
  QBrush brush = _brush;
  brush.setTransform(painter->worldTransform().inverted());
  painter->setBrush(brush);

  // #pragma omp parallel for
  for (auto& rect : get_rect_list()) {
    painter->drawRect(rect);
    // drawText(painter, rect, "stdCell", lod);
  }
}

bool GuiSpeedupDrc::is_visible() {
  igui::GuiTreeNode& tree_node = guiConfig->get_drc_tree();

  if (get_type() == GuiSpeedupItemType::kDrcCutEOL) {
    return tree_node.isChecked("Cut EOL Spacing") && guiConfig->isLayerVisible(_z_order);
  }

  if (get_type() == GuiSpeedupItemType::kDrcCutSpacing) {
    return tree_node.isChecked("Cut Spacing") && guiConfig->isLayerVisible(_z_order);
  }

  if (get_type() == GuiSpeedupItemType::kDrcCutEnclosure) {
    return tree_node.isChecked("Cut Enclosure") && guiConfig->isLayerVisible(_z_order);
  }

  if (get_type() == GuiSpeedupItemType::kDrcEOL) {
    return tree_node.isChecked("EndOfLine Spacing") && guiConfig->isLayerVisible(_z_order);
  }

  if (get_type() == GuiSpeedupItemType::kDrcMetalShort) {
    return tree_node.isChecked("Metal Short") && guiConfig->isLayerVisible(_z_order);
  }

  if (get_type() == GuiSpeedupItemType::kDrcPRL) {
    return tree_node.isChecked("ParallelRunLength Spacing") && guiConfig->isLayerVisible(_z_order);
  }

  if (get_type() == GuiSpeedupItemType::kDrcNotchSpacing) {
    return tree_node.isChecked("Notch Spacing") && guiConfig->isLayerVisible(_z_order);
  }

  if (get_type() == GuiSpeedupItemType::kDrcMinStep) {
    return tree_node.isChecked("MinStep") && guiConfig->isLayerVisible(_z_order);
  }

  if (get_type() == GuiSpeedupItemType::kDrcMinArea) {
    return tree_node.isChecked("Minimum Area") && guiConfig->isLayerVisible(_z_order);
  }

  return false;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GuiSpeedupDrcList::init(QRectF boundingbox, int z_order) {
  qreal step_x  = boundingbox.width() / GUI_INSTANCE_GRID_COL;
  qreal step_y  = boundingbox.height() / GUI_INSTANCE_GRID_ROW;
  qreal x_start = boundingbox.left();
  qreal y_start = boundingbox.top();
  for (int y = 0; y < GUI_INSTANCE_GRID_COL + 1; ++y) {
    for (int x = 0; x < GUI_INSTANCE_GRID_ROW + 1; ++x) {
      GuiSpeedupDrc* item_new = addCurrentItem();
      item_new->set_z_order(z_order);
      item_new->set_bounding_rect(QRectF(x_start + x * step_x, y_start + y * step_y, step_x, step_y));
    }
  }

  set_number(GUI_INSTANCE_GRID_COL + 1, GUI_INSTANCE_GRID_ROW + 1);
  set_step(step_x, step_y);
}
