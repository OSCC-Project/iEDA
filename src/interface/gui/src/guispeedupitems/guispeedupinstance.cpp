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
#include "guispeedupinstance.h"

#include "guiConfig.h"
#include "guiConfigTree.h"
#include "omp.h"

void GuiSpeedupInstance::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) {
  Q_UNUSED(widget);

  GuiSpeedupItem::paint(painter, option, widget);
}

void GuiSpeedupInstance::paintScaleTop(QPainter* painter, qreal lod) {
  QPen pen = _pen;
  //   pen.setWidthF(pen.widthF() / lod);
  pen.setWidthF(0);
  painter->setPen(pen);

  //   int step        = 1;
  //   auto& rect_list = get_rect_list();
  //   if (rect_list.size() >= 100 && rect_list.size() < 500) {
  //     step = 2;
  //   } else if (rect_list.size() >= 500 && rect_list.size() < 1000) {
  //     step = 5;
  //   } else if (rect_list.size() >= 1000 && rect_list.size() < 2000) {
  //     step = 10;
  //   } else if (rect_list.size() >= 2000 && rect_list.size() < 5000) {
  //     step = 20;
  //   } else if (rect_list.size() >= 5000 && rect_list.size() < 10000) {
  //     step = 100;
  //   } else {
  //     step = 1;
  //   }

  //   for (int i = 0; i < rect_list.size(); i += step) {
  //     painter->drawPoint(rect_list[i].center());
  //   }

  for (auto& rect : get_rect_list()) {
    painter->drawPoint(rect.center());
  }
}

void GuiSpeedupInstance::paintScale_1st(QPainter* painter, qreal lod) {
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

void GuiSpeedupInstance::paintScale_2nd(QPainter* painter, qreal lod) {
  QPen pen = _pen;
  pen.setWidthF(pen.widthF() / lod);
  pen.setWidthF(0);
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

void GuiSpeedupInstance::paintScale_3rd(QPainter* painter, qreal lod) {
  QPen pen = _pen;
  pen.setWidthF(pen.widthF() / lod);
  pen.setWidthF(0);
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

  /// draw pin shape
  // #pragma omp parallel for
  for (auto& pin : _pin_list) {
    painter->fillRect(pin, QBrush(QColor(240, 210, 24)));
  }
}

bool GuiSpeedupInstance::is_visible() {
  igui::GuiTreeNode& tree_node = guiConfig->get_instance_tree();

  if (get_type() == GuiSpeedupItemType::kInstStandarCell) {
    return tree_node.isChecked("Standard Cell");
  }

  if (get_type() == GuiSpeedupItemType::kInstIoCell) {
    return tree_node.isChecked("IO Cell");
  }

  if (get_type() == GuiSpeedupItemType::kInstBlock) {
    return tree_node.isChecked("Block");
  }

  if (get_type() == GuiSpeedupItemType::kInstPad) {
    return tree_node.isChecked("Pad");
  }

  return false;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GuiSpeedupInstanceList::init(QRectF boundingbox) {
  qreal step_x  = boundingbox.width() / GUI_INSTANCE_GRID_COL;
  qreal step_y  = boundingbox.height() / GUI_INSTANCE_GRID_ROW;
  qreal x_start = boundingbox.left();
  qreal y_start = boundingbox.top();
  for (int y = 0; y < GUI_INSTANCE_GRID_COL + 1; ++y) {
    for (int x = 0; x < GUI_INSTANCE_GRID_ROW + 1; ++x) {
      GuiSpeedupInstance* item_new = addCurrentItem();
      item_new->set_bounding_rect(QRectF(x_start + x * step_x, y_start + y * step_y, step_x, step_y));
    }
  }

  set_number(GUI_INSTANCE_GRID_COL + 1, GUI_INSTANCE_GRID_ROW + 1);
  set_step(step_x, step_y);
}
