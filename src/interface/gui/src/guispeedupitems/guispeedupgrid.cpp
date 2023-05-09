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
#include "guispeedupgrid.h"

#include "guiConfig.h"
#include "guiConfigTree.h"

void GuiSpeedupGrid::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) {
  Q_UNUSED(widget);

  if (!is_visible()) {
    return;
  }

  const qreal lod = option->levelOfDetailFromTransform(painter->worldTransform());
  if (lod < 1) {
    paintScaleTop(painter, lod);
  } else if (lod >= 1 && lod < 10) {
    paintScale_1st(painter, lod);
  } else if (lod >= 10 && lod < 100) {
    paintScale_2nd(painter, lod);
  } else {
    paintScale_3rd(painter, lod);
  }

  GuiSpeedupItem::paint(painter, option, widget);
}

void GuiSpeedupGrid::paintScaleTop(QPainter* painter, qreal lod) {
  //   QPen pen = _pen;
  //   pen.setWidthF(pen.widthF() / lod);
  //   painter->setPen(pen);
  //   painter->setBrush(Qt::BrushStyle::NoBrush);

  //   for (auto points : get_points()) {
  //     painter->drawLine(points.first, points.second);
  //   }
}

void GuiSpeedupGrid::paintScale_1st(QPainter* painter, qreal lod) {
  //   QPen pen = _pen;
  //   pen.setWidthF(pen.widthF() / lod);
  //   painter->setPen(pen);

  //   painter->setBrush(Qt::BrushStyle::NoBrush);

  //   for (auto points : get_points()) {
  //     painter->drawLine(points.first, points.second);
  //   }
}

void GuiSpeedupGrid::paintScale_2nd(QPainter* painter, qreal lod) {
  QPen pen = _pen;
  //   pen.setWidthF(pen.widthF() / lod);
  pen.setWidthF(0);
  painter->setPen(pen);

  painter->setBrush(Qt::BrushStyle::NoBrush);

  for (auto points : get_points()) {
    painter->drawLine(points.first, points.second);
  }
}

void GuiSpeedupGrid::paintScale_3rd(QPainter* painter, qreal lod) {
  QPen pen = _pen;
  //   pen.setWidthF(pen.widthF() / lod);
  pen.setWidthF(0);
  painter->setPen(pen);

  QBrush brush = _brush;
  brush.setTransform(painter->worldTransform().inverted());
  painter->setBrush(brush);
  for (auto points : get_points()) {
    painter->drawLine(points.first, points.second);
  }
}

bool GuiSpeedupGrid::is_visible() {
  igui::GuiTreeNode& tree_node = guiConfig->get_trackgrid_tree();

  if (get_type() == GuiSpeedupItemType::kTrackGridPrefer) {
    return tree_node.isChecked("Prefer") && guiConfig->isLayerVisible((int32_t)zValue());
  }

  if (get_type() == GuiSpeedupItemType::kTrackGridNonPrefer) {
    return tree_node.isChecked("NonPrefer") && guiConfig->isLayerVisible((int32_t)zValue());
  }

  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GuiSpeedupGridList::init(QRectF boundingbox, IdbLayerDirection direction) {
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
      GuiSpeedupGrid* item_new = addCurrentItem();
      item_new->set_bounding_rect(QRectF(x_start + x * step_x, y_start + y * step_y, step_x, step_y));
    }
  }

  set_number(number_x, number_y);
  set_step(step_x, step_y);
}

void GuiSpeedupGridList::initPanel(QRectF boundingbox, IdbLayerDirection direction) {
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
      GuiSpeedupGrid* item_new = addCurrentItem();
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
      GuiSpeedupGrid* item_new = addCurrentItem();
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