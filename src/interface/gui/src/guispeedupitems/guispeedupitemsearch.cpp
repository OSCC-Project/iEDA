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
#include "guispeedupitemsearch.h"

QRectF GuiSpeedupItemSearch::boundingRect() const {
  qreal ll_x = INT32_MAX;
  qreal ll_y = INT32_MAX;
  qreal ur_x = 0;
  qreal ur_y = 0;

  for (auto& rect : _rect_list) {
    ll_x = std::min(ll_x, rect.left());
    ll_y = std::min(ll_y, rect.top());
    ur_x = std::max(ur_x, rect.right());
    ur_y = std::max(ur_y, rect.bottom());
  }

  for (auto& point : _pin_list) {
    ll_x = std::min(ll_x, point.x());
    ll_y = std::min(ll_y, point.y());
    ur_x = std::max(ur_x, point.x());
    ur_y = std::max(ur_y, point.y());
  }

  return QRectF(ll_x - 1, ll_y - 1, ur_x - ll_x + 2, ur_y - ll_y + 2);
}

QPainterPath GuiSpeedupItemSearch::shape() const {
  QPainterPath path;
  path.addRect(_bounding_box);
  return path;
}

void GuiSpeedupItemSearch::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) {
  Q_UNUSED(widget);

  const qreal lod = option->levelOfDetailFromTransform(painter->worldTransform());

  QPen pen = _pen;
  pen.setColor(QColor(255, 0, 0));
  pen.setWidthF(pen.widthF() * 2 / lod);
  painter->setPen(pen);

  /// draw pin coordinate
  for (auto& point : _pin_list) {
    painter->drawLine(_average_pin_coord, point);
  }

  /// draw pin shape
  auto& pin_rect_list = get_pin_rect_list();
  for (size_t i = 0; i < pin_rect_list.size(); i++) {
    painter->drawRect(pin_rect_list[i]);
  }

  /// draw segment
  pen.setColor(QColor(255, 255, 255));
  painter->setPen(pen);

  painter->setBrush(Qt::BrushStyle::NoBrush);
  auto& rect_list = get_rect_list();
  for (size_t i = 0; i < rect_list.size(); i++) {
    painter->drawRect(rect_list[i]);
  }

  QGraphicsRectItem::paint(painter, option, widget);
}
