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
#include "guispeedupclocktree.h"

QRectF GuiSpeedupClockTreeItem::boundingRect() const {
  return QRectF(_ll_x - 1, _ll_y - 1, _ur_x - _ll_x + 2, _ur_y - _ll_y + 2);
}

QPainterPath GuiSpeedupClockTreeItem::shape() const {
  QRectF rect = boundingRect();
  QPainterPath path;
  path.addRect(rect);
  return path;
}

bool GuiSpeedupClockTreeItem::is_visible() {
  igui::GuiTreeNode& tree_node = guiConfig->get_clock_tree();
  return tree_node.isChecked("Show");
}

void GuiSpeedupClockTreeItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) {
  Q_UNUSED(widget);

  if (!is_visible()) {
    return;
  }

  const qreal lod = option->levelOfDetailFromTransform(painter->worldTransform());

  if (lod < 1) {
    paintScaleTop(painter, lod);
  } else if (lod >= 1 && lod < 10) {
    paintScale_1st(painter, lod);
  } else if (lod >= 10 && lod < 50) {
    paintScale_2nd(painter, lod);
  } else {
    paintScale_3rd(painter, lod);
  }

  QGraphicsRectItem::paint(painter, option, widget);
}

void GuiSpeedupClockTreeItem::paintScaleTop(QPainter* painter, qreal lod) {
  QPen pen = _pen;
  pen.setColor(QColor(100, 100, 100));
  pen.setWidthF(pen.widthF() / lod);
  painter->setPen(pen);

  /// draw instance as rect with pin shape
  QBrush brush = _brush;
  brush.setTransform(painter->worldTransform().inverted());
  painter->setBrush(brush);

  qreal poly_width = _poly_width / lod;

  if (_type == GuiClockType::kRoot) {
    // #pragma omp parallel for
    for (auto& pt : _polygon_list) {
      QPolygonF polygon;
      polygon.push_back(QPointF(pt.x() - poly_width, pt.y() - poly_width));
      polygon.push_back(QPointF(pt.x() + poly_width, pt.y() - poly_width));
      polygon.push_back(QPointF(pt.x(), pt.y() + poly_width));
      painter->drawPolygon(polygon);
    }
  }

  // #pragma omp parallel for
  for (auto& pt_pair : _point_list) {
    painter->drawLine(pt_pair.first, pt_pair.second);
  }
}

void GuiSpeedupClockTreeItem::paintScale_1st(QPainter* painter, qreal lod) {
  QPen pen = _pen;
  pen.setColor(QColor(100, 100, 100));
  pen.setWidthF(pen.widthF() / lod);
  painter->setPen(pen);

  /// draw instance as rect with pin shape
  QBrush brush = _brush;
  brush.setTransform(painter->worldTransform().inverted());
  painter->setBrush(brush);

  qreal poly_width = _type == GuiClockType::kRoot ? _poly_width / lod : _poly_width * 0.5 / lod;

  // #pragma omp parallel for
  for (auto& pt : _polygon_list) {
    QPolygonF polygon;
    polygon.push_back(QPointF(pt.x() - poly_width, pt.y() - poly_width));
    polygon.push_back(QPointF(pt.x() + poly_width, pt.y() - poly_width));
    polygon.push_back(QPointF(pt.x(), pt.y() + poly_width));
    painter->drawPolygon(polygon);
  }

  // #pragma omp parallel for
  for (auto& pt_pair : _point_list) {
    painter->drawLine(pt_pair.first, pt_pair.second);
  }
}

void GuiSpeedupClockTreeItem::paintScale_2nd(QPainter* painter, qreal lod) {
  QPen pen = _pen;
  pen.setColor(QColor(100, 100, 100));
  pen.setWidthF(pen.widthF() / lod);
  painter->setPen(pen);

  /// draw instance as rect with pin shape
  QBrush brush = _brush;
  brush.setTransform(painter->worldTransform().inverted());
  painter->setBrush(brush);

  if (_type == GuiClockType::kLeaf) {
    qreal poly_width = _poly_width * 0.2 / lod;
    // #pragma omp parallel for
    for (auto& pt : _leaf_list) {
      painter->drawRect(QRectF(pt.x() - poly_width, pt.y() - poly_width, poly_width * 2, poly_width * 2));
    }
  } else {
    qreal poly_width = _type == GuiClockType::kRoot ? _poly_width / lod : _poly_width * 0.5 / lod;
    // #pragma omp parallel for
    for (auto& pt : _polygon_list) {
      QPolygonF polygon;
      polygon.push_back(QPointF(pt.x() - poly_width, pt.y() - poly_width));
      polygon.push_back(QPointF(pt.x() + poly_width, pt.y() - poly_width));
      polygon.push_back(QPointF(pt.x(), pt.y() + poly_width));
      painter->drawPolygon(polygon);
    }
  }

  // #pragma omp parallel for
  for (auto& pt_pair : _point_list) {
    painter->drawLine(pt_pair.first, pt_pair.second);
  }
}

void GuiSpeedupClockTreeItem::paintScale_3rd(QPainter* painter, qreal lod) {
  QPen pen = _pen;
  pen.setColor(QColor(100, 100, 100));
  pen.setWidthF(pen.widthF() / lod);
  painter->setPen(pen);

  /// draw instance as rect with pin shape
  QBrush brush = _brush;
  brush.setTransform(painter->worldTransform().inverted());
  painter->setBrush(brush);

  qreal poly_width = _poly_width / lod;

  if (_type == GuiClockType::kLeaf) {
    // #pragma omp parallel for
    for (auto& pt : _leaf_list) {
      painter->drawRect(QRectF(pt.x() - poly_width, pt.y() - poly_width, poly_width * 2, poly_width * 2));
    }
  } else {
    // #pragma omp parallel for
    for (auto& pt : _polygon_list) {
      QPolygonF polygon;
      polygon.push_back(QPointF(pt.x() - poly_width, pt.y() - poly_width));
      polygon.push_back(QPointF(pt.x() + poly_width, pt.y() - poly_width));
      polygon.push_back(QPointF(pt.x(), pt.y() + poly_width));
      painter->drawPolygon(polygon);
    }
  }

  // #pragma omp parallel for
  for (auto& pt_pair : _point_list) {
    painter->drawLine(pt_pair.first, pt_pair.second);
  }
}

void GuiSpeedupClockTreeItem::drawText(QPainter* painter, QRectF rect, QString str, const qreal lod) {
  return;
  QFont font("Arial", 1);
  painter->setFont(font);
  painter->save();
  QPen pen = painter->pen();
  pen.setWidthF(pen.widthF() / lod);
  pen.setColor(Qt::white);
  painter->setPen(pen);
  painter->scale(0.1, 0.1);
  painter->drawText(rect.center(), str);
  painter->restore();
}
