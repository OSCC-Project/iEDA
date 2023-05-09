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
#include "guipolygon.h"

#include <QDebug>
#include <QKeyEvent>
#include <QPainter>
#include <QStyleOptionGraphicsItem>
GuiPolygon::GuiPolygon(QGraphicsItem* parent)
    : GuiItem(new GuiPolygonPrivate(), parent)
{
  _data = static_cast<GuiPolygonPrivate*>(_d_ptr);
}

GuiPolygon::GuiPolygon(GuiPolygonPrivate* data, QGraphicsItem* parent)
    : GuiItem(data, parent)
{
  _data = static_cast<GuiPolygonPrivate*>(_d_ptr);
}
GuiPolygon::~GuiPolygon()
{
}

QPolygonF GuiPolygon::get_polygon()
{
  return _data->_polygon;
}

QRectF GuiPolygon::boundingRect() const
{
  return isAllowToShow() ? _data->_rect : QRectF();
}
QPainterPath GuiPolygon::shape() const
{
  QPainterPath path;
  path.addPolygon(_data->_polygon);
  return path;
}

void GuiPolygon::paint(QPainter*                       painter,
                       const QStyleOptionGraphicsItem* option,
                       QWidget*                        widget)
{
  Q_UNUSED(widget);
  if (_data->_polygon.empty()) {
    return;
  }

  const qreal lod
      = option->levelOfDetailFromTransform(painter->worldTransform());
  painter->setRenderHint(QPainter::Antialiasing, true);
  QPen pen = _data->_pen;
  pen.setWidthF(pen.widthF() / lod);
  painter->setPen(pen);

  painter->setBrush(_data->_brush);

  painter->drawPolygon(_data->_polygon, Qt::OddEvenFill);
}

GuiPolygonPrivate::GuiPolygonPrivate() : _polygon(QPolygonF())
{
}
GuiPolygonPrivate::~GuiPolygonPrivate()
{
}
