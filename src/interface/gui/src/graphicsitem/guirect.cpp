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
#include "guirect.h"

#include <QDebug>
#include <QKeyEvent>
#include <QPainter>
#include <QStyleOptionGraphicsItem>
GuiRect::GuiRect(QGraphicsItem* parent) : GuiItem(new GuiRectPrivate(), parent)
{
  _data = static_cast<GuiRectPrivate*>(_d_ptr);
}

GuiRect::GuiRect(GuiRectPrivate* data, QGraphicsItem* parent)
    : GuiItem(data, parent)
{
  _data = static_cast<GuiRectPrivate*>(_d_ptr);
}
GuiRect::~GuiRect()
{
}

QRectF GuiRect::get_rect()
{
  return _data->_rect;
}

void GuiRect::set_rect(const QRectF& rect, IdbOrient orientation)
{
  _data->_rect        = rect;
  _data->_orientation = orientation;
  DrawOrientationLine();
//   update();
}
// void GuiRect::set_orientation(IdbOrient orientation) {
//    int old = _data->_orientation;
//    int cur = orientation;
//    if (old / 4 != cur / 4) {
//        qreal width = _data->_rect.height();
//        _data->_rect.setHeight(_data->_rect.width());
//        _data->_rect.setWidth(width);
//    }
//    _data->_orientation = orientation;
//    if (old % 4 != cur % 4) {
//        DrawOrientationLine();
//    }
//    update();
//}

QRectF GuiRect::boundingRect() const
{
  return isAllowToShow() ? _data->_rect : QRectF();
}

QPainterPath GuiRect::shape() const
{
  QPainterPath path;
  path.addRect(boundingRect());
  return path;
}

void GuiRect::paint(QPainter*                       painter,
                    const QStyleOptionGraphicsItem* option,
                    QWidget*                        widget)
{
  Q_UNUSED(widget);

  if (_data->_rect.isNull())
    return;

  const qreal lod
      = option->levelOfDetailFromTransform(painter->worldTransform());

  QPen pen = _data->_pen;
  pen.setWidthF(pen.widthF() / lod);
  painter->setPen(pen);

  if (lod < 10) {
    painter->drawLine(_data->_rect.topLeft(), _data->_rect.topRight());
    painter->drawLine(_data->_rect.bottomLeft(), _data->_rect.bottomRight());
    painter->drawLine(_data->_rect.topLeft(), _data->_rect.bottomLeft());
    painter->drawLine(_data->_rect.topRight(), _data->_rect.bottomRight());
    return;
  }

  QBrush brush = _data->_brush;
  brush.setTransform(painter->worldTransform().inverted());
  painter->setBrush(brush);

  if (lod < 0.4) {
    painter->drawRect(_data->_rect);
    return;
  }

  if (option->state & QStyle::State_Selected) {
    pen.setColor(pen.color().lighter());
  }
  painter->setPen(pen);
  
  painter->drawRect(_data->_rect);
  if (!_data->_line.isNull() && (_data->_orientation != IdbOrient::kNone)) {
    painter->drawLine(_data->_line);
  }
}

void GuiRect::DrawOrientationLine()
{
  qreal   width  = _data->_rect.width();
  qreal   height = _data->_rect.height();
  QPointF p1;
  QPointF p2;
  switch (_data->_orientation) {
    case IdbOrient::kN_R0:
    case IdbOrient::kFW_MX90: {
      QPointF bottom_left = _data->_rect.bottomLeft();
      p1                  = bottom_left + QPointF(0, -(height * 0.25));
      p2                  = bottom_left + QPointF(width * 0.25, 0);
      _data->_line        = QLineF(p1, p2);
      break;
    }
    case IdbOrient::kW_R90:
    case IdbOrient::kFN_MY: {
      QPointF bottom_right = _data->_rect.bottomRight();
      p1                   = bottom_right + QPointF(0, -(height * 0.25));
      p2                   = bottom_right + QPointF(-(width * 0.25), 0);
      _data->_line         = QLineF(p1, p2);
      break;
    }
    case IdbOrient::kE_R270:
    case IdbOrient::kFS_MX: {
      QPointF top_left = _data->_rect.topLeft();
      p1               = top_left + QPointF(0, (height * 0.25));
      p2               = top_left + QPointF(width * 0.25, 0);
      _data->_line     = QLineF(p1, p2);
      break;
    }
    case IdbOrient::kS_R180:
    case IdbOrient::kFE_MY90: {
      QPointF top_right = _data->_rect.topRight();
      p1                = top_right + QPointF(0, (height * 0.25));
      p2                = top_right + QPointF(-(width * 0.25), 0);
      _data->_line      = QLineF(p1, p2);
      break;
    }
    default:
      break;
  }
}

GuiRectPrivate::GuiRectPrivate() : _rect(QRectF())
{
  _item_info = "";
}
GuiRectPrivate::~GuiRectPrivate()
{
}
