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
#include "arrowdrc.h"

#include <QStyleOptionGraphicsItem>
#include <QPainter>
#include <QMouseEvent>
#include <QDebug>
#include <QToolTip>

ArrowDrc::ArrowDrc(QGraphicsItem *parent )
    :GuiRect(new ArrowDrcPrivate(),parent),
      _data(static_cast<ArrowDrcPrivate *>(_d_ptr)){
  set_pen(QColor(Qt::white));
  set_brush(QColor(Qt::white));

  this->setAcceptHoverEvents(Qt::LeftButton);
  setFlag(QGraphicsItem::ItemIsSelectable);

}
ArrowDrc::ArrowDrc(ArrowDrcPrivate *data, QGraphicsItem *parent)
    : GuiRect(data, parent), _data(static_cast<ArrowDrcPrivate *>(_d_ptr)){}

void ArrowDrc::set_drc(qreal x_pos, qreal y_pos){
  _data->_pos = QPointF(x_pos,y_pos);
  _data->_arrow[0].setLine(x_pos-10.0,y_pos-10.0,x_pos+10.0,y_pos+10.0);
  _data->_arrow[1].setLine(x_pos+10.0,y_pos-10.0,x_pos-10.0,y_pos+10.0);
  update();
}
QRectF ArrowDrc::boundingRect() const{
  const qreal x = _data->_pos.x() - 10.0;
  const qreal y = _data->_pos.y() - 10.0;
  return QRectF(x,y,20.0,20.0);
}

QPainterPath ArrowDrc::shape() const{
  QPainterPath path;
  path.addRect(boundingRect());
  return path;
}
void ArrowDrc::paint(QPainter*                       painter,
                     const QStyleOptionGraphicsItem* option,
                     QWidget*                        widget)
{
  Q_UNUSED(widget);

  if (isAllowToShow()) {
    if(_data->_arrow[2].isNull()) return;
    const qreal lod =
        option->levelOfDetailFromTransform(painter->worldTransform());
    painter->setRenderHint(QPainter::Antialiasing,true);

    QPen pen = _data->_pen;
    pen.setWidthF(pen.widthF() / lod);
    pen.setWidthF(5);
    painter->setPen(pen);

    QBrush brush = _data->_brush;
    brush.setTransform(QTransform(painter->worldTransform().inverted()));
    painter->setBrush(brush);

    painter->drawLines(_data->_arrow,2);

    GuiRect::paint(painter,option,widget);
    setToolTip("drc show");
  }
  if (isAllowToSelect()) {
    setFlag(QGraphicsItem::ItemIsSelectable, true);
  }
  else setFlag(QGraphicsItem::ItemIsSelectable, false);

}

//inline bool ArrowDrc::isAllowToShow() const{return true;}
//inline bool ArrowDrc::isAllowToSelect() const{return true;}
ArrowDrcPrivate::ArrowDrcPrivate(){}
ArrowDrcPrivate::~ArrowDrcPrivate(){}
