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
#include "line.h"
#include <QDebug>
#include <QPainter>
#include <QStyleOptionGraphicsItem>
Line::Line(QGraphicsItem *parent)
{
}
Line::~Line(){}

void Line::startDraw(QGraphicsSceneMouseEvent *  event)
{
  newLine = QLineF( event->scenePos(),  event->scenePos());
}

void Line::drawing(QGraphicsSceneMouseEvent *  event)
{
  QPointF e = event->scenePos();
  qreal w = qAbs(e.x()) - qAbs(start.x());
  qreal h = qAbs(e.y()) - qAbs(start.y());
  if(qAbs(w) > qAbs(h)){
    moving = QPointF(e.x(),start.y());
  }
  else{
    moving = QPointF(start.x(),e.y());
  }
  set_end(moving);
  newLine = QLineF(start,moving);
}

void Line::paint(QPainter *painter,const QStyleOptionGraphicsItem *option,
                 QWidget *widget){
  Q_UNUSED(widget);
  painter->setRenderHint(QPainter::Antialiasing, true);

  set_pen(2);
  painter->setPen(pen);
  painter->drawLine(newLine);
}
