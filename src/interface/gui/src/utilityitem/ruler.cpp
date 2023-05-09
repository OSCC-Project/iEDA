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
#include "ruler.h"
#include "math.h"

#include <QDebug>
#include <QEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QStyleOptionGraphicsItem>

Ruler::Ruler(QGraphicsItem *parent)
{
}
Ruler::~Ruler()
{
}

void Ruler::startDraw(QGraphicsSceneMouseEvent *event)
{
  qreal w = end.x() - start.x();
  QPointF turning  = QPointF(start.x() + w,start.y());
  line1 = QLineF(start,turning);
  line2 = QLineF(turning,end);
}

void Ruler::drawing(QGraphicsSceneMouseEvent *event)
{
  //QPointF pos = event->scenePos();
  set_end(event->scenePos());
  qreal w = end.x() - start.x();
  QPointF turning  = QPointF(start.x() + w,start.y());
  line1 = QLineF(start,turning);
  line2 = QLineF(turning,end);
  //setLine(newline);
}

void Ruler::paint(QPainter*                       painter,
                  const QStyleOptionGraphicsItem* option,
                  QWidget*                        widget)
{
  Q_UNUSED(widget);

  const qreal _lod =
      option->levelOfDetailFromTransform(painter->worldTransform());

  QFont font(QStringLiteral("Microsoft YaHei"),9);
  if(_lod < 0.7)  { r_scale = 10; }
  if(_lod > 0.7 && _lod < 1.5){ r_scale = 5; }
  if(_lod >= 1.5){ r_scale = 3; }

  painter->setFont(font);
  QPen pen(Qt::yellow);
  pen.setWidth(1);
  painter->setPen(pen);
  painter->translate(0,0);
  painter->setRenderHint(QPainter::Antialiasing, true);


  qreal w = end.x() - start.x();
  //qreal h = end.y() - start.y();
  turning  = QPointF(start.x() + w,start.y());

  QString tText = QString::number(qAbs(turning.x() - start.x()));
  QString eTest = QString::number(qAbs(turning.y() - end.y()));
  painter->drawText(turning,tText);
  painter->drawText(end,eTest);
  drawHorizontalScale(painter,&pen,start,turning);
  drawVerticalScale(painter,&pen,turning,end);
}

void Ruler::drawHorizontalScale(QPainter *painter, QPen *pen, QPointF _pos1, QPointF _pos2)
{

  painter->drawLine(_pos1,_pos2) ;
  float fDrawLeft = _pos1.x();
  int nIndex = 0;
  int nScaleHeight = 20;

  if(_pos1.x()<_pos2.x()){
    while(fDrawLeft < _pos2.x()){
      if((nIndex % 2 == 0) && (nIndex % 5 == 0)){
        nScaleHeight = 10;
        //QString sText = QString::number(nIndex / 10) ;
        // painter->drawText(QPoint(fDrawLeft,_pos1.y() + nScaleHeight + 15),sText);
      }
      else
      {
        pen->setWidth(1);
        painter->setPen(*pen);
        nScaleHeight = (nIndex % 5) == 0 ? 10 : 6;
      }
      painter->drawRect(QRectF(fDrawLeft,_pos1.y(),0.1,nScaleHeight));

      fDrawLeft += r_scale;
      nIndex += 1;
    }
  }
  if(_pos1.x() > _pos2.x()){
    while(fDrawLeft > _pos2.x()){
      if((nIndex % 2 == 0) && (nIndex % 5 == 0)){
        nScaleHeight = 10;
        //QString sText = QString::number(nIndex / 10) ;
        //painter->drawText(QPoint(fDrawLeft,_pos1.y() + nScaleHeight + 15),sText);
      }
      else
      {
        pen->setWidth(1);
        painter->setPen(*pen);
        nScaleHeight = (nIndex % 5) == 0 ? 10 : 6;
      }
      painter->drawRect(QRectF(fDrawLeft,_pos1.y(),0.1,nScaleHeight));
      fDrawLeft -= r_scale;
      nIndex +=1;
    }
  }
}
void Ruler::drawVerticalScale(QPainter *painter, QPen *pen, QPointF _pos1, QPointF _pos2)
{
  painter->drawLine(_pos1,_pos2);
  float fDrawTop = _pos1.y();
  int nIndex = 0;
  int nScaleWidth = 20;

  if(_pos1.y() < _pos2.y()){
    while(fDrawTop <= _pos2.y()){
      if((nIndex % 2 == 0) && (nIndex % 5 == 0)){
        nScaleWidth = 10;
        //QString sText = QString::number(nIndex / 10);
        //painter->drawText(QPoint(_pos1.x() - nScaleWidth -10,fDrawTop),sText);
      }
      else
      {
        pen->setWidth(1);
        painter->setPen(*pen);
        nScaleWidth = (nIndex % 5) == 0 ? 10 : 6;
      }
      //painter->drawRect(QRectF(_pos1.x() - nScaleWidth,fDrawTop,nScaleWidth,0.1));
      painter->drawRect(QRectF(_pos1.x(),fDrawTop,-nScaleWidth,0.1));

      fDrawTop += r_scale;
      nIndex += 1;
    }
  }

  if(_pos1.y() > _pos2.y()){
    while(fDrawTop >= _pos2.y()){
      if((nIndex % 2 == 0) && (nIndex % 5 == 0)){
        nScaleWidth = 10;
        //QString sText = QString::number(nIndex / 10);
        //painter->drawText(QPoint(_pos1.x() - nScaleWidth -10,fDrawTop),sText);
      }
      else{
        pen->setWidth(1);
        painter->setPen(*pen);
        nScaleWidth = (nIndex % 5) == 0 ? 10 : 6;
      }
      painter->drawRect(QRectF(_pos1.x(),fDrawTop,-nScaleWidth,0.1));
      fDrawTop -= r_scale;
      nIndex += 1;
    }
  }
}

