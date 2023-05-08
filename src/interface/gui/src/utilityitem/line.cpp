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
