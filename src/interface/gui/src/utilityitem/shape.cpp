#include "shape.h"

Shape::Shape()
{
}

QRectF Shape::boundingRect() const{
  qreal w = end.x() - start.x();
  qreal h = end.y() - start.y();
  return QRectF(start.x(),start.y(),w,h);
}
QPainterPath Shape::shape() const{
  QPainterPath path;
  path.addRect(boundingRect());
  return path;
}
void Shape::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
                  QWidget *widget){
  Q_UNUSED(widget);

}

