#ifndef LINE_H
#define LINE_H

#include <QGraphicsSceneMouseEvent>
#include <QGraphicsLineItem>
#include <QPainter>
#include "shape.h"
#include "guiitem.h"
#include "transform.h"

class Line : public Shape
{
 public:
  explicit Line(QGraphicsItem *parent = nullptr);
  ~Line();

  //getter
  QPointF get_start_point()
  {
    //return _tranform->guidb_calculate_real_coordinate(start);
    return start;
  }
  QPointF get_end_point()
  {
    //return _tranform->guidb_calculate_real_coordinate(end);
    return end;
  }

  //setter
  void set_pen(qreal width) {
    pen.setWidthF(width);
    pen.setColor(Qt::blue);
    update();
  }
  void set_start(QPointF s) { start = s; }
  void set_end(QPointF e) { end = e; }


  void startDraw(QGraphicsSceneMouseEvent *  event) override;
  void drawing(QGraphicsSceneMouseEvent *  event) override;
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
             QWidget *widget = nullptr) override;



 protected:
  Transform* _tranform = nullptr;
  QPointF start;
  QPointF end;
  QPointF moving;
  QPen pen;
  QLineF newLine;
};

#endif // LINE_H

