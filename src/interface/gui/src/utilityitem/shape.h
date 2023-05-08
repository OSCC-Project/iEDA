#ifndef SHAPE_H
#define SHAPE_H

#include <QGraphicsItem>
#include <QGraphicsSceneMouseEvent>

class Shape : public QGraphicsItem {
 public:
  enum Code {
    None,
    Line,
    Ruler,
  };

  Shape();

  virtual void startDraw(QGraphicsSceneMouseEvent* event) = 0;
  virtual void drawing(QGraphicsSceneMouseEvent* event)   = 0;
  QRectF boundingRect() const override;
  QPainterPath shape() const override;
  void virtual paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

 protected:
  QPointF start;
  QPointF end;
};
#endif  // SHAPE_H
