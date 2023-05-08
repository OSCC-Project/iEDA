#ifndef ARROWDRC_H
#define ARROWDRC_H
#include <QGraphicsItem>
#include "guiitem.h"
#include "guirect.h"

class ArrowDrcPrivate;
class ArrowDrc : public GuiRect
{
 public:
  explicit ArrowDrc(QGraphicsItem *parent = nullptr);

  void set_drc(qreal x_pos,qreal y_pos);

  void paint(QPainter*                       painter,
             const QStyleOptionGraphicsItem* option,
             QWidget*                        widget) override;

  QRectF       boundingRect() const override;
  QPainterPath shape() const override;

 protected:
  ArrowDrc(ArrowDrcPrivate *data,QGraphicsItem *parent = nullptr);
  //  inline bool isAllowToShow() const override;
  //  inline bool isAllowToSelect() const override;

 private:
  ArrowDrcPrivate *_data;
};

class ArrowDrcPrivate : public GuiRectPrivate {
 public:
  explicit ArrowDrcPrivate();
  ~ArrowDrcPrivate();

 private:
  friend class ArrowDrc;
};
#endif // ARROWDRC_H
