#ifndef GUIPOLYGON_H
#define GUIPOLYGON_H

#include <QGraphicsPolygonItem>

#include "guiitem.h"
using namespace idb;

class GuiPolygonPrivate;
class GuiPolygon : public GuiItem {
 public:
  explicit GuiPolygon(QGraphicsItem* parent = nullptr);
  virtual ~GuiPolygon();

  QPolygonF get_polygon();
  inline void setPolygon(qreal x_indicator, qreal y_indicator, IdbConnectDirection direction = IdbConnectDirection::kNone);
  QRectF boundingRect() const override;
  QPainterPath shape() const override;

  void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

 protected:
  GuiPolygon(GuiPolygonPrivate* data, QGraphicsItem* parent);

 private:
  void setPointArea(const QPointF& point);
  GuiPolygonPrivate* _data;
};

class GuiPolygonPrivate : public GuiItemPrivate {
 public:
  explicit GuiPolygonPrivate();
  ~GuiPolygonPrivate();

 protected:
  //   QVector<QPolygonF> _iopins;
  QRectF _rect;
  QPointF _indicator;
  IdbConnectDirection _direction;
  QPolygonF _polygon;
  QLineF _line;
  friend class GuiPolygon;
};
#endif  // GUIPOLYGON_H
