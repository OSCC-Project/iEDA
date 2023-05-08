/**
 * @file guipower.h
 * @author Fuxing Huang (yellowstar4869@foxmail.com)
 * @brief This class inherits from GuiRect to represent power in the GUI
 * @version 0.2
 * @date 2021-07-19(V0.1), 2021-08-13(V0.2)
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef GUIPIN_H
#define GUIPIN_H
#include "guipolygon.h"
#include "guirect.h"

class GuiPinPrivate;

class GuiPin : public GuiPolygon {
 public:
  explicit GuiPin(QGraphicsItem* parent = nullptr);
  ~GuiPin();

  /// QT Function
  void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;
  QRectF boundingRect() const override;

  /// getter
  QPointF get_pos() const;

  /// setter
  inline void set_pos(const QPointF& pos) { set_pos(pos.x(), pos.y()); }
  void set_pos(qreal x_pos, qreal y_pos);

  void set_IOPin(qreal x_indicator, qreal y_indicator, qreal pin_width, IdbConnectDirection direction,
                 RectEdgePosition pin_position);

  void setTopPin(qreal x_indicator, qreal y_indicator, qreal pin_width, IdbConnectDirection direction);
  void setBottomPin(qreal x_indicator, qreal y_indicator, qreal pin_width, IdbConnectDirection direction);
  void setLeftPin(qreal x_indicator, qreal y_indicator, qreal pin_width, IdbConnectDirection direction);
  void setRightPin(qreal x_indicator, qreal y_indicator, qreal pin_width, IdbConnectDirection direction);

  void set_lod(qreal lod) { _lod = lod; }

  /// operator

 protected:
  inline bool isAllowToShow() const override;
  inline bool isAllowToSelect() const override;

 private:
  GuiPinPrivate* _data;
  //   static const int TRIANGLE_HALF_WIDTH = 10;
  //   static const int TRIANGLE_HIGH       = 40;
  qreal _lod;
};

class GuiPinPrivate : public GuiPolygonPrivate {
 public:
  explicit GuiPinPrivate() { }
  ~GuiPinPrivate() = default;

 private:
  QPointF _pos;

  friend class GuiPin;
};

#endif  // GUIPIN_H
