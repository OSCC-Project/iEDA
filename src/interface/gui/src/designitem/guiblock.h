/**
 * @file guiblock.h
 * @author Fuxing Huang (yellowstar4869@foxmail.com)
 * @brief This class inherits from GuiOrientationRect to represent Block in the GUI
 * @version 0.2
 * @date 2021-07-09(V0.1) 2021-08-12(V0.2)
 *
 * @copyright Copyright (c) 2021 PCNL
 *
 */

#ifndef GUIBLOCK_H
#define GUIBLOCK_H
#include "guiinstance.h"
class GuiBlockPrivate;
class GuiBlock : public GuiInstance {
 public:
  explicit GuiBlock(QGraphicsItem *parent = nullptr);

  QPen get_halo_pen();
  void set_halo_pen(const QPen &halo_pen);
  void set_halo_pen(QColor color, qreal width = 0, Qt::PenStyle style = Qt::SolidLine);

  QBrush get_halo_brush();
  void set_halo_brush(const QBrush &halo_brush);
  void set_halo_brush(const QColor &color, Qt::BrushStyle style = Qt::SolidPattern);

  QRectF get_halo_rect();
  void set_halo_rect(const QRectF &halo_rect);
  inline void set_halo_rect(qreal x1, qreal y1, qreal x2, qreal y2);

  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;

 protected:
  inline bool isAllowToShow() const override;
  inline bool isAllowToSelect() const override;

 private:
  GuiBlockPrivate *_data;
};
inline void GuiBlock::set_halo_rect(qreal x1, qreal y1, qreal x2, qreal y2) {
  set_halo_rect(QRectF(QPointF(x1, y1), QPointF(x2, y2)));
}

class GuiBlockPrivate : public GuiInstancePrivate {
 public:
  explicit GuiBlockPrivate();
  ~GuiBlockPrivate();

 protected:
  QPen _halo_pen;
  QBrush _halo_brush;
  QRectF _halo_rect;
  friend class GuiBlock;
};

#endif  // GUIBLOCK_H
