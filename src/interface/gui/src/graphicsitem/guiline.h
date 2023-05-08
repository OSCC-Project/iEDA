/**
 * @file guiline.h
 * @author Wang Jun (wen8365@gmail.com)
 *         Fuxing Huang (yellowstar4869@foxmail.com)
 * @brief Gui line item base class
 * @version 0.2
 * @date 2021-07-02(V0.1) 2021-08-10(V0.2)
 *
 *
 *
 */

#ifndef GUILINE_H
#define GUILINE_H
#include "guiitem.h"

class GuiLinePrivate;
class GuiLine : public GuiItem {
 public:
  explicit GuiLine(QGraphicsItem *parent = nullptr);
  ~GuiLine();

  QLineF get_line() const;
  void set_line(const QLineF &line);
  inline void set_line(qreal x1, qreal y1, qreal x2, qreal y2);

  QRectF boundingRect() const override;
  QPainterPath shape() const override;

  
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
             QWidget *widget = nullptr) override;

 protected:
  GuiLine(GuiLinePrivate *data, QGraphicsItem *parent);

 private:
  GuiLinePrivate *_data;
};
inline void GuiLine::set_line(qreal x1, qreal y1, qreal x2, qreal y2) {
  set_line(QLineF(x1, y1, x2, y2));
}
class GuiLinePrivate : public GuiItemPrivate {
 public:
  explicit GuiLinePrivate();
  ~GuiLinePrivate();

 protected:
  QLineF _line;
  friend class GuiLine;
};

#endif  // GUILINE_H
