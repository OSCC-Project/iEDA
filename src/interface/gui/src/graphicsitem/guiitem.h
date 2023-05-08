/**
 * @file guiitem.h
 * @author Wang Jun (wen8365@gmail.com)
 *         Fuxing Huang (yellowstar4869@foxmail.com)
 * @brief The GuiItem is the base class for all graphics item.
 *        The GraphicsItemFactory means that we create graphics item using
 *        the factory pattern.
 * @version 0.2
 * @date 2021-07-02(V0.1), 2021-08-01(V0.2)
 *
 *
 *
 */

#ifndef GUIITEM_H
#define GUIITEM_H
#include <QBrush>
#include <QGraphicsItem>
#include <QPen>

#include "IdbEnum.h"
#include "IdbGeometry.h"

class GuiItemPrivate;
class GuiItem : public QGraphicsItem {
 public:
  /**
   * @brief The following enums are in order. We use the enum value(eg. R0 =
   *        0,we need to use its int value 0) when we rotate the item.
   *        So, do not disturb it.
   *
   */
  //  enum Orientation {
  //    NoOrientation = -1,
  //    R0,
  //    MY,
  //    MX,
  //    R180,
  //    MX90,
  //    M90,
  //    R270,
  //    MY90
  //  };
  /*
   * default NoOrientation:
   *  _____
   * |  ↑  |
   * |width|← height
   * |     |
   * |_____|
   * --------------------------------------------------
   * R0: ____  MY: ____  MX: ____  R180: ____
   *    |    |    |    |    |/   |      |   \|
   *    |    |    |    |    |    |      |    |
   *    |    |    |    |    |    |      |    |
   *    |\___|    |___/|    |____|      |____|
   * ---------------------------------------------------
   * MX90: ________ M90: ________ R270: ________ MY90: ________
   *      |        |    |        |     |/       |     |       \|
   *      |\_______|    |_______/|     |________|     |________|
   */
  explicit GuiItem(QGraphicsItem* parent = nullptr);
  virtual ~GuiItem();

  QPen get_pen() const;
  void set_pen(const QPen& pen);
  void set_pen(QColor color, qreal width = 0, Qt::PenStyle style = Qt::SolidLine);

  QBrush get_brush() const;
  void set_brush(const QBrush& brush);
  void set_brush(const QColor& color, Qt::BrushStyle style = Qt::SolidPattern);

  bool isObscuredBy(const QGraphicsItem* item) const override;

 protected:
  GuiItemPrivate* _d_ptr;
  GuiItem(GuiItemPrivate* data, QGraphicsItem* parent);
  virtual inline bool isAllowToShow() const { return true; }
  virtual inline bool isAllowToSelect() const { return true; }

 private:
  GuiItemPrivate* _data;
};

class GuiItemPrivate {
 public:
  GuiItemPrivate() : _pen(QPen()), _brush(QBrush()) { }
  virtual ~GuiItemPrivate() { }

  QPen& get_pen() { return _pen; }
  QBrush& get_brush() { return _brush; }

 protected:
  QPen _pen;
  QBrush _brush;
  friend class GuiItem;
};

#endif  // GUIITEM_H
