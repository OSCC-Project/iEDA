/**
 * @file guiround.h
 * @author Wang Jun (wen8365@gmail.com)
 * @brief Gui round item base class
 * @version 0.1
 * @date 2021-07-02
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef GUIROUND_H
#define GUIROUND_H

#include <QGraphicsEllipseItem>

#include "guiitem.h"

class GuiRoundItem : public QObject, public QGraphicsEllipseItem {
  Q_OBJECT
 public:
  explicit GuiRoundItem(QGraphicsItem *parent = nullptr)
      : QGraphicsEllipseItem(parent){};
  virtual ~GuiRoundItem() = default;

  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
             QWidget *widget = nullptr) override {
    QGraphicsEllipseItem::paint(painter, option, widget);
  };
  QRectF boundingRect() const override { return rect(); };
  void setRound(qreal x, qreal y, qreal radius) {
    setRect(x, y, radius * 2, radius * 2);
  };
  Q_INVOKABLE void setBrush(const QBrush &brush) {
    QGraphicsEllipseItem::setBrush(brush);
  };
  Q_INVOKABLE void setPen(const QPen &pen) {
    QGraphicsEllipseItem::setPen(pen);
  };
};

class GuiRound : public GuiItem<GuiRoundItem> {
 public:
  explicit GuiRound(QGraphicsItem *parent = nullptr)
      : GuiItem<GuiRoundItem>(parent){};
  virtual ~GuiRound() = default;
  void setRound(qreal x, qreal y, qreal radius) {
    get_item()->setRound(x, y, radius);
  };
    
  void onCreated() override {
    makePen(Qt::white);
    makeBrush(Qt::gray, Qt::BrushStyle::SolidPattern);
  };
};

#endif  // GUIROUND_H
