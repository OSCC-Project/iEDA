// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file guiround.h
 * @author Wang Jun (wen8365@gmail.com)
 * @brief Gui round item base class
 * @version 0.1
 * @date 2021-07-02
 * 
 *
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
