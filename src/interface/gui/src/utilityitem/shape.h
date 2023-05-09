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
