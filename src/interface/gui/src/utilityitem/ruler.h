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
#ifndef RULER_H
#define RULER_H

#include <QtGui>
#include "shape.h"

class Ruler : public Shape{
 public:
  explicit Ruler(QGraphicsItem *parent = nullptr);
  ~Ruler();

  //getter
  QPointF get_start_point() { return start; }
  QPointF get_end_point() { return end; }

  //setter
  void set_start(QPointF s) { start = s; }
  void set_end(QPointF e) { end = e; }

 protected:
  //Draw a horizontal scale
  void drawHorizontalScale(QPainter *painter,QPen *pen,QPointF _pos1,QPointF _pos2);
  //Draw a vertical scale
  void drawVerticalScale(QPainter *painter,QPen *pen,QPointF _pos1,QPointF _pos2);

  void startDraw(QGraphicsSceneMouseEvent *  event) override;
  void drawing(QGraphicsSceneMouseEvent* event) override;
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
             QWidget *widget = nullptr) override;

 private:
  QPointF start;
  QPointF end;
  QPointF turning;
  float r_scale = 5.0;
  QLineF line1;
  QLineF line2;
};

#endif // RULER_H
