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
