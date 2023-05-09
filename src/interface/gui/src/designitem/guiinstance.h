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
 * @file GuiInstance.h
 * @author Fuxing Huang (yellowstar4869@foxmail.com)
 * @brief This class inherits from GuiRect to represent GuiInstance in the GUI.
 *        Inherited by: GuiStandardCell, GuiBlock, GuiPad.
 * @version 0.1
 * @date 2021-08-16(V0.1)
 */
#ifndef GUIINSTANCE_H
#define GUIINSTANCE_H

#include "guiflightline.h"
#include "guirect.h"
#include "guiwire.h"

class GuiInstancePrivate;
class GuiInstance : public GuiRect {
 public:
  explicit GuiInstance(QGraphicsItem* parent = nullptr);

  void add_flight_line(GuiFlightLine* flight_lines);
  void add_wire(GuiWire* wire);
  void add_pin(qreal x_pos, qreal y_pos, qreal width = -1);

  void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

 protected:
  GuiInstance(GuiInstancePrivate* data, QGraphicsItem* parent = nullptr);
  void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
  void mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event) override;
  void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;

 private:
  GuiInstancePrivate* _data;
};
class GuiInstancePrivate : public GuiRectPrivate {
 public:
  explicit GuiInstancePrivate();
  ~GuiInstancePrivate();

 private:
  QVector<QRectF> _pins;
  QVector<GuiFlightLine*> _flight_lines;
  QVector<GuiWire*> _wires;
  friend class GuiInstance;
};

#endif  // GUIINSTANCE_H
