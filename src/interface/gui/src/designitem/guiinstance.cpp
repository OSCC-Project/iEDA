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
#include "guiinstance.h"

#include <QPainter>
#include <QStyleOptionGraphicsItem>
GuiInstance::GuiInstance(QGraphicsItem* parent)
    : GuiRect(new GuiInstancePrivate(), parent),
      _data(static_cast<GuiInstancePrivate*>(_d_ptr))
{
}

GuiInstance::GuiInstance(GuiInstancePrivate* data, QGraphicsItem* parent)
    : GuiRect(data, parent), _data(static_cast<GuiInstancePrivate*>(_d_ptr))
{
}

void GuiInstance::add_flight_line(GuiFlightLine* flight_lines)
{
  _data->_flight_lines.push_back(flight_lines);
}

void GuiInstance::add_wire(GuiWire* wire)
{
  _data->_wires.push_back(wire);
}

void GuiInstance::add_pin(qreal x_pos, qreal y_pos, qreal width)
{
  if (width <= 0) {
    _data->_pins.push_back(QRectF(x_pos - 0.02, y_pos - 0.02, 0.02, 0.02));
  } else {
    _data->_pins.push_back(
        QRectF(x_pos - width / 2, y_pos - width / 2, width, width));
  }
}

void GuiInstance::paint(QPainter*                       painter,
                        const QStyleOptionGraphicsItem* option,
                        QWidget*                        widget)
{
  const qreal lod
      = option->levelOfDetailFromTransform(painter->worldTransform());

  if (lod > 1.5) {
    QBrush brush = QBrush(QColor(240, 210, 24));
    foreach (QRectF pin, _data->_pins) {
      painter->fillRect(pin, brush);
    }
  }
  GuiRect::paint(painter, option, widget);
}

void GuiInstance::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
  foreach (GuiFlightLine* flight_line, _data->_flight_lines) {
    flight_line->show();
  }
}

void GuiInstance::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event)
{
  foreach (GuiFlightLine* flight_line, _data->_flight_lines) {
    flight_line->show();
  }
  foreach (GuiWire* wire, _data->_wires) {
    wire->show();
  }
}
void GuiInstance::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
  foreach (GuiFlightLine* flight_line, _data->_flight_lines) {
    flight_line->hide();
  }
  foreach (GuiWire* wire, _data->_wires) {
    wire->hide();
  }
}

GuiInstancePrivate::GuiInstancePrivate()
{
}
GuiInstancePrivate::~GuiInstancePrivate()
{
}
