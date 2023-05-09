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
#include "guigr.h"

#include <QPainter>
#include <QStyleOptionGraphicsItem>

#include "guiattribute.h"

GuiGrRect::GuiGrRect(QGraphicsItem* parent)
    : GuiRect(new GuiGrPrivate(), parent), _data(static_cast<GuiGrPrivate*>(_d_ptr)) { }

GuiGrRect::GuiGrRect(GuiGrPrivate* data, QGraphicsItem* parent)
    : GuiRect(data, parent), _data(static_cast<GuiGrPrivate*>(_d_ptr)) { }

void GuiGrRect::set_layer(const std::string layer) {
  _data->set_layer(layer);
  set_pen(attributeInst->getLayerColor(layer));
  set_brush(attributeInst->getLayerColor(layer), Qt::BrushStyle::FDiagPattern);
}

void GuiGrRect::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) {
  if (!isAllowToShow()) {
    return;
  }

  const qreal lod = option->levelOfDetailFromTransform(painter->worldTransform());

  //   if (lod > 1.5) {
  //     QBrush brush = QBrush(QColor(240, 210, 24));
  //     foreach (QRectF pin, _data->_pins) {
  //       painter->fillRect(pin, brush);
  //     }
  //   }

  GuiRect::paint(painter, option, widget);

  QFont font("Arial", 1, QFont::Thin, true);
  painter->setFont(font);

  QPen& pen = _data->get_pen();
  //   pen.setWidthF(1);
  pen.setColor(Qt::white);
  painter->setPen(pen);
  painter->drawText(_data->get_rect().topLeft(), QString(_data->get_item_info().c_str()));

  //   pen.setWidthF(0.01);
  //   painter->setPen(pen);

  std::vector<std::string>& info_list = _data->get_info_list();
  int i                               = 4;
  QPointF point                       = _data->get_rect().topLeft();
  for (std::string info : info_list) {
    painter->drawText(point.x(), point.y() + i, QString(info.c_str()));
    i++;
  }
}

inline bool GuiGrRect::isAllowToShow() const { return true; }

GuiGrPrivate::GuiGrPrivate() { }
GuiGrPrivate::~GuiGrPrivate() { }
