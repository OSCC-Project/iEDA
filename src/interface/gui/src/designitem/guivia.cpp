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
#include "guivia.h"

#include <QPainter>

#include "guiattribute.h"

GuiVia::GuiVia(QGraphicsItem* parent) : GuiRect(new GuiViaPrivate(), parent), _data(static_cast<GuiViaPrivate*>(_d_ptr)) { }
void GuiVia::set_layer(const std::string& layer) {
  _data->_layer = layer;
  //   _data->_layer.resize(4);
  set_pen(attributeInst->getLayerColor(_data->_layer));
  set_brush(attributeInst->getLayerColor(_data->_layer), Qt::BrushStyle::DiagCrossPattern);
}
void GuiVia::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) {
  Q_UNUSED(widget);
  if (isAllowToShow()) {
    painter->setPen(_data->_pen);
    painter->drawLine(_data->_rect.bottomLeft(), _data->_rect.topRight());
    painter->drawLine(_data->_rect.bottomRight(), _data->_rect.topLeft());
    GuiRect::paint(painter, option, widget);
  }
}

inline bool GuiVia::isAllowToShow() const { return true; }
inline bool GuiVia::isAllowToSelect() const { return true; }
