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
#include "guipad.h"
GuiPad::GuiPad(QGraphicsItem* parent)
    : GuiInstance(new GuiPadPrivate(), parent), _data(static_cast<GuiPadPrivate*>(_d_ptr)) {
  set_pen(QColor(130, 130, 130), 2);
  set_brush(QColor(140, 140, 140), Qt::BrushStyle::Dense7Pattern);
}
void GuiPad::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) {
  if (isAllowToShow()) {
    GuiInstance::paint(painter, option, widget);
  }
  if (isAllowToSelect()) {
    setFlag(QGraphicsItem::ItemIsSelectable, true);
  } else
    setFlag(QGraphicsItem::ItemIsSelectable, false);
}

inline bool GuiPad::isAllowToShow() const { return true; }
inline bool GuiPad::isAllowToSelect() const { return true; }
