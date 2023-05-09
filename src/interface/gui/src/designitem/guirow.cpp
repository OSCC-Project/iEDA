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
#include "guirow.h"
GuiRow::GuiRow(QGraphicsItem* parent) : GuiRect(new GuiRowPrivate(), parent), _data(static_cast<GuiRowPrivate*>(_d_ptr)) {
  set_pen(QColor(45, 45, 45));
}
void GuiRow::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) {
  if (isAllowToShow()) {
    GuiRect::paint(painter, option, widget);
  }
  if (isAllowToSelect()) {
    setFlag(QGraphicsItem::ItemIsSelectable, true);
  } else
    setFlag(QGraphicsItem::ItemIsSelectable, false);
}

inline bool GuiRow::isAllowToShow() const { return true; }
inline bool GuiRow::isAllowToSelect() const { return true; }
