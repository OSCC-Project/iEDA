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
#include "guiblock.h"

#include <QPainter>
#include <QStyleOptionGraphicsItem>
GuiBlock::GuiBlock(QGraphicsItem* parent)
    : GuiInstance(new GuiBlockPrivate(), parent), _data(static_cast<GuiBlockPrivate*>(_d_ptr)) {
  set_pen(QColor(130, 130, 130), 2);
  set_brush(QColor(140, 140, 140), Qt::Dense4Pattern);
  set_halo_brush(QColor(201, 108, 108), Qt::SolidPattern);
}

QPen GuiBlock::get_halo_pen() { return _data->_halo_pen; }
void GuiBlock::set_halo_pen(const QPen& halo_pen) {
  if (_data->_halo_pen == halo_pen)
    return;
  _data->_halo_pen = halo_pen;
  //   update();
}
void GuiBlock::set_halo_pen(QColor color, qreal width, Qt::PenStyle style) {
  _data->_halo_pen.setColor(color);
  _data->_halo_pen.setWidthF(width);
  _data->_halo_pen.setStyle(style);
  //   update();
}

QBrush GuiBlock::get_halo_brush() { return _data->_halo_brush; }
void GuiBlock::set_halo_brush(const QBrush& halo_brush) {
  if (_data->_halo_brush == halo_brush)
    return;
  _data->_halo_brush = halo_brush;
  //   update();
}
void GuiBlock::set_halo_brush(const QColor& color, Qt::BrushStyle style) { set_halo_brush(QBrush(color, style)); }

QRectF GuiBlock::get_halo_rect() { return _data->_halo_rect; }
void GuiBlock::set_halo_rect(const QRectF& halo_rect) {
  _data->_halo_rect = halo_rect;
  //   update();
}
void GuiBlock::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) {
  Q_UNUSED(widget);
  if (isAllowToShow()) {
    if (!_data->_halo_rect.isNull()) {
      QBrush brush = _data->_halo_brush;

      brush.setTransform(painter->worldTransform().inverted());
      painter->setBrush(brush);

      painter->fillRect(_data->_halo_rect, brush);
    }

    GuiInstance::paint(painter, option, widget);
  }
  if (isAllowToSelect()) {
    setFlag(QGraphicsItem::ItemIsSelectable, true);
  } else
    setFlag(QGraphicsItem::ItemIsSelectable, false);
}

inline bool GuiBlock::isAllowToShow() const { return true; }
inline bool GuiBlock::isAllowToSelect() const { return true; }

GuiBlockPrivate::GuiBlockPrivate() : _halo_pen(QPen()), _halo_brush(QBrush()), _halo_rect(QRectF()) { }
GuiBlockPrivate::~GuiBlockPrivate() { }
