#include "guiitem.h"

GuiItem::GuiItem(QGraphicsItem* parent) : _d_ptr(new GuiItemPrivate()), QGraphicsItem(parent) {
  _data = _d_ptr;
  setCacheMode(QGraphicsItem::ItemCoordinateCache);
}

GuiItem::GuiItem(GuiItemPrivate* data, QGraphicsItem* parent) : _d_ptr(data), QGraphicsItem(parent) { _data = _d_ptr; }

GuiItem::~GuiItem() { delete _d_ptr; }

QPen GuiItem::get_pen() const { return _data->_pen; }

void GuiItem::set_pen(const QPen& pen) {
  if (_data->_pen == pen)
    return;
  _data->_pen = pen;
}

void GuiItem::set_pen(QColor color, qreal width, Qt::PenStyle style) {
  _data->_pen.setColor(color);
  _data->_pen.setWidthF(width);
  _data->_pen.setStyle(style);
}

QBrush GuiItem::get_brush() const { return _data->_brush; }

void GuiItem::set_brush(const QBrush& brush) {
  if (_data->_brush == brush)
    return;
  _data->_brush = brush;
}

void GuiItem::set_brush(const QColor& color, Qt::BrushStyle style) { set_brush(QBrush(color, style)); }

bool GuiItem::isObscuredBy(const QGraphicsItem* item) const { return QGraphicsItem::isObscuredBy(item); }
