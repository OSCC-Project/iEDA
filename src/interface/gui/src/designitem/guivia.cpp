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
