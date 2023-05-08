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
