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
