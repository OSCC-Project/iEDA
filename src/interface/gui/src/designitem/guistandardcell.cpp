#include "guistandardcell.h"

#include <QDebug>
GuiStandardCell::GuiStandardCell(QGraphicsItem* parent)
    : GuiInstance(new GuiStandardCellPrivate(), parent), _data(static_cast<GuiStandardCellPrivate*>(_d_ptr)) {
  set_pen(QColor(130, 130, 130));
  set_brush(QColor(140, 140, 140), Qt::BrushStyle::Dense7Pattern);
}
void GuiStandardCell::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) {
  if (isAllowToShow()) {
    GuiInstance::paint(painter, option, widget);
  }
  if (isAllowToSelect()) {
    setFlag(QGraphicsItem::ItemIsSelectable, true);
  } else
    setFlag(QGraphicsItem::ItemIsSelectable, false);
}

inline bool GuiStandardCell::isAllowToShow() const { return true; }
inline bool GuiStandardCell::isAllowToSelect() const { return true; }
