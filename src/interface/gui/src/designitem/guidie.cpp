#include "guidie.h"
GuiDie::GuiDie(QGraphicsItem *parent) : GuiRect(parent) {
    set_pen(Qt::white, 0, Qt::PenStyle::DashDotLine);
}
