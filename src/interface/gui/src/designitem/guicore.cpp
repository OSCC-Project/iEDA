#include "guicore.h"
GuiCore::GuiCore(QGraphicsItem *parent) : GuiRect(parent) {
    set_pen(Qt::white, 0);
}