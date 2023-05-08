#include "guiwire.h"
GuiWire::GuiWire(QVector<QLineF> wires, QGraphicsItem *parent ) :  QGraphicsItemGroup(parent){
    this->_wires = wires;
    foreach (QLineF line, wires) {
        GuiLine * l = new GuiLine();
        l->set_line(line);
        l->set_pen(Qt::white);
        addToGroup(l);
    }
}
