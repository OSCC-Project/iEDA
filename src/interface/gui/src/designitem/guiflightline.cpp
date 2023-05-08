#include "guiflightline.h"
GuiFlightLine::GuiFlightLine(QVector<QPointF> pins, QGraphicsItem *parent) : QGraphicsItemGroup(parent){
    this->_pins = pins;
    QPointF this_pin = _pins.first();
    for(int i = 1; i<_pins.size();i++){
        GuiLine * l = new GuiLine();
        l->set_line(this_pin.x(),this_pin.y(),pins[i].x(),pins[i].y());
        l->set_pen(Qt::blue);
        addToGroup(l);
    }
}
