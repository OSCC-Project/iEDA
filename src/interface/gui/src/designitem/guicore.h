/**
 * @file guicorearea.h
 * @author Fuxing Huang (yellowstar4869@foxmail.com)
 * @brief Core area, just a rectangle, nothing special for now.
 * @version 0.2
 * @date 2021-07-01(V0.1) 2021-08-11(V0.2)
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#ifndef GUICOREAREA_H
#define GUICOREAREA_H
#include "guirect.h"

class GuiCore : public GuiRect {
 public:
     explicit GuiCore(QGraphicsItem *parent = nullptr);
};

#endif  // GUICOREAREA_H
