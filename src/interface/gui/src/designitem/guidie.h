/**
 * @file guicorearea.h
 * @author Fuxing Huang (yellowstar4869@foxmail.com)
 * @brief Die, Just a rectangle on the outside, nothing special for now.
 * @version 0.2
 * @date 2021-07-01(V0.1) 2021-08-11(V0.2)
 */
#ifndef GUIDIE_H
#define GUIDIE_H
#include "guirect.h"

class GuiDie : public GuiRect {
 public:
  explicit GuiDie(QGraphicsItem *parent = nullptr);
};

#endif  // GUIDIE_H
