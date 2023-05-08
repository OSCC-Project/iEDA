/**
 * @file guipad.h
 * @author Fuxing Huang (yellowstar4869@foxmail.com)
 * @brief This class inherits from GuiInstance to represent Pad in the GUI
 * @version 0.2
 * @date 2021-07-09(V0.1), 2021-08-13(V0.2)
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef GUIPAD_H
#define GUIPAD_H
#include "guiinstance.h"

class GuiPadPrivate;

class GuiPad : public GuiInstance {
 public:
  explicit GuiPad(QGraphicsItem *parent = nullptr);
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;

 protected:
  inline bool isAllowToShow() const override;
  inline bool isAllowToSelect() const override;

 private:
  GuiPadPrivate *_data;
};

class GuiPadPrivate : public GuiInstancePrivate {
 public:
  explicit GuiPadPrivate() { }
  ~GuiPadPrivate() = default;
};

#endif  // GUIPAD_H
