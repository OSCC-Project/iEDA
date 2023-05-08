/**
 * @file guigr.h
 * @author
 * @brief Test function.

 * @version 0.1
 * @date 2021-08-16(V0.1)
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef GUIGR_H
#define GUIGR_H

#include "guirect.h"

class GuiGrPrivate : public GuiRectPrivate
{
 public:
  explicit GuiGrPrivate();
  ~GuiGrPrivate();

  /// getter
  std::string get_layer() { return _layer; }

  /// setter

  void set_layer(std::string layer) { _layer = layer; }

 private:
  std::string _layer;
  //   friend class GuiGrRect;
};

class GuiGrRect : public GuiRect
{
 public:
  explicit GuiGrRect(QGraphicsItem* parent = nullptr);

  void paint(QPainter*                       painter,
             const QStyleOptionGraphicsItem* option,
             QWidget*                        widget = nullptr) override;

  void set_layer(const std::string layer);
  void add_info(std::string info) { _data->add_info(info); }
  void set_item_info(std::string info) { _data->set_item_info(info); }

 protected:
  GuiGrRect(GuiGrPrivate* data, QGraphicsItem* parent = nullptr);
  inline bool isAllowToShow() const override;

 private:
  GuiGrPrivate* _data;
};

#endif  // GUIGR_H
