// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file guigr.h
 * @author
 * @brief Test function.

 * @version 0.1
 * @date 2021-08-16(V0.1)
 *
 *
 */
#ifndef GUIGR_H
#define GUIGR_H

#include "guirect.h"

class GuiGrPrivate : public GuiRectPrivate {
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

class GuiGrRect : public GuiRect {
 public:
  explicit GuiGrRect(QGraphicsItem* parent = nullptr);

  void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

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
