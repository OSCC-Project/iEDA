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
