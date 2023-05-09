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
#include "TimingNet.hpp"

namespace eval {

// void TimingNet::add_pin_pair(std::string name_1, int32_t x_1, int32_t y_1, int layer_id_1, std::string name_2, int32_t x_2, int32_t y_2,
//                              int layer_id_2)
// {
//   TimingPin* pin_1 = new TimingPin();
//   TimingPin* pin_2 = new TimingPin();
//   pin_1->set_name(name_1);
//   pin_1->set_coord(Point<int64_t>(x_1, y_1));
//   pin_1->set_layer_id(layer_id_1);
//   if (name_1 == "fake") {
//     pin_1->set_is_real_pin(false);
//     pin_1->set_id(_fake_pin_count++);
//   } else {
//     pin_1->set_is_real_pin(true);
//   }
//   pin_2->set_name(name_2);
//   pin_2->set_coord(Point<int64_t>(x_2, y_2));
//   pin_2->set_layer_id(layer_id_2);
//   if (name_2 == "fake") {
//     pin_2->set_is_real_pin(false);
//     pin_2->set_id(_fake_pin_count++);
//   } else {
//     pin_2->set_is_real_pin(true);
//   }
//   add_pin_pair(pin_1, pin_2);
// }

// void TimingNet::add_pin_pair(int32_t x_1, int32_t y_1, int32_t x_2, int32_t y_2, int layer_id_1, int layer_id_2 = -1,
//                              std::string name_1 = "fake", std::string name_2 = "fake")
// {
//   TimingPin* pin_1 = new TimingPin();
//   TimingPin* pin_2 = new TimingPin();
//   pin_1->set_coord(Point<int64_t>(x_1, y_1));
//   pin_2->set_coord(Point<int64_t>(x_2, y_2));
//   if (x_1 == x_2 && y_1 == y_2) {
//     assert(layer_id_2 != (int) -1);
//     pin_1->set_layer_id(layer_id_1);
//     pin_2->set_layer_id(layer_id_2);
//   } else {
//     pin_1->set_layer_id(layer_id_1);
//     pin_2->set_layer_id(layer_id_1);
//   }
//   pin_1->set_name(name_1);
//   pin_2->set_name(name_2);
//   if (name_1 == "fake") {
//     pin_1->set_is_real_pin(false);
//     pin_1->set_id(_fake_pin_count++);
//   } else {
//     pin_1->set_is_real_pin(true);
//   }
//   if (name_2 == "fake") {
//     pin_2->set_is_real_pin(false);
//     pin_2->set_id(_fake_pin_count++);
//   } else {
//     pin_2->set_is_real_pin(true);
//   }
//   add_pin_pair(pin_1, pin_2);
// }

}  // namespace eval
