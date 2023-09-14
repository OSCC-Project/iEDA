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
 * @file Pin.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */

#include "Pin.hh"
namespace icts {

void Pin::init()
{
  if (_inst->isSink()) {
    set_type(NodeType::kSinkPin);
  } else if (_inst->isBuffer()) {
    set_type(NodeType::kBufferPin);
  } else {
    set_type(NodeType::kNoneLibPin);
  }
}
std::string Pin::get_cell_master() const
{
  return _inst->get_cell_master();
}

}  // namespace icts