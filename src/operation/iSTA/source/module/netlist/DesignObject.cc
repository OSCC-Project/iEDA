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
 * @file DesignObject.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of base class.
 * @version 0.1
 * @date 2021-02-03
 */

#include "DesignObject.hh"
#include "string/Str.hh"

namespace ista {

DesignObject::DesignObject(const char* name) : _name(name) {}

DesignObject::DesignObject(DesignObject&& other) noexcept
    : _name(std::move(other._name)) {}

DesignObject& DesignObject::operator=(DesignObject&& rhs) noexcept {
  _name = std::move(rhs._name);

  return *this;
}

}  // namespace ista
