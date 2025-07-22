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
#include "Utility.hpp"

namespace idrc {

// public

void Utility::initInst()
{
  if (_util_instance == nullptr) {
    _util_instance = new Utility();
  }
}

Utility& Utility::getInst()
{
  if (_util_instance == nullptr) {
    initInst();
  }
  return *_util_instance;
}

void Utility::destroyInst()
{
  if (_util_instance != nullptr) {
    delete _util_instance;
    _util_instance = nullptr;
  }
}

// private

Utility* Utility::_util_instance = nullptr;

}  // namespace idrc
