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
#include "LGMethodCreator.hh"

#include "Abacus.hh"
#include "LGCustomization.hh"
#include "LGMethodInterface.hh"

namespace ieda_solver {

LGMethodInterface* LGMethodCreator::createMethod(LG_METHOD method_type)
{
  LGMethodInterface* method = nullptr;

  switch (method_type) {
    case LG_METHOD::kAbacus:
      method = new Abacus();
      break;
    case LG_METHOD::kCustomized:
      method = new LGCustomization();
      break;

    default:
      break;
  }

  return method;
}

}  // namespace ieda_solver