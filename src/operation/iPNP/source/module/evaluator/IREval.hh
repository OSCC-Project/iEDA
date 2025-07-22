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
 * @file IREval.hh
 * @author Xinhao li
 * @brief
 * @version 0.1
 * @date 2024-07-15
 */

#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <limits>
#include <utility>

#include "api/PowerEngine.hh"
#include "api/TimingEngine.hh"
#include "api/Power.hh"
#include "PNPConfig.hh"

namespace idb {
  class IdbDesign;
  class IdbBuilder;
}

namespace ipower {
  class PowerEngine;
}

namespace ipnp {
  class IREval
  {
  public:

    IREval() = default;
    ~IREval() = default;

    void initIREval(idb::IdbBuilder* idb_builder, PNPConfig* pnp_config = nullptr);
    void runIREval(idb::IdbBuilder* idb_builder);

    // getter
    double getMaxIRDrop() const;
    double getMinIRDrop() const;
    double getAvgIRDrop() const;
    std::map<ista::Instance::Coordinate, double> get_Coord_IR_map() { return _coord_ir_map; }


  private:
    ista::TimingEngine* _timing_engine;
    ipower::PowerEngine* _power_engine;
    Sta* _ista;
    ipower::Power* _ipower;

    std::map<ista::Instance::Coordinate, double> _coord_ir_map;

  };

}  // namespace ipnp
