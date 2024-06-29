// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************

/**
 * @file TimingDBAdapter.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief TimingDBAdapter is a adapter convert the idb to timing netlist.
 * @version 0.1
 * @date 2021-09-04
 */
#pragma once
#include "FlatMap.hh"
#include "sta/Sta.hh"

namespace ista {

/**
 * @brief The timing db adapter.
 *
 */
class TimingDBAdapter {
 public:
  explicit TimingDBAdapter(Sta* ista);
  virtual ~TimingDBAdapter() = default;

  virtual bool isPlaced(DesignObject* pin_or_port) {
    LOG_FATAL << "The function is not implemented.";
    return true;
  }
  virtual double dbuToMeters(int distance) const {
    LOG_FATAL << "The function is not implemented.";
    return 0.0;
  }

  virtual void location(DesignObject* pin_or_port,
                        // Return values.
                        double& x, double& y, bool& exists) {
    LOG_FATAL << "The function is not implemented.";
  }

  virtual unsigned convertDBToTimingNetlist(bool link_all_cell = false) {
    LOG_FATAL << "The function is not implemented.";
    return 1;
  }

  virtual unsigned BuildRCTreeWithRoutingSegment() {
    LOG_FATAL << "The function is not implemented.";
    return 1;
  }

  Netlist* getNetlist() { return _ista->get_netlist(); }

 protected:
  Sta* _ista = nullptr;

 private:
  FORBIDDEN_COPY(TimingDBAdapter);
};

}  // namespace ista
