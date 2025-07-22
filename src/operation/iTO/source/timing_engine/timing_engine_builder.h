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
#pragma once

#include <list>
#include <map>
#include <string>
#include <vector>

#include "define.h"
#include "timing_engine.h"

using namespace ista;

namespace ito {
using namespace std;

class TimingEngineBuilder
{
 public:
  TimingEngineBuilder();
  ~TimingEngineBuilder();

  void buildEngine();

 private:
  /// data structure
  static constexpr float _slew_2_load_cap_factor = 10.0;

  /////////init engine
  void initISTA();
  void findEquivLibCells();
  void findDrvrVertices();
  void findBufferCells();

  void calcCellTargetLoads();
  void calcTargetSlewsForBuffer();
  void calcTargetSlewsForBuffer(TOSlew rise_fall_slew[], int rise_fall_number[], LibCell* buffer);
  void calcArcSlew(TOSlew rise_fall_slew[], int rise_fall_number[], LibPort* port, TransType trans_type, LibArc* arc);
  void calcTargetLoad(LibCell* cell);
  void calcTargetLoad(float& target_total_load, int& total_arc_num, LibArc* arc, TransType rf);
  float calcTargetLoad(LibArc* arc, TransType in_type, TransType out_type);
  TOSlew calcSlewDiffOfGate(TransType in_type, float cap_load, TOSlew in_slew, TOSlew out_slew, LibArc* arc);
};

}  // namespace ito