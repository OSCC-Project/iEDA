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
#include <map>
#include <set>
#include <string>
#include <vector>

namespace ista {
class TimingEngine;
class TimingIDBAdapter;
class TimingDBAdapter;
class DesignObject;
class RctNode;
class Net;
class Pin;
class Instance;
class StaVertex;
class LibCell;
class LibPort;
class LibArc;
class StaSeqPathData;
enum class AnalysisMode;
enum class TransType;
}  // namespace ista

namespace ito {

using namespace ista;

using TODelay = double;
using TOSlew = TODelay;
using TOSlack = TODelay;
using TOSlacks = TOSlack[2][2];
using TORequired = TODelay;
using TOArcDelay = TODelay;

using TOLevel = int;
using TOVertexSeq = std::vector<StaVertex*>;
using TOLibertyCellSeq = std::vector<LibCell*>;
using TODesignObjSeq = std::vector<ista::DesignObject*>;
using TOVertexSet = std::set<StaVertex*>;

using TOSlewTarget = std::array<TOSlew, 2>;  // TransType: kFall / kRise;

}  // namespace ito