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

namespace ieda {
class Log;
class Str;
class Time;
}  // namespace ieda
namespace icts {
struct PathInfo;
enum class LayerPattern;
class Node;
class Pin;
class Net;
class Inst;
enum class TopoType;
class CtsCellLib;
class CtsLibs;
class CtsReportTable;
class CtsLog;
class CtsInstance;
class CtsConfig;
class CtsDesign;
class CtsDBWrapper;
class CtsPin;
class CtsSignalWire;
class CtsNet;
class EvalNet;
class Evaluator;
class Endpoint;
enum class FitType;
class ModelBase;
class ModelFactory;
template <typename T>
class CtsPoint;
using Point = icts::CtsPoint<int>;
}  // namespace icts

namespace ista {
class TimingEngine;
class TimingIDBAdapter;
class DesignObject;
class RctNode;
class Net;
}  // namespace ista
namespace idb {
class IdbPin;
class IdbInstance;
class IdbNet;
class IdbRegularWireSegment;
class IdbLayerShape;
class FeatureSummary;
}  // namespace idb

namespace ito {
class Tree;
}  // namespace ito
namespace idrc {

class RegionQuery;
class DrcRect;

}  // namespace idrc

namespace irt {

class Violation;
class LayerCoord;
template <typename T>
class Segment;

}  // namespace irt

namespace eval {

class TimingNet;
class TimingPin;
class TileGrid;

}  // namespace eval
