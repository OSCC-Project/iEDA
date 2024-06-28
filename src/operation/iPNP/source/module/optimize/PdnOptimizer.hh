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

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace idb {
class IdbLayer;
class IdbSpecialWireSegment;
class IdbRegularWireSegment;
class IdbBlockageList;
class IdbInstance;
class IdbRect;
class IdbVia;
class IdbLayerCut;
class IdbPin;
class IdbSpecialNet;
class IdbLayerRouting;
class IdbSpecialWire;

enum class SegmentType : int8_t;
enum class IdbWireShapeType : uint8_t;
enum class IdbOrient : uint8_t;

template <typename T>
class IdbCoordinate;
}  // namespace idb

namespace ipnp{

class PdnOptMethod {
  
};

class PdnOptimizer{
 public:
  PdnOptimizer();
  ~PdnOptimizer();
  
  idb::IdbRegularWireSegment* optimize(idb::IdbRegularWireSegment* initial_pdn); //整个优化流程，包括评估器预测、调整网络;调用具体算法

 private:
  idb::IdbRegularWireSegment* input_pdn_grid;
  idb::IdbRegularWireSegment* output_pdn_grid;
  idb::IdbLayer* output_def; //整个优化完成后的输出DEF文件
  double opt_score;

};
}

