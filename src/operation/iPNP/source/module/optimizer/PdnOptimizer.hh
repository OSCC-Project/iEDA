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
 * @file PdnOptimizer.hh
 * @author Xinhao li
 * @brief
 * @version 0.1
 * @date 2024-07-15
 */

#pragma once

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "CongestionEval.hh"
#include "FastPlacer.hh"
#include "GridManager.hh"
#include "IREval.hh"

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

namespace ipnp {

  struct RegionData;
  struct OptimizationResult;

  class PdnOptMethod
  {
  public:
    virtual ~PdnOptMethod() = default;
  };

  class SimulatedAnnealing;

  class PdnOptimizer
  {
  public:
    PdnOptimizer();
    ~PdnOptimizer();

    GridManager get_out_put_grid() { return _output_pdn_grid; }
    double get_opt_score() const { return _opt_score; }

    // 获取区域评估数据
    // std::vector<std::vector<std::vector<RegionData>>> get_region_data() const { return _region_data; }

    // 优化网络 - 两阶段方法（先全局，再区域）
    void optimize(GridManager initial_pdn, idb::IdbBuilder* idb_builder);

    // 仅全局优化
    void optimizeGlobal(GridManager initial_pdn, idb::IdbBuilder* idb_builder);

    // 仅区域化优化
    void optimizeByRegion(GridManager initial_pdn, idb::IdbBuilder* idb_builder);

    // 设置权重
    void setWeights(double ir_drop_weight, double overflow_weight);

    // 设置模拟退火参数
    void setAnnealingParams(double initial_temp, double cooling_rate, double min_temp, int iterations_per_temp);

  private:
    GridManager _input_pdn_grid;
    GridManager _output_pdn_grid;
    double _opt_score;

    // 区域评估数据
    std::vector<std::vector<std::vector<RegionData>>> _region_data;

    // 模拟退火算法参数
    double _initial_temp;
    double _cooling_rate;
    double _min_temp;
    int _iterations_per_temp;

    // 权重
    double _ir_drop_weight;
    double _overflow_weight;
  };

}  // namespace ipnp
