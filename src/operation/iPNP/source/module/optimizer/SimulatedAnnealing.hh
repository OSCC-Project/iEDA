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
 * @file SimulatedAnnealing.hh
 * @author Xinhao li
 * @brief
 * @version 0.1
 * @date 2024-07-15
 */

 // SimulatedAnnealing.hh
#pragma once

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <random>
#include <cmath>

#include "GridManager.hh"
#include "IREval.hh"
#include "CongestionEval.hh"

namespace idb {
  class IdbBuilder;
}

namespace ipnp {
  

  struct RegionData {
    double ir_drop;
    int32_t overflow;
  };

  struct OptimizationResult {
    GridManager best_grid;
    double best_cost;
    std::vector<std::vector<std::vector<RegionData>>> region_data; // [layer][row][col]
  };

  struct CostResult {
    double cost;
    double ir_drop;
    int32_t overflow;
    double normalized_ir_drop;
    double normalized_overflow;
  };

  class SimulatedAnnealing
  {
  public:
    SimulatedAnnealing(double initial_temp = 100.0,
      double cooling_rate = 0.95,
      double min_temp = 0.1,
      int iterations_per_temp = 10);
    ~SimulatedAnnealing() = default;

    // 全局优化函数
    OptimizationResult optimize(const GridManager& initial_grid, idb::IdbBuilder* idb_builder);

    // 区域化优化函数
    OptimizationResult optimizeByRegion(const GridManager& initial_grid, idb::IdbBuilder* idb_builder);

    // 设置权重
    void setWeights(double ir_drop_weight, double overflow_weight);

    // 评估解的质量 - 返回加权的成本值
    CostResult evaluateCost(const GridManager& new_solution, const GridManager& current_solution, idb::IdbBuilder* idb_builder);

    // 获取所有区域评估数据
    std::vector<std::vector<std::vector<RegionData>>> evaluateAllRegions(const GridManager& grid);


  private:
    CongestionEval _cong_eval;
    IREval _ir_eval;

    // 模拟退火参数
    double _initial_temp;
    double _cooling_rate;
    double _min_temp;
    int _iterations_per_temp;
    int _max_no_improvement;

    // 随机数生成器
    std::mt19937 _rng;

    // 权重系数
    double _ir_drop_weight;
    double _overflow_weight;

    double _max_ir_drop;
    double _min_ir_drop;
    int32_t _max_overflow;
    int32_t _min_overflow;

    // 内部功能函数

    // 生成邻域解 - 随机选择一个区域并修改其模板
    GridManager generateNeighbor(const GridManager& current);

    // 生成单个区域的邻域解 - 针对指定区域选择合适的模板
    GridManager generateRegionNeighbor(const GridManager& current, int layer, int row, int col);

    // 评估单个区域的质量 - 返回区域的IR drop和overflow
    RegionData evaluateRegion(const GridManager& solution, int layer, int row, int col);

    // 接受准则 - 决定是否接受新解
    bool acceptSolution(double current_cost, double new_cost, double temp);

    // 终止条件检查
    bool shouldTerminate(int iterations, double temp, int no_improvement_count);

    // 随机选择模板
    SingleTemplate selectRandomTemplate(bool is_horizontal);

    // 选择稀疏模板 - 用于降低拥塞
    SingleTemplate selectSparseTemplate(bool is_horizontal);

    // 选择密集模板 - 用于降低IR drop
    SingleTemplate selectDenseTemplate(bool is_horizontal);

    // 归一化IR drop值
    double normalizeIRDrop(double max_ir_drop, double min_ir_drop, double avg_ir_drop);

    // 归一化overflow值
    double normalizeOverflow(int32_t overflow);

    // 判断两个模板是否相同
    bool isSameTemplate(const SingleTemplate& t1, const SingleTemplate& t2);
  };

}  // namespace ipnp
