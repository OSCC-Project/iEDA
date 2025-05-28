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
 * @file SimulatedAnnealing.cpp
 * @author Xinhao li
 * @brief
 * @version 0.1
 * @date 2024-07-15
 */

 // SimulatedAnnealing.cpp
#include "SimulatedAnnealing.hh"
#include "iPNPIdbWrapper.hh"
namespace ipnp {

  SimulatedAnnealing::SimulatedAnnealing(double initial_temp, double cooling_rate,
    double min_temp, int iterations_per_temp)
    : _initial_temp(initial_temp),
    _cooling_rate(cooling_rate),
    _min_temp(min_temp),
    _iterations_per_temp(iterations_per_temp),
    _max_no_improvement(20),
    _ir_drop_weight(0.7),
    _overflow_weight(0.3),
    _max_ir_drop(std::numeric_limits<double>::min()),
    _min_ir_drop(std::numeric_limits<double>::max()),
    _max_overflow(std::numeric_limits<int32_t>::min()),
    _min_overflow(std::numeric_limits<int32_t>::max())
  {
    // 初始化随机数生成器
    std::random_device rd;
    _rng = std::mt19937(rd());
  }

  OptimizationResult SimulatedAnnealing::optimize(const GridManager& initial_grid, idb::IdbBuilder* idb_builder)
  {
    GridManager current_solution = initial_grid;
    GridManager best_solution = current_solution;
    _ir_eval.initIREval(idb_builder);

    CostResult current_result = evaluateCost(current_solution, idb_builder);
    double current_cost = current_result.cost;
    double best_cost = current_cost;

    // 打印初始解信息
    LOG_INFO << "Initial solution: " << std::endl
      << "  IR Drop: " << current_result.ir_drop
      << ", Normalized IR: " << current_result.normalized_ir_drop
      << ", Overflow: " << current_result.overflow
      << ", Normalized OF: " << current_result.normalized_overflow
      << ", Total Cost: " << current_cost
      << " (IR权重:" << _ir_drop_weight << ", OF权重:" << _overflow_weight << ")" << std::endl;

    double temperature = _initial_temp;
    int no_improvement_count = 0;
    int total_iterations = 0;

    // 主循环
    while (!shouldTerminate(total_iterations, temperature, no_improvement_count)) {
      for (int i = 0; i < _iterations_per_temp; i++) {
        // 生成邻域解
        GridManager new_solution = generateNeighbor(current_solution);

        // 评估新解
        CostResult new_result = evaluateCost(new_solution, idb_builder);
        double new_cost = new_result.cost;

        // 打印每个邻域解的详细信息（所有CostResult字段）
        LOG_INFO << "Iter[" << total_iterations << "] Temp[" << temperature << "]" << std::endl
          << "  IR Drop: " << new_result.ir_drop
          << ", Normalized IR: " << new_result.normalized_ir_drop << std::endl
          << "  Overflow: " << new_result.overflow
          << ", Normalized OF: " << new_result.normalized_overflow << std::endl
          << "  Total Cost: " << new_cost
          << " (IR权重:" << _ir_drop_weight << ", OF权重:" << _overflow_weight << ")" << std::endl;

        // 接受准则
        if (acceptSolution(current_cost, new_cost, temperature)) {
          current_solution = new_solution;
          current_cost = new_cost;

          LOG_INFO << "  Solution ACCEPTED! New current cost: " << current_cost << std::endl;

          // 更新全局最优解
          if (new_cost < best_cost) {
            best_solution = new_solution;
            best_cost = new_cost;
            no_improvement_count = 0;
            LOG_INFO << "  NEW BEST SOLUTION! Cost: " << best_cost
              << ", IR Drop: " << new_result.ir_drop
              << ", Overflow: " << new_result.overflow << std::endl;
          }
          else {
            LOG_INFO << "  Solution accepted but not better than best. No change to best solution." << current_cost << std::endl;
            no_improvement_count++;
          }
        }
        else {
          no_improvement_count++;
        }

        total_iterations++;
      }

      // 降温
      temperature *= _cooling_rate;
      LOG_INFO << "===== Temperature: " << temperature
        << ", Current cost: " << current_cost
        << ", Best cost so far: " << best_cost
        << ", No improvement count: " << no_improvement_count << " =====" << std::endl;


      LOG_INFO << "Temperature: " << temperature << ", Best cost: " << best_cost;
    }

    // 返回优化结果
    OptimizationResult result;
    result.best_grid = best_solution;
    result.best_cost = best_cost;
    result.region_data = evaluateAllRegions(best_solution);

    return result;
  }

  // OptimizationResult SimulatedAnnealing::optimizeByRegion(const GridManager& initial_grid, idb::IdbBuilder* idb_builder)
  // {
  //   GridManager current_solution = initial_grid;
  //   GridManager best_solution = current_solution;

  //   double current_cost = evaluateCost(current_solution, idb_builder);
  //   double best_cost = current_cost;

  //   // 对每个区域独立进行模拟退火
  //   for (int layer = 0; layer < current_solution.get_layer_count(); layer++) {
  //     for (int row = 0; row < current_solution.get_ho_region_num(); row++) {
  //       for (int col = 0; col < current_solution.get_ver_region_num(); col++) {
  //         double temperature = _initial_temp;
  //         int no_improvement_count = 0;
  //         int total_iterations = 0;

  //         LOG_INFO << "Optimizing region [" << layer << "][" << row << "][" << col << "]";

  //         while (!shouldTerminate(total_iterations, temperature, no_improvement_count)) {
  //           for (int i = 0; i < _iterations_per_temp; i++) {
  //             // 生成区域邻域解
  //             GridManager new_solution = generateRegionNeighbor(current_solution, layer, row, col);

  //             // 评估新解
  //             double new_cost = evaluateCost(new_solution, idb_builder);

  //             // 接受准则
  //             if (acceptSolution(current_cost, new_cost, temperature)) {
  //               current_solution = new_solution;
  //               current_cost = new_cost;

  //               // 更新全局最优解
  //               if (new_cost < best_cost) {
  //                 best_solution = new_solution;
  //                 best_cost = new_cost;
  //                 no_improvement_count = 0;
  //               }
  //               else {
  //                 no_improvement_count++;
  //               }
  //             }
  //             else {
  //               no_improvement_count++;
  //             }

  //             total_iterations++;
  //           }

  //           // 降温
  //           temperature *= _cooling_rate;
  //         }

  //         LOG_INFO << "Region [" << layer << "][" << row << "][" << col << "] optimized. Best cost: " << best_cost;
  //       }
  //     }
  //   }

  //   // 返回优化结果
  //   OptimizationResult result;
  //   result.best_grid = best_solution;
  //   result.best_cost = best_cost;
  //   result.region_data = evaluateAllRegions(best_solution);

  //   return result;
  // }

  void SimulatedAnnealing::setWeights(double ir_drop_weight, double overflow_weight)
  {
    _ir_drop_weight = ir_drop_weight;
    _overflow_weight = overflow_weight;
  }

  GridManager SimulatedAnnealing::generateNeighbor(const GridManager& current)
  {
    GridManager neighbor = current;

    // 随机选择一个层 - 只选择M6-M3 (对应layer索引3-6)
    std::uniform_int_distribution<int> layer_dist(3, 6); // 只修改M6-M3层
    std::uniform_int_distribution<int> row_dist(0, current.get_ho_region_num() - 1);
    std::uniform_int_distribution<int> col_dist(0, current.get_ver_region_num() - 1);

    int layer = layer_dist(_rng);
    int row = row_dist(_rng);
    int col = col_dist(_rng);

    // 获取当前模板信息
    const auto& old_template = current.get_template_data()[layer][row][col];

    // 打印当前模板信息
    LOG_INFO << "Attempting to change template at: Layer M" << current.get_power_layers()[layer]
      << " (idx:" << layer << "), Region[" << row << "][" << col << "]" << std::endl;
    LOG_INFO << "  Old template information: " << std::endl;
    if (old_template.get_direction() == StripeDirection::kHorizontal) {
      LOG_INFO << "direction: Horizontal" << std::endl;
    }
    else {
      LOG_INFO << "direction: Vertical" << std::endl;
    }
    LOG_INFO << "width: " << old_template.get_width() << std::endl;
    LOG_INFO << "pg_offset: " << old_template.get_pg_offset() << std::endl;
    LOG_INFO << "space: " << old_template.get_space() << std::endl;
    LOG_INFO << "offset: " << old_template.get_offset() << std::endl;


    SingleTemplate new_template;
    std::vector<SingleTemplate> vertical_templates = current.get_vertical_templates();
    std::vector<SingleTemplate> horizontal_templates = current.get_horizontal_templates();

    // 随机选择一个新模板
    bool is_vertical = (layer % 2 == 0); // M9的索引是0，偶数，Vertical
    if (is_vertical) {
      // 找到旧模板在模板库中的索引
      size_t old_idx = 0;
      for (size_t i = 0; i < vertical_templates.size(); i++) {
        if (isSameTemplate(old_template, vertical_templates[i])) {
          old_idx = i;
          break;
        }
      }

      // 选择不同的模板
      std::uniform_int_distribution<size_t> template_dist(0, vertical_templates.size() - 2);
      size_t template_idx = template_dist(_rng);
      // 如果选中的索引大于等于旧模板索引，则向后移一位以跳过旧模板
      if (template_idx >= old_idx) {
        template_idx++;
      }
      new_template = vertical_templates[template_idx];
    }
    else {
      // 找到旧模板在模板库中的索引
      size_t old_idx = 0;
      bool found = false;
      for (size_t i = 0; i < horizontal_templates.size(); i++) {
        if (isSameTemplate(old_template, horizontal_templates[i])) {
          old_idx = i;
          found = true;
          break;
        }
      }

      // 选择不同的模板
      std::uniform_int_distribution<size_t> template_dist(0, horizontal_templates.size() - 2);
      size_t template_idx = template_dist(_rng);
      // 如果选中的索引大于等于旧模板索引，则向后移一位以跳过旧模板
      if (found && template_idx >= old_idx) {
        template_idx++;
      }
      new_template = horizontal_templates[template_idx];
    }

    // 打印新模板信息
    LOG_INFO << "New template information: " << std::endl;
    if (new_template.get_direction() == StripeDirection::kHorizontal) {
      LOG_INFO << "direction: Horizontal" << std::endl;
    }
    else {
      LOG_INFO << "direction: Vertical" << std::endl;
    }
    LOG_INFO << "width: " << new_template.get_width() << std::endl;
    LOG_INFO << "pg_offset: " << new_template.get_pg_offset() << std::endl;
    LOG_INFO << "space: " << new_template.get_space() << std::endl;
    LOG_INFO << "offset: " << new_template.get_offset() << std::endl;

    // 设置新模板
    neighbor.set_single_template(layer, row, col, new_template);

    return neighbor;
  }

  GridManager SimulatedAnnealing::generateRegionNeighbor(const GridManager& current, int layer, int row, int col)
  {
    GridManager neighbor = current;

    // // 评估该区域当前的情况
    // RegionData region_data = evaluateRegion(current, layer, row, col);

    // // 根据区域状态选择适当的模板
    // bool is_horizontal = (layer % 2 == 0); // 假设偶数层是水平方向
    // SingleTemplate new_template;

    // // 随机决定是优化IR drop还是overflow
    // std::uniform_real_distribution<double> dist(0.0, 1.0);
    // double random_value = dist(_rng);

    // if (region_data.overflow > region_data.ir_drop || random_value < 0.3) {
    //   // 拥塞问题更严重，或者有30%概率选择稀疏模板 
    //   new_template = selectSparseTemplate(is_horizontal);
    // }
    // else {
    //   // IR drop问题更严重，或者有70%概率选择密集模板
    //   new_template = selectDenseTemplate(is_horizontal);
    // }

    // // 设置新模板
    // neighbor.set_single_template(layer, row, col, new_template);

    return neighbor;
  }

  CostResult SimulatedAnnealing::evaluateCost(const GridManager& solution, idb::IdbBuilder* idb_builder)
  {

    iPNPIdbWrapper temp_wrapper;
    temp_wrapper.set_idb_design(idb_builder->get_def_service()->get_design());
    temp_wrapper.set_idb_builder(idb_builder);
    temp_wrapper.saveToIdb(solution);
    temp_wrapper.writeIdbToDef("/home/sujianrong/iEDA/src/operation/iPNP/data/test/debug.def");

    _cong_eval.evalEGR(idb_builder);
    int32_t overflow = _cong_eval.get_total_overflow_union();
 
    _ir_eval.runIREval(idb_builder);
    double max_ir_drop = _ir_eval.getMaxIRDrop();
    double min_ir_drop = _ir_eval.getMinIRDrop();

    // 归一化并加权
    double normalized_ir_drop = normalizeIRDrop(max_ir_drop, min_ir_drop);
    double normalized_overflow = normalizeOverflow(overflow);

    double cost = _ir_drop_weight * normalized_ir_drop + _overflow_weight * normalized_overflow;


    return { cost, max_ir_drop, overflow, normalized_ir_drop, normalized_overflow };
  }

  RegionData SimulatedAnnealing::evaluateRegion(const GridManager& solution, int layer, int row, int col)
  {
    RegionData data;

    // TODO: 调用IREval和CongestionEval获取特定区域的IR drop和overflow
    // 这里是示例实现，需要替换为实际调用

    // // 获取该区域的模板
    // const SingleTemplate& template_data = solution.get_template_data()[layer][row][col];

    // // 模拟计算IR drop - 与电源线密度成反比
    // double wire_density = template_data.get_wire_width() / (template_data.get_wire_width() + template_data.get_wire_spacing());
    // data.ir_drop = 1.0 / (wire_density + 0.1); // 避免除以零

    // // 模拟计算overflow - 与电源线密度成正比
    // data.overflow = wire_density * 2.0;

    return data;
  }

  bool SimulatedAnnealing::acceptSolution(double current_cost, double new_cost, double temp)
  {
    if (new_cost < current_cost) {
      return true;
    }

    // 根据Metropolis准则计算接受概率
    // 概率公式: P = exp(-ΔE/T)，其中ΔE是成本差值，T是当前温度
    // 温度越高，接受较差解的概率越大；温度越低，算法越倾向于只接受更优解
    double delta = new_cost - current_cost;
    double acceptance_probability = std::exp(-delta / temp);

    // 如果随机数小于接受概率，则接受新解
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(_rng) < acceptance_probability;
  }

  bool SimulatedAnnealing::shouldTerminate(int iterations, double temp, int no_improvement_count)
  {
    // 温度低于阈值
    if (temp < _min_temp) return true;

    // 连续多次无改进
    if (no_improvement_count > _max_no_improvement) return true;

    return false;
  }

  std::vector<std::vector<std::vector<RegionData>>> SimulatedAnnealing::evaluateAllRegions(const GridManager& grid)
  {
    std::vector<std::vector<std::vector<RegionData>>> region_data;

    // // 分配空间
    // region_data.resize(grid.get_layer_count());
    // for (int layer = 0; layer < grid.get_layer_count(); layer++) {
    //   region_data[layer].resize(grid.get_ho_region_num());
    //   for (int row = 0; row < grid.get_ho_region_num(); row++) {
    //     region_data[layer][row].resize(grid.get_ver_region_num());
    //   }
    // }

    // // 评估每个区域
    // for (int layer = 0; layer < grid.get_layer_count(); layer++) {
    //   for (int row = 0; row < grid.get_ho_region_num(); row++) {
    //     for (int col = 0; col < grid.get_ver_region_num(); col++) {
    //       region_data[layer][row][col] = evaluateRegion(grid, layer, row, col);
    //     }
    //   }
    // }

    return region_data;
  }

  SingleTemplate SimulatedAnnealing::selectRandomTemplate(bool is_horizontal)
  {
    // TODO: 从模板库中随机选择一个模板
    // 这里是示例实现，需要替换为实际调用

    // 随机生成线宽和间距
    // std::uniform_real_distribution<double> width_dist(0.5, 3.0);
    // std::uniform_real_distribution<double> spacing_dist(0.5, 3.0);

    // double wire_width = width_dist(_rng);
    // double wire_spacing = spacing_dist(_rng);

    SingleTemplate template_data;
    // template_data.set_wire_width(wire_width);
    // template_data.set_wire_spacing(wire_spacing);
    // template_data.set_is_horizontal(is_horizontal);

    return template_data;
  }

  SingleTemplate SimulatedAnnealing::selectSparseTemplate(bool is_horizontal)
  {
    // 生成较稀疏的模板 - 较小的线宽和较大的间距
    // std::uniform_real_distribution<double> width_dist(0.5, 1.5);
    // std::uniform_real_distribution<double> spacing_dist(2.0, 3.0);

    // double wire_width = width_dist(_rng);
    // double wire_spacing = spacing_dist(_rng);

    SingleTemplate template_data;
    // template_data.set_wire_width(wire_width);
    // template_data.set_wire_spacing(wire_spacing);
    // template_data.set_is_horizontal(is_horizontal);

    return template_data;
  }

  SingleTemplate SimulatedAnnealing::selectDenseTemplate(bool is_horizontal)
  {
    // 生成较密集的模板 - 较大的线宽和较小的间距
    // std::uniform_real_distribution<double> width_dist(2.0, 3.0);
    // std::uniform_real_distribution<double> spacing_dist(0.5, 1.5);

    // double wire_width = width_dist(_rng);
    // double wire_spacing = spacing_dist(_rng);

    SingleTemplate template_data;
    // template_data.set_wire_width(wire_width);
    // template_data.set_wire_spacing(wire_spacing);
    // template_data.set_is_horizontal(is_horizontal);

    return template_data;
  }

  double SimulatedAnnealing::normalizeIRDrop(double max_ir_drop, double min_ir_drop)
  {
    // 更新历史最值
    _max_ir_drop = std::max(_max_ir_drop, max_ir_drop);
    _min_ir_drop = std::min(_min_ir_drop, min_ir_drop);

    // Min-Max归一化到[0,1]区间
    if (_max_ir_drop - _min_ir_drop < 1e-10) {
      return 0.0; // 避免除以接近0的值
    }
    return (max_ir_drop - _min_ir_drop) / (_max_ir_drop - _min_ir_drop);
  }

  double SimulatedAnnealing::normalizeOverflow(int32_t overflow)
  {
    // 更新历史最值
    _max_overflow = std::max(_max_overflow, overflow);
    _min_overflow = std::min(_min_overflow, overflow);

    // Min-Max归一化到[0,1]区间
    if (_max_overflow - _min_overflow < 1e-10) {
      if (_max_overflow == _min_overflow && _max_overflow != 0) {
        // 只有一个非零样本点，返回一个中间值0.5
        return 0.5;
      }
      return 0; // 避免除以接近0的值
    }
    return (overflow - _min_overflow) / (_max_overflow - _min_overflow);
  }

  bool SimulatedAnnealing::isSameTemplate(const SingleTemplate& t1, const SingleTemplate& t2)
  {
    // 比较两个模板的所有关键属性
    return (t1.get_direction() == t2.get_direction() &&
      t1.get_width() == t2.get_width() &&
      t1.get_pg_offset() == t2.get_pg_offset() &&
      t1.get_space() == t2.get_space() &&
      t1.get_offset() == t2.get_offset());
  }

}  // namespace ipnp