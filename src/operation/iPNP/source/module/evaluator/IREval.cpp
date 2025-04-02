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
 * @file IREval.cpp
 * @author Xinhao li
 * @brief
 * @version 0.1
 * @date 2024-07-15
 */

#include "IREval.hh"
#include "api/PowerEngine.hh"
#include "log/Log.hh"

namespace ipnp {

/**
 * @brief 运行IR分析
 * 
 * @param power_net_name 电源网络名称，默认为VDD
 * @return unsigned 分析结果
 */
unsigned IREval::runIRAnalysis(const std::string& power_net_name) {
  try {
    // 获取PowerEngine单例
    ipower::PowerEngine* power_engine = ipower::PowerEngine::getOrCreatePowerEngine();
    
    // 重置IR分析数据
    power_engine->resetIRAnalysisData();
    
    // 构建电源网络拓扑
    power_engine->buildPGNetWireTopo();
    
    // 运行IR分析
    unsigned result = power_engine->runIRAnalysis(power_net_name);
    
    // 获取IR分布图
    auto ir_drop_map = power_engine->displayIRDropMap();
    
    // 转换坐标格式
    std::map<std::pair<double, double>, double> coordinate_ir_map;
    for (const auto& [coord, ir_drop] : ir_drop_map) {
      coordinate_ir_map[{coord.first, coord.second}] = ir_drop;
    }
    
    // 处理IR分析结果
    processIRResults(coordinate_ir_map);
    
    return result;
  } catch (const std::exception& e) {
    LOG_ERROR << "Error during IR analysis: " << e.what();
    return 0;
  }
}

/**
 * @brief 处理IR分析结果
 * 
 * @param ir_drop_map IR电压降映射
 */
void IREval::processIRResults(const std::map<std::pair<double, double>, double>& ir_drop_map) {
  // 清空原有IR map
  ir_map.clear();
  
  // 如果没有数据，直接返回
  if (ir_drop_map.empty()) {
    ir_value = 0.0;
    return;
  }
  
  // 计算IR平均值
  double total_ir = 0.0;
  int count = 0;
  
  // 找出坐标范围
  double min_x = std::numeric_limits<double>::max();
  double max_x = std::numeric_limits<double>::min();
  double min_y = std::numeric_limits<double>::max();
  double max_y = std::numeric_limits<double>::min();
  
  for (const auto& [coord, ir_drop] : ir_drop_map) {
    min_x = std::min(min_x, coord.first);
    max_x = std::max(max_x, coord.first);
    min_y = std::min(min_y, coord.second);
    max_y = std::max(max_y, coord.second);
    
    total_ir += ir_drop;
    count++;
  }
  
  // 计算平均IR drop
  if (count > 0) {
    ir_value = total_ir / count;
  } else {
    ir_value = 0.0;
  }
  
  // 后续可以根据需要将点数据转换为网格数据
  // 例如创建一个二维网格来表示IR分布
  // 这里简单实现为示例
  const int grid_size = 100;  // 网格大小
  
  // 初始化网格
  ir_map.resize(grid_size, std::vector<double>(grid_size, 0.0));
  
  // 计算网格单元大小
  double cell_width = (max_x - min_x) / grid_size;
  double cell_height = (max_y - min_y) / grid_size;
  
  if (cell_width <= 0 || cell_height <= 0) {
    return;  // 避免除以零
  }
  
  // 填充网格
  for (const auto& [coord, ir_drop] : ir_drop_map) {
    int grid_x = static_cast<int>((coord.first - min_x) / cell_width);
    int grid_y = static_cast<int>((coord.second - min_y) / cell_height);
    
    // 确保在网格范围内
    if (grid_x >= 0 && grid_x < grid_size && grid_y >= 0 && grid_y < grid_size) {
      ir_map[grid_y][grid_x] = ir_drop;
    }
  }
}

}  // namespace ipnp