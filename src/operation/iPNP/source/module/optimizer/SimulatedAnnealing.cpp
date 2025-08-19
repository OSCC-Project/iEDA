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
 * @author JianrongSu
 * @brief
 * @version 1.0
 * @date 2025-06-23
 */

#include "SimulatedAnnealing.hh"

#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <fstream>
#include <iomanip>

#include "PNPConfig.hh"
#include "PNPIdbWrapper.hh"
#include "idm.h"

namespace ipnp {

SimulatedAnnealing::SimulatedAnnealing(double initial_temp, double cooling_rate, double min_temp, int iterations_per_temp,
                                       double ir_drop_weight, double overflow_weight)
    : _initial_temp(initial_temp),
      _cooling_rate(cooling_rate),
      _min_temp(min_temp),
      _iterations_per_temp(iterations_per_temp),
      _max_no_improvement(20),
      _ir_drop_weight(ir_drop_weight),
      _overflow_weight(overflow_weight),
      _max_ir_drop(std::numeric_limits<double>::min()),
      _min_ir_drop(std::numeric_limits<double>::max()),
      _max_overflow(std::numeric_limits<int32_t>::min()),
      _min_overflow(std::numeric_limits<int32_t>::max())
{
  std::random_device rd;
  _rng = std::mt19937(rd());
}

// Helper function to create directories
bool create_directories(const std::string& path)
{
  size_t pos = 0;
  std::string dir;
  int ret = 0;

  if (path[0] == '/') {
    pos = 1;
  }

  while ((pos = path.find('/', pos)) != std::string::npos) {
    dir = path.substr(0, pos++);
    if (dir.empty())
      continue;

    ret = mkdir(dir.c_str(), 0755);
    if (ret != 0 && errno != EEXIST) {
      return false;
    }
  }
  return true;
}

OptimizationResult SimulatedAnnealing::optimize(const PNPGridManager& initial_grid)
{
  PNPGridManager current_solution = initial_grid;
  PNPGridManager best_solution = current_solution;

  CostResult current_result = evaluateCost(current_solution, current_solution);
  double current_cost = current_result.cost;
  double best_cost = current_cost;

  // Get report_path directory from config
  std::string report_path = "SA_report.txt";

  // Use the passed config object

  std::string dir_path = pnpConfig->get_report_path();

  if (dir_path.back() != '/') {
    dir_path += '/';
  }

  report_path = dir_path + "SA_report.txt";
  LOG_INFO << "Using directory from config for SA report: " << dir_path << std::endl;

  LOG_INFO << "Report will be written to: " << report_path << std::endl;

  // Ensure the directory for the report file exists
  std::string directory = report_path.substr(0, report_path.find_last_of('/'));
  if (!directory.empty()) {
    if (!create_directories(directory)) {
      LOG_ERROR << "Failed to create directory: " << directory << ", error: " << strerror(errno) << std::endl;
    } else {
      LOG_INFO << "Directory ensured: " << directory << std::endl;
    }
  }

  // Check if the file exists
  struct stat buffer;
  bool file_exists = (stat(report_path.c_str(), &buffer) == 0);

  if (file_exists) {
    LOG_INFO << "Report file already exists, will overwrite it" << std::endl;
  } else {
    LOG_INFO << "Report file does not exist, will create it" << std::endl;
  }

  // Open report file for writing with explicit mode
  std::ofstream report_file(report_path.c_str(), std::ios::out | std::ios::trunc);
  if (!report_file.is_open()) {
    LOG_ERROR << "Failed to open report file for writing: " << report_path << ", error: " << strerror(errno) << std::endl;
    return {best_solution, best_cost};
  } else {
    LOG_INFO << "Successfully opened report file for writing" << std::endl;
  }

  // Set formatting for floating point numbers
  report_file << std::fixed << std::setprecision(6);

  // Write header and initial solution information
  report_file << "======================================================================" << std::endl;
  report_file << "                     SIMULATED ANNEALING REPORT                       " << std::endl;
  report_file << "======================================================================" << std::endl;
  report_file << std::endl;
  report_file << "INITIAL SOLUTION:" << std::endl;
  report_file << "  IR Drop          : " << std::setw(12) << current_result.ir_drop << std::endl;
  report_file << "  Normalized IR    : " << std::setw(12) << current_result.normalized_ir_drop << std::endl;
  report_file << "  Overflow         : " << std::setw(12) << current_result.overflow << std::endl;
  report_file << "  Normalized OF    : " << std::setw(12) << current_result.normalized_overflow << std::endl;
  report_file << "  Total Cost       : " << std::setw(12) << current_cost << std::endl;
  report_file << "  IR Weight        : " << std::setw(12) << _ir_drop_weight << std::endl;
  report_file << "  Overflow Weight  : " << std::setw(12) << _overflow_weight << std::endl;
  report_file << std::endl;
  report_file << "----------------------------------------------------------------------" << std::endl;
  report_file << std::endl;

  LOG_INFO << "Initial solution: " << std::endl
           << "  IR Drop: " << current_result.ir_drop << ", Normalized IR: " << current_result.normalized_ir_drop
           << ", Overflow: " << current_result.overflow << ", Normalized OF: " << current_result.normalized_overflow
           << ", Total Cost: " << current_cost << " (IR weight:" << _ir_drop_weight << ", OF weight:" << _overflow_weight << ")"
           << std::endl;

  double temperature = _initial_temp;
  int no_improvement_count = 0;
  int total_iterations = 0;

  // Outer loop
  while (!shouldTerminate(total_iterations, temperature, no_improvement_count)) {
    // Inner loop
    for (int i = 0; i < _iterations_per_temp; i++) {
      // Generate neighboring solution
      PNPGridManager new_solution = generateNeighbor(current_solution);

      // Evaluate new solution
      CostResult new_result = evaluateCost(new_solution, current_solution);
      double new_cost = new_result.cost;

      // Write iteration information to report file
      report_file << "ITERATION: " << std::setw(5) << total_iterations << " | TEMPERATURE: " << std::setw(12) << temperature << std::endl;
      report_file << "  IR Drop          : " << std::setw(12) << new_result.ir_drop << std::endl;
      report_file << "  Normalized IR    : " << std::setw(12) << new_result.normalized_ir_drop << std::endl;
      report_file << "  Overflow         : " << std::setw(12) << new_result.overflow << std::endl;
      report_file << "  Normalized OF    : " << std::setw(12) << new_result.normalized_overflow << std::endl;
      report_file << "  Total Cost       : " << std::setw(12) << new_cost << std::endl;

      LOG_INFO << "Iter[" << total_iterations << "] Temp[" << temperature << "]";
      LOG_INFO << "  IR Drop: " << new_result.ir_drop;
      LOG_INFO << "  Normalized IR: " << new_result.normalized_ir_drop;
      LOG_INFO << "  Overflow: " << new_result.overflow;
      LOG_INFO << "  Normalized OF: " << new_result.normalized_overflow;
      LOG_INFO << "  Total Cost: " << new_cost;
      LOG_INFO << "  IR weight:" << _ir_drop_weight << ", OF weight:" << _overflow_weight;

      // Acceptance criteria
      if (acceptSolution(current_cost, new_cost, temperature)) {
        current_solution = new_solution;
        current_cost = new_cost;

        // Save to IDB only after accepting the solution
        PNPIdbWrapper wrapper;
        wrapper.saveToIdb(current_solution);
        // wrapper.writeIdbToDef("/home/sujianrong/iEDA/src/operation/iPNP/data/case/s9234/result/debug.def");

        LOG_INFO << "  Solution ACCEPTED! New current cost: " << current_cost << std::endl;
        report_file << "  STATUS           : ACCEPTED" << std::endl;
        report_file << "  New Current Cost : " << std::setw(12) << current_cost << std::endl;

        // Update global best solution
        if (new_cost < best_cost) {
          best_solution = new_solution;
          best_cost = new_cost;
          no_improvement_count = 0;
          LOG_INFO << "  NEW BEST SOLUTION! Cost: " << best_cost << std::endl;
          LOG_INFO << "  IR Drop: " << new_result.ir_drop << std::endl;
          LOG_INFO << "  Overflow: " << new_result.overflow << std::endl;

          report_file << "  BEST SOLUTION    : YES" << std::endl;
          report_file << "  Best Cost        : " << std::setw(12) << best_cost << std::endl;
        } else {
          LOG_INFO << "  Solution accepted but not better than best. No change to best solution." << current_cost << std::endl;
          report_file << "  BEST SOLUTION    : NO" << std::endl;
          no_improvement_count++;
        }
      } else {
        report_file << "  STATUS           : REJECTED" << std::endl;
        no_improvement_count++;
      }

      report_file << "----------------------------------------------------------------------" << std::endl;
      report_file << std::endl;

      total_iterations++;
    }

    // Cooling
    temperature *= _cooling_rate;
    LOG_INFO << "===== Temperature: " << temperature << ", Current cost: " << current_cost << ", Best cost so far: " << best_cost
             << ", No improvement count: " << no_improvement_count << " =====" << std::endl;

    report_file << "=====================================================================" << std::endl;
    report_file << "TEMPERATURE UPDATE: " << std::setw(12) << temperature << std::endl;
    report_file << "  Current Cost     : " << std::setw(12) << current_cost << std::endl;
    report_file << "  Best Cost        : " << std::setw(12) << best_cost << std::endl;
    report_file << "  No Improvement   : " << std::setw(12) << no_improvement_count << std::endl;
    report_file << "=====================================================================" << std::endl;
    report_file << std::endl;

    LOG_INFO << "Temperature: " << temperature << ", Best cost: " << best_cost;
  }

  // Write final results
  report_file << "======================================================================" << std::endl;
  report_file << "                     FINAL OPTIMIZATION RESULTS                       " << std::endl;
  report_file << "======================================================================" << std::endl;
  report_file << "  Best Cost        : " << std::setw(12) << best_cost << std::endl;
  report_file << "  Total Iterations : " << std::setw(12) << total_iterations << std::endl;
  report_file << "  Final Temperature: " << std::setw(12) << temperature << std::endl;
  report_file << "======================================================================" << std::endl;

  // Close file
  report_file.close();

  // Return optimization results
  OptimizationResult result;
  result.best_grid = best_solution;
  result.best_cost = best_cost;

  return result;
}

PNPGridManager SimulatedAnnealing::generateNeighbor(const PNPGridManager& current)
{
  PNPGridManager neighbor = current;

  int modifiable_min_layer = pnpConfig->get_sa_modifiable_layer_min();
  int modifiable_max_layer = pnpConfig->get_sa_modifiable_layer_max();

  // Use configured range to generate random layer
  std::uniform_int_distribution<int> layer_dist(modifiable_min_layer, modifiable_max_layer);
  std::uniform_int_distribution<int> row_dist(0, current.get_ho_region_num() - 1);
  std::uniform_int_distribution<int> col_dist(0, current.get_ver_region_num() - 1);

  int layer = layer_dist(_rng);
  int row = row_dist(_rng);
  int col = col_dist(_rng);

  // Get current template information
  const auto& old_template = current.get_template_data()[layer][row][col];

  // Print current template information
  LOG_INFO << "Attempting to change template at: Layer M" << current.get_power_layers()[layer] << " (idx:" << layer << "), Region[" << row
           << "][" << col << "]" << std::endl;
  LOG_INFO << "  Old template information: " << std::endl;
  if (old_template.get_direction() == StripeDirection::kHorizontal) {
    LOG_INFO << "direction: Horizontal" << std::endl;
  } else {
    LOG_INFO << "direction: Vertical" << std::endl;
  }
  LOG_INFO << "width: " << old_template.get_width() << std::endl;
  LOG_INFO << "pg_offset: " << old_template.get_pg_offset() << std::endl;
  LOG_INFO << "space: " << old_template.get_space() << std::endl;
  LOG_INFO << "offset: " << old_template.get_offset() << std::endl;

  SingleTemplate new_template;
  std::vector<SingleTemplate> vertical_templates = current.get_vertical_templates();
  std::vector<SingleTemplate> horizontal_templates = current.get_horizontal_templates();

  // Randomly select a new template
  bool is_vertical = (layer % 2 == 0);  // M9 has index 0, which is even, so it's Vertical
  if (is_vertical) {
    size_t old_idx = 0;
    for (size_t i = 0; i < vertical_templates.size(); i++) {
      if (isSameTemplate(old_template, vertical_templates[i])) {
        old_idx = i;
        break;
      }
    }
    std::uniform_int_distribution<size_t> template_dist(0, vertical_templates.size() - 2);
    size_t template_idx = template_dist(_rng);
    if (template_idx >= old_idx) {
      template_idx++;
    }
    new_template = vertical_templates[template_idx];
  } else {
    size_t old_idx = 0;
    bool found = false;
    for (size_t i = 0; i < horizontal_templates.size(); i++) {
      if (isSameTemplate(old_template, horizontal_templates[i])) {
        old_idx = i;
        found = true;
        break;
      }
    }
    std::uniform_int_distribution<size_t> template_dist(0, horizontal_templates.size() - 2);
    size_t template_idx = template_dist(_rng);
    if (found && template_idx >= old_idx) {
      template_idx++;
    }
    new_template = horizontal_templates[template_idx];
  }

  // Print new template information
  LOG_INFO << "New template information: " << std::endl;
  if (new_template.get_direction() == StripeDirection::kHorizontal) {
    LOG_INFO << "direction: Horizontal" << std::endl;
  } else {
    LOG_INFO << "direction: Vertical" << std::endl;
  }
  LOG_INFO << "width: " << new_template.get_width() << std::endl;
  LOG_INFO << "pg_offset: " << new_template.get_pg_offset() << std::endl;
  LOG_INFO << "space: " << new_template.get_space() << std::endl;
  LOG_INFO << "offset: " << new_template.get_offset() << std::endl;

  neighbor.set_single_template(layer, row, col, new_template);
  return neighbor;
}

CostResult SimulatedAnnealing::evaluateCost(const PNPGridManager& new_solution, const PNPGridManager& current_solution)
{
  PNPIdbWrapper temp_wrapper;
  temp_wrapper.saveToIdb(new_solution);

  _cong_eval.evalEGR();
  int32_t overflow = _cong_eval.get_total_overflow_union();

  _ir_eval.runIREval();
  double max_ir_drop = _ir_eval.getMaxIRDrop();
  double min_ir_drop = _ir_eval.getMinIRDrop();
  double avg_ir_drop = _ir_eval.getAvgIRDrop();

  double normalized_ir_drop = normalizeIRDrop(max_ir_drop, min_ir_drop, avg_ir_drop);
  double normalized_overflow = normalizeOverflow(overflow);
  double cost = _ir_drop_weight * normalized_ir_drop + _overflow_weight * normalized_overflow;

  // New solution not yet accepted, restore current solution
  temp_wrapper.saveToIdb(current_solution);

  return {cost, max_ir_drop, overflow, normalized_ir_drop, normalized_overflow};
}

bool SimulatedAnnealing::acceptSolution(double current_cost, double new_cost, double temp)
{
  if (new_cost < current_cost) {
    return true;
  }

  // Calculate acceptance probability according to Metropolis criterion
  // Probability formula: P = exp(-ΔE/T), where ΔE is the cost difference and T is the current temperature
  // Higher temperature = higher probability of accepting worse solutions; lower temperature = more likely to only accept better solutions
  double delta = new_cost - current_cost;
  double acceptance_probability = std::exp(-delta / temp);

  // Accept new solution if random number is less than acceptance probability
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(_rng) < acceptance_probability;
}

bool SimulatedAnnealing::shouldTerminate(int iterations, double temp, int no_improvement_count)
{
  // Temperature below threshold
  if (temp < _min_temp)
    return true;

  // Multiple consecutive iterations with no improvement
  if (no_improvement_count > _max_no_improvement)
    return true;

  return false;
}

double SimulatedAnnealing::normalizeIRDrop(double max_ir_drop, double min_ir_drop, double avg_ir_drop)
{
  _max_ir_drop = std::max(_max_ir_drop, max_ir_drop);
  _min_ir_drop = std::min(_min_ir_drop, min_ir_drop);

  // Min-Max normalization to [0,1] range
  if (_max_ir_drop - _min_ir_drop < 1e-10) {
    return 0.0;  // Avoid division by near-zero value
  }
  return (avg_ir_drop - _min_ir_drop) / (_max_ir_drop - _min_ir_drop);
}

double SimulatedAnnealing::normalizeOverflow(int32_t overflow)
{
  _max_overflow = std::max(_max_overflow, overflow);
  _min_overflow = std::min(_min_overflow, overflow);

  // Min-Max normalization to [0,1] range
  if (_max_overflow - _min_overflow < 1e-10) {
    if (_max_overflow == _min_overflow && _max_overflow != 0) {
      // Only one non-zero sample point, return a middle value of 0.5
      return 0.5;
    }
    return 0;  // Avoid division by near-zero value
  }
  return static_cast<double>(overflow - _min_overflow) / static_cast<double>(_max_overflow - _min_overflow);
}

bool SimulatedAnnealing::isSameTemplate(const SingleTemplate& t1, const SingleTemplate& t2)
{
  // Compare all key attributes of the two templates
  return (t1.get_direction() == t2.get_direction() && t1.get_width() == t2.get_width() && t1.get_pg_offset() == t2.get_pg_offset()
          && t1.get_space() == t2.get_space() && t1.get_offset() == t2.get_offset());
}

}  // namespace ipnp