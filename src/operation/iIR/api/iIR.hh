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
 * @file iIR.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief The top interface of the iIR tools.
 * @version 0.1
 * @date 2023-08-18
 *
 */

#pragma once

#include <map>
#include <string_view>
#include <vector>

namespace iir {

/**
 * @brief The instance power data.
 *
 * 
 */
struct IRInstancePower {
  const char* _instance_name;
  double _nominal_voltage;
  double _internal_power;
  double _switch_power;
  double _leakage_power;
  double _total_power;
};

/**
 * @brief The ir solver method enum.
 * 
 */
enum class IRSolverMethod {
  kLUSolver,  // LU solver
  kCGSolver,  // Conjugate Gradient solver
};

/**
 * @brief The IR top interface.
 * 
 */
class iIR {
 public:
  void set_rc_data(const void* rust_rc_data) { _rc_data = rust_rc_data; }
  auto* get_rc_data() { return _rc_data; }

  void set_nominal_voltage(double nominal_voltage) { _nominal_voltage = nominal_voltage; }
  double get_nominal_voltage() { return _nominal_voltage; }

  auto& get_instance_to_ir_drop() { return _instance_to_ir_drop; }

  unsigned init();
  unsigned readSpef(std::string_view spef_file_path);
  unsigned readInstancePowerDB(std::string_view instance_power_file_path);
  unsigned setInstancePowerData(std::vector<IRInstancePower> instance_power_data);

  unsigned solveIRDrop(const char* net_name);

 private:
  const void* _rc_data = nullptr;
  const void* _power_data = nullptr;

  double _nominal_voltage = 0.0;

  std::map<std::string, double> _instance_to_ir_drop;

  IRSolverMethod _solver_method = IRSolverMethod::kCGSolver;  //!< The IR solver method.
};
}  // namespace iir