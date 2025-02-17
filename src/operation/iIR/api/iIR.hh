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
 * @brief The IR top interface.
 * 
 */
class iIR {
 public:
  unsigned init();
  unsigned readSpef(std::string_view spef_file_path);
  unsigned readInstancePowerDB(std::string_view instance_power_file_path);
  unsigned setInstancePowerData(std::vector<IRInstancePower> instance_power_data);

  unsigned solveIRDrop(const char* net_name);

 private:
  const void* _rc_data = nullptr;
  const void* _power_data = nullptr;
};
}  // namespace iir