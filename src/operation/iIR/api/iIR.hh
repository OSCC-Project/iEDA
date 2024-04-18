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

namespace iir {
class iIR {
 public:
  unsigned readSpef(std::string_view spef_file_path);
  unsigned readInstancePowerDB(std::string_view instance_power_file_path);

  unsigned solveIRDrop(const char* net_name);

 private:
  const void* _rc_data = nullptr;
  const void* _power_data = nullptr;
};
}  // namespace iir