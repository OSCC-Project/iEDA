/**
 * @file PythonReadVcd.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief the python api for read vcd
 * @version 0.1
 * @date 2023-09-05
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "PythonPower.hh"
#include "api/Power.hh"
#include "sta/Sta.hh"

namespace ipower {
bool read_vcd(std::string vcd_file, std::string top_instance_name) {
  ista::Sta* ista = ista::Sta::getOrCreateSta();
  ipower::Power* ipower = ipower::Power::getOrCreatePower(&(ista->get_graph()));

  return ipower->readVCD(vcd_file, top_instance_name);
}

}  // namespace ipower
