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
 * @file SdcAllPorts.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The file is the all inputs or outputs class of sdc.
 * @version 0.1
 * @date 2024-02-06
 */

#pragma once

#include "SdcCommand.hh"
#include "netlist/DesignObject.hh"
#include "netlist/Netlist.hh"

namespace ista {

/**
 * @brief The class for all_inputs collection.
 *
 */
class SdcAllInputPorts : public SdcCommandObj {
 public:
  SdcAllInputPorts() = default;
  ~SdcAllInputPorts() override = default;

  void addPort(Port* input_port) { _input_ports.push_back(input_port); }
  auto& get_input_ports() { return _input_ports; }

  unsigned isAllInputPorts() override { return 1; }

 private:
  std::vector<Port*> _input_ports;  //!< The input ports.
};

/**
 * @brief The class for all_outputs collection.
 *
 */
class SdcAllOutputPorts : public SdcCommandObj {
 public:
  SdcAllOutputPorts() = default;
  ~SdcAllOutputPorts() override = default;

  void addPort(Port* output_port) { _output_ports.push_back(output_port); }
  auto& get_output_ports() { return _output_ports; }

  unsigned isAllOutputPorts() override { return 1; }

 private:
  std::vector<Port*> _output_ports;  //!< The input ports.
};

}  // namespace ista