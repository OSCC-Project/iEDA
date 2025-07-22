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
 * @file CmdSetUnits.cc
 * @author caijianfeng (caijianfeng@foxmail.com)
 * @brief support `set units` command in sdc
 * @version 0.1
 * @date 2023-09-08
 */

#include <cstring>

#include "Cmd.hh"

namespace ista {

CmdSetUnits::CmdSetUnits(const char* cmd_name) : TclCmd(cmd_name) {
  auto* time_unit = new TclStringOption("-time", 0, "NS");
  addOption(time_unit);
  auto* cap_unit = new TclStringOption("-capacitance", 0, "PF");
  addOption(cap_unit);
  auto* current_unit = new TclStringOption("-current", 0, "mA");
  addOption(current_unit);
  auto* voltage_unit = new TclStringOption("-voltage", 0, "V");
  addOption(voltage_unit);
  auto* res_unit = new TclStringOption("-resistance", 0, "kOhm");
  addOption(res_unit);
}

unsigned CmdSetUnits::check() {
  TclOption* time_unit = getOptionOrArg("-time");
  TclOption* cap_unit = getOptionOrArg("-capacitance");
  LOG_FATAL_IF(!time_unit);
  LOG_FATAL_IF(!cap_unit);
  if (!(time_unit->is_set_val() || cap_unit->is_set_val())) {
    LOG_ERROR << "'time unit' and 'capacitance unit' are missing.";
    return 0;
  }

  const char* time_opt = time_unit->getStringVal();
  const char* cap_opt = cap_unit->getStringVal();

  if (!isOptionValid("time", time_opt) ||
      !isOptionValid("capacitance", cap_opt)) {
    LOG_ERROR << "time unit or capacitance unit not valid" << std::endl
              << "valid time units: NS, PS, FS " << std::endl
              << "valid capacitance unitd: F, PF, FF " << std::endl;
    return 0;
  }

  return 1;
}

unsigned CmdSetUnits::exec() {
  if (!check()) {
    return 0;
  }

  auto* time_unit_option = getOptionOrArg("-time");
  auto* cap_unit_option = getOptionOrArg("-capacitance");

  Sta* ista = Sta::getOrCreateSta();

  if (time_unit_option->is_set_val()) {
    std::map<std::string, TimeUnit> timeUnitMap = {
        {"PS", TimeUnit::kPS}, {"FS", TimeUnit::kFS}, {"NS", TimeUnit::kNS}};

    ista->setTimeUnit(
        timeUnitMap[Str::toUpper(time_unit_option->getStringVal())]);
    LOG_INFO << "set ista time unit" << std::endl;
  }
  if (cap_unit_option->is_set_val()) {
    std::map<std::string, CapacitiveUnit> capUnitMap = {
        {"PF", CapacitiveUnit::kPF},
        {"FF", CapacitiveUnit::kFF},
        {"F", CapacitiveUnit::kF}};

    ista->setCapUnit(capUnitMap[Str::toUpper(cap_unit_option->getStringVal())]);
    LOG_INFO << "set ista capacitance unit" << std::endl;
  }
  return 1;
}

bool CmdSetUnits::isOptionValid(const char* unit_type, std::string inputStr) {
  if (Str::noCaseEqual(unit_type, "time") &&
      (Str::noCaseEqual(inputStr.c_str(), "ps") ||
       Str::noCaseEqual(inputStr.c_str(), "ns") ||
       Str::noCaseEqual(inputStr.c_str(), "fs")))
    return true;
  if (Str::noCaseEqual(unit_type, "capacitance") &&
      (Str::noCaseEqual(inputStr.c_str(), "f") ||
       Str::noCaseEqual(inputStr.c_str(), "pf") ||
       Str::noCaseEqual(inputStr.c_str(), "ff")))
    return true;
  return false;
};

}  // namespace ista