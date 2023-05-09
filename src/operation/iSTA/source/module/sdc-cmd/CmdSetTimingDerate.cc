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
 * @file CmdSetTimingDerate.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The sdc set_timing_derate cmd implemention.
 * @version 0.1
 * @date 2021-09-21
 */
#include "Cmd.hh"
#include "sdc/SdcConstrain.hh"
#include "sdc/SdcTimingDerate.hh"
#include "sta/Sta.hh"

namespace ista {

CmdSetTimingDerate::CmdSetTimingDerate(const char* cmd_name)
    : TclCmd(cmd_name) {
  auto* cell_delay_option = new TclSwitchOption("-cell_delay");
  addOption(cell_delay_option);

  auto* net_delay_option = new TclSwitchOption("-net_delay");
  addOption(net_delay_option);

  auto* early_option = new TclSwitchOption("-early");
  addOption(early_option);

  auto* late_option = new TclSwitchOption("-late");
  addOption(late_option);

  auto* clock_option = new TclSwitchOption("-clock");
  addOption(clock_option);

  auto* data_option = new TclSwitchOption("-data");
  addOption(data_option);

  auto* dynamic_option = new TclSwitchOption("-dynamic");
  addOption(dynamic_option);

  auto* static_option = new TclSwitchOption("-static");
  addOption(static_option);

  auto* derate_value_arg = new TclDoubleOption("derate_value", 1, 0.0);
  addOption(derate_value_arg);
}

/**
 * @brief The set_timing_derate cmd legally check.
 *
 * @return unsigned
 */
unsigned CmdSetTimingDerate::check() {
  // TODO(to taosimin) fix check
  return 1;
}

/**
 * @brief The set_timing_derate execute body.
 *
 * @return unsigned
 */
unsigned CmdSetTimingDerate::exec() {
  if (!check()) {
    return 0;
  }

  auto* cell_delay_option = getOptionOrArg("-cell_delay");
  auto* net_delay_option = getOptionOrArg("-net_delay");

  auto* clock_option = getOptionOrArg("-clock");
  auto* data_option = getOptionOrArg("-data");

  auto* early_option = getOptionOrArg("-early");
  auto* late_option = getOptionOrArg("-late");

  auto* derate_value_arg = getOptionOrArg("derate_value");
  LOG_FATAL_IF(!derate_value_arg);

  auto* set_timing_derate =
      new SdcTimingDerate(derate_value_arg->getDoubleVal());

  if (cell_delay_option->is_set_val() || !net_delay_option->is_set_val()) {
    set_timing_derate->set_is_cell_delay(true);
  }

  if (net_delay_option->is_set_val() || !cell_delay_option->is_set_val()) {
    set_timing_derate->set_is_net_delay(true);
  }

  if (clock_option->is_set_val() || !data_option->is_set_val()) {
    set_timing_derate->set_is_clock_delay(true);
  }

  if (data_option->is_set_val() || !clock_option->is_set_val()) {
    set_timing_derate->set_is_data_delay(true);
  }

  if (early_option->is_set_val() || !late_option->is_set_val()) {
    set_timing_derate->set_is_early_delay(true);
  }

  if (late_option->is_set_val() || !early_option->is_set_val()) {
    set_timing_derate->set_is_late_delay(true);
  }

  Sta* ista = Sta::getOrCreateSta();
  SdcConstrain* the_constrain = ista->getConstrain();

  the_constrain->addTimingDerate(set_timing_derate);

  return 1;
}

}  // namespace ista
