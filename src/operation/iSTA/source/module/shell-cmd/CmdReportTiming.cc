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
 * @file CmdReportTiming.cc
 * @author Wang Hao (harry0789@qq.com)
 * @brief
 * @version 0.1
 * @date 2021-10-12
 */
#include "ShellCmd.hh"
#include "json/json.hpp"
#include "sta/Sta.hh"

namespace ista {
CmdReportTiming::CmdReportTiming(const char* cmd_name) : TclCmd(cmd_name) {
  auto* digits_option = new TclIntOption("-digits", 0, 3);
  addOption(digits_option);
  auto* path_delay_option = new TclStringOption("-delay_type", 0, "max_min");
  addOption(path_delay_option);
  auto* exclude_cell_names_option =
      new TclStringListOption("-exclude_cell_names", 0, {});
  addOption(exclude_cell_names_option);
  auto* derate_option = new TclSwitchOption("-derate");
  addOption(derate_option);
  auto* is_clock_cap_option = new TclSwitchOption("-is_clock_cap");
  addOption(is_clock_cap_option);
  auto* is_snappot_option = new TclSwitchOption("-is_not_bak_rpt");
  addOption(is_snappot_option);

  auto* max_paths_option = new TclIntOption("-max_path", 0, 3);
  addOption(max_paths_option);

  auto* n_worst_option = new TclIntOption("-nworst", 0, 1);
  addOption(n_worst_option);

  auto* from_option = new TclStringListOption("-from", 0, {});
  addOption(from_option);

  auto* to_option = new TclStringListOption("-to", 0, {});
  addOption(to_option);

  auto* through_option = new TclStringListListOption("-through", 0);
  addOption(through_option);

  auto* help_option = new TclSwitchOption("-help");
  addOption(help_option);

  auto* json_report_option = new TclSwitchOption("-json");
  addOption(json_report_option);
}

unsigned CmdReportTiming::check() { return 1; }

unsigned CmdReportTiming::printHelp() {
  std::string help_msg = R"(  
  Report timing path, Usage: 

    report_timing [options]  

  Options:
    [-delay-type <string>]        : Specify the type of path to report. 
                                    Legal Values: max, min, max_min, default: max_min

    [-digits <int>]               : The significant digits the report show.
                                    default: 3

    [-max_path <int>]             : The max timing path per clock.
                                    default: 3

    [-nworst <int>]               : The max timing path per endpoint.
                                    default: 1

    [-from <string list>]         : Specify the timing path start points name.
                                    example: report_timing -from {dpath/b_reg/_55_:CK}

    [-to <string list>]           : Specify the timing path end points name.
                                    example: report_timing -from {dpath/a_reg/_55_:D}

    [-through <string list list>] : Specify the timing path through points.
                                    example: report_timing -through {dpath/a_reg/_55_:Q dpath/a_reg/_39_:A} -through {dpath/a_reg/_39_:B}

    [-json]                       : Output report in JSON format.
                                    example: report_timing -json
  
  )";

  std::cout << help_msg << std::endl;
  return 1;
}

unsigned CmdReportTiming::exec() {
  if (!check()) {
    return 0;
  }

  TclOption* help_option = getOptionOrArg("-help");
  if (help_option->is_set_val()) {
    printHelp();
    return 1;
  }

  TclOption* delay_type_option = getOptionOrArg("-delay_type");
  char* delay_type;
  if (delay_type_option) {
    delay_type = delay_type_option->getStringVal();
  }

  TclOption* exclude_cell_names_option = getOptionOrArg("-exclude_cell_names");
  std::set<std::string> new_exclude_cell_names;
  if (exclude_cell_names_option) {
    auto exclude_cell_names = exclude_cell_names_option->getStringList();
    for (const auto& exclude_cell_name : exclude_cell_names) {
      new_exclude_cell_names.insert(exclude_cell_name);
    }
  }

  TclOption* derate_option = getOptionOrArg("-derate");
  bool is_derate = false;
  if (derate_option->is_set_val()) {
    is_derate = true;
  }

  TclOption* is_clock_cap_option = getOptionOrArg("-is_clock_cap");
  bool is_clock_cap = false;
  if (is_clock_cap_option->is_set_val()) {
    is_clock_cap = true;
  }

  TclOption* is_not_bak_rpt_option = getOptionOrArg("-is_not_bak_rpt");
  bool is_not_bak_rpt = true;
  if (is_not_bak_rpt_option->is_set_val()) {
    is_not_bak_rpt = false;
  }

  Sta* ista = Sta::getOrCreateSta();

  auto* digits_option = getOptionOrArg("-digits");
  unsigned num_digits = digits_option->getIntVal();
  ista->set_significant_digits(num_digits);

  if (delay_type) {
    Str::equal(delay_type, "max_min")
        ? ista->set_analysis_mode(AnalysisMode::kMaxMin)
        : Str::equal(delay_type, "max")
              ? ista->set_analysis_mode(AnalysisMode::kMax)
              : ista->set_analysis_mode(AnalysisMode::kMin);
  }

  auto* max_path_option = getOptionOrArg("-max_path");
  unsigned max_path = max_path_option->getIntVal();
  ista->set_n_worst_path_per_clock(max_path);

  auto* nworst_option = getOptionOrArg("-nworst");
  unsigned nworst = nworst_option->getIntVal();
  ista->set_n_worst_path_per_endpoint(nworst);

  {
    auto* from_option = getOptionOrArg("-from");
    auto from_list = from_option->getStringList();

    auto* through_option = getOptionOrArg("-through");
    auto through_list_list = through_option->getStringListList();

    auto* to_option = getOptionOrArg("-to");
    auto to_list = to_option->getStringList();

    ista->setReportSpec(std::move(from_list), std::move(through_list_list),
                        std::move(to_list));
  }

  auto* json_report_option = getOptionOrArg("-json");
  if (json_report_option->is_set_val()) {
    ista->enableJsonReport();
  }

  ista->buildGraph();

  ista->updateTiming();
  ista->reportTiming(std::move(new_exclude_cell_names), is_derate, is_clock_cap,
                     is_not_bak_rpt);
  // ista->dumpNetlistData();
  return 1;
}

}  // namespace ista