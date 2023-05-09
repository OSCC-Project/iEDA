/**
 * @file CmdSetClockGroups.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The set_clock_groups sdc cmd implemention.
 * @version 0.1
 * @date 2022-07-11
 */
#include "Cmd.hh"

namespace ista {

CmdSetClockGroups::CmdSetClockGroups(const char* cmd_name) : TclCmd(cmd_name) {
  auto* name_option = new TclStringOption("-name", 0, nullptr);
  addOption(name_option);

  auto* asynchronous_option = new TclSwitchOption("-asynchronous");
  addOption(asynchronous_option);

  auto* exclusive_option = new TclSwitchOption("-exclusive");
  addOption(exclusive_option);

  auto* group = new TclStringListListOption("-group", 0);
  addOption(group);
}

/**
 * @brief The set_clock_group cmd legally check.
 *
 * @return unsigned
 */
unsigned CmdSetClockGroups::check() { return 1; }

/**
 * @brief The set_clock_group execute body.
 *
 * @return unsigned
 */
unsigned CmdSetClockGroups::exec() {
  if (!check()) {
    return 0;
  }

  Sta* ista = Sta::getOrCreateSta();
  SdcConstrain* the_constrain = ista->getConstrain();
  Netlist* design_nl = ista->get_netlist();

  std::string group_name;
  auto* name_option = getOptionOrArg("-name");
  if (name_option->is_set_val()) {
    group_name = name_option->getStringVal();
  }

  TclOption* group_option = getOptionOrArg("-group");
  if (group_option->is_set_val()) {
    auto clock_groups = std::make_unique<SdcClockGroups>(std::move(group_name));
    auto group_list_list = group_option->getStringListList();
    for (auto&& group_list : group_list_list) {
      for (auto&& group_clock : group_list) {
        SdcClockGroup clock_group;
        auto clock_names =
            GetClockName(group_clock.c_str(), design_nl, the_constrain);
        for (auto& clock_name : clock_names) {
          clock_group.addClock(std::move(clock_name));
        }
        clock_groups->addClockGroup(std::move(clock_group));
      }
    }
    the_constrain->addClockGroups(std::move(clock_groups));
  }
  return 1;
}

}  // namespace ista
