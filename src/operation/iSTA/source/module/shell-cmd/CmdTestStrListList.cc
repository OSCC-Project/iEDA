#include "ShellCmd.hh"

namespace ista {

CmdTESTSLL::CmdTESTSLL(const char* cmd_name) : TclCmd(cmd_name) {
  auto* strll_option = new TclStringListListOption("-puts", 0);
  addOption(strll_option);
}

unsigned CmdTESTSLL::check() {
  return 1;
}

unsigned CmdTESTSLL::exec() {
  TclOption* strll_option = getOptionOrArg("-puts");
  if (strll_option->is_set_val()) {
    auto str_list_list = strll_option->getStringListList();
    for (auto&& str_list : str_list_list) {
      std::cout << "< " ;
      for (auto&& str : str_list) {
        std::cout << "\"" << str << "\" ";
      }
      std::cout << ">\n";
    }
  }

  return 1;
}
}