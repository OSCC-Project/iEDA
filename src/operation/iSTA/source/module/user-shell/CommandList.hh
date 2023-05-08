#pragma once
#include <set>
#include <string>
namespace ista {

class Command {
 public:
  Command(const char* str, int len) { _cmd = new char[len]; }

  // get copy of string before first separator
  Command(const char* str, const char separator) {
    size_t i;
    for (i = 0; (str[i] != separator) && (str[i] != '\n'); ++i) {
    }
    _cmd = new char[i + 1];
    strncpy(_cmd, str, i);
    _cmd[i] = '\0';
  }

  ~Command() { delete[] _cmd; }

  const char* get_cmd() const { return _cmd; }

 private:
  char* _cmd;
};

const static std::set<std::string> cmd_list = {
    "read_liberty", "read_verilog",  "link_design",        "read_spef",
    "read_sdc",     "report_checks", "report_check_types",
};

bool isCommandValid(const char* cmd) {
  Command temp_cmd(cmd, ' ');
  auto cmd_it = cmd_list.find(temp_cmd.get_cmd());
  return cmd_it != cmd_list.end();
}
}  // namespace ista