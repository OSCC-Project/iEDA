
#ifndef IDRC_SRC_CFG_DRC_CONFIG_H_
#define IDRC_SRC_CFG_DRC_CONFIG_H_

#include <map>
#include <string>
#include <vector>

namespace idrc {

class DrcConfig
{
 public:
  DrcConfig() {}
  ~DrcConfig() {}

  // getter
  std::vector<std::string>& get_lef_paths() { return _lef_paths; }
  std::string& get_def_path() { return _def_path; }
  // OUTPUT
  std::string& get_output_dir_path() { return _output_dir_path; }

  // setter
  // INPUT
  void set_lef_paths(const std::vector<std::string>& lef_paths) { _lef_paths = lef_paths; }
  void set_def_path(const std::string& def_path) { _def_path = def_path; }
  // OUTPUT
  void set_output_dir_path(const std::string& output_dir_path) { _output_dir_path = output_dir_path; }
  // function

 private:
  // INPUT
  std::vector<std::string> _lef_paths;
  std::string _def_path;
  // OUTPUT
  std::string _output_dir_path;
};

}  // namespace idrc
#endif  // IDR_SRC_CFG_CONFIG_H_