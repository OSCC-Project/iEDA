#include "RTAPI.hpp"
#include "tcl_egr.h"
#include "tcl_util.h"

namespace tcl {

TclRunEGR::TclRunEGR(const char* cmd_name) : TclCmd(cmd_name)
{
  _config_list.push_back(std::make_pair("-temp_directory_path", ValueType::kString));
  _config_list.push_back(std::make_pair("-thread_number", ValueType::kInt));
  _config_list.push_back(std::make_pair("-congestion_cell_x_pitch", ValueType::kInt));
  _config_list.push_back(std::make_pair("-congestion_cell_y_pitch", ValueType::kInt));
  _config_list.push_back(std::make_pair("-bottom_routing_layer", ValueType::kString));
  _config_list.push_back(std::make_pair("-top_routing_layer", ValueType::kString));
  _config_list.push_back(std::make_pair("-accuracy", ValueType::kInt));
  _config_list.push_back(std::make_pair("-skip_net_name_list", ValueType::kStringList));
  _config_list.push_back(std::make_pair("-strategy", ValueType::kString));

  TclUtil::addOption(this, _config_list);
}

unsigned TclRunEGR::exec()
{
  std::map<std::string, std::any> config_map = TclUtil::getConfigMap(this, _config_list);
  RTAPI_INST.runEGR(config_map);
  return 1;
}

}  // namespace tcl
