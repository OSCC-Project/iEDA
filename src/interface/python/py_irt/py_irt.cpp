#include "py_irt.h"

#include <tcl_util.h>

#include <iRT/api/RTAPI.hpp>
#include <string>
namespace python_interface {

bool destroyRT()
{
  RTAPIInst.destroyRT();
  return true;
}

bool runDR()
{
  RTAPIInst.runDR();
  return true;
}

bool runGR()
{
  RTAPIInst.runGR();
  return true;
}

bool runRT()
{
  RTAPIInst.runRT();
  return true;
}

bool initConfigMapByDict(std::map<std::string, std::string>& config_dict, std::map<std::string, std::any>& config_map);
bool initConfigMapByJSON(const std::string& config, std::map<std::string, std::any>& config_map);

bool initRT(std::string& config, std::map<std::string, std::string>& config_dict)
{
  std::map<std::string, std::any> config_map;

  bool pass = false;
  pass = !pass ? initConfigMapByJSON(config, config_map) : pass;
  pass = !pass ? initConfigMapByDict(config_dict, config_map) : pass;
  if (!pass) {
    return false;
  }
  RTAPIInst.initRT(config_map);
  return true;
}

}  // namespace python_interface
