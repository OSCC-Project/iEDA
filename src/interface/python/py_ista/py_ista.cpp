#include "py_ista.h"

#include <tool_manager.h>

#include <sta/Sta.hh>

namespace python_interface {
bool staRun(const std::string& output)
{
  bool run_ok = iplf::tmInst->autoRunSTA(output);
  return run_ok;
}

bool staInit(const std::string& output)
{
  bool run_ok = iplf::tmInst->initSTA(output);
  return run_ok;
}

bool staReport(const std::string& output)
{
  bool run_ok = iplf::tmInst->runSTA(output);
  return run_ok;
}

bool setDesignWorkSpace(const std::string& design_workspace)
{
  auto* ista = ista::Sta::getOrCreateSta();
  ista->set_design_work_space(design_workspace.c_str());
  return true;
}

bool readVerilog(const std::string& file_name)
{
  auto* ista = ista::Sta::getOrCreateSta();

  ista->readVerilog(file_name.c_str());
  return true;
}

bool readLiberty(const std::string& file_name)
{
  auto* ista = ista::Sta::getOrCreateSta();
  ista->readLiberty(file_name.c_str());
  return true;
}

bool linkDesign(const std::string& cell_name)
{
  auto* ista = ista::Sta::getOrCreateSta();
  ista->set_top_module_name(cell_name.c_str());
  ista->linkDesign(cell_name.c_str());
  return true;
}

bool readSpef(const std::string& file_name)
{
  auto* ista = ista::Sta::getOrCreateSta();
  ista->readSpef(file_name.c_str());
  return true;
}

bool readSdc(const std::string& file_name)
{
  auto* ista = ista::Sta::getOrCreateSta();
  return ista->readSdc(file_name.c_str());
}

bool reportTiming(int digits, const std::string& delay_type, std::set<std::string> exclude_cell_names, bool derate)
{
  auto* ista = ista::Sta::getOrCreateSta();
  ista->buildGraph();
  ista->updateTiming();
  ista->reportTiming(std::move(exclude_cell_names), derate);
  return true;
}

}  // namespace python_interface