/*
 * @FilePath: wirelength_api.cpp
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */
#include "wirelength_api.h"

#include "wirelength_eval.h"

namespace ieval {

#define EVAL_WIRELENGTH_INST (ieval::WirelengthEval::getInst())

WirelengthAPI* WirelengthAPI::_wirelength_api_inst = nullptr;

WirelengthAPI::WirelengthAPI()
{
}

WirelengthAPI::~WirelengthAPI()
{
}

WirelengthAPI* WirelengthAPI::getInst()
{
  if (_wirelength_api_inst == nullptr) {
    _wirelength_api_inst = new WirelengthAPI();
  }

  return _wirelength_api_inst;
}

void WirelengthAPI::destroyInst()
{
  if (_wirelength_api_inst != nullptr) {
    delete _wirelength_api_inst;
    _wirelength_api_inst = nullptr;
  }
}

TotalWLSummary WirelengthAPI::totalWL(PointSets point_sets)
{
  TotalWLSummary total_wirelength_summary;

  total_wirelength_summary.HPWL = EVAL_WIRELENGTH_INST->evalTotalHPWL(point_sets);
  total_wirelength_summary.FLUTE = EVAL_WIRELENGTH_INST->evalTotalFLUTE(point_sets);
  total_wirelength_summary.HTree = EVAL_WIRELENGTH_INST->evalTotalHTree(point_sets);
  total_wirelength_summary.VTree = EVAL_WIRELENGTH_INST->evalTotalVTree(point_sets);

  return total_wirelength_summary;
}

TotalWLSummary WirelengthAPI::totalWL()
{
  TotalWLSummary total_wirelength_summary;

  EVAL_WIRELENGTH_INST->initIDB();
  EVAL_WIRELENGTH_INST->initEGR();
  EVAL_WIRELENGTH_INST->initFlute();
  total_wirelength_summary.HPWL = EVAL_WIRELENGTH_INST->evalTotalHPWL();
  total_wirelength_summary.FLUTE = EVAL_WIRELENGTH_INST->evalTotalFLUTE();
  total_wirelength_summary.HTree = EVAL_WIRELENGTH_INST->evalTotalHTree();
  total_wirelength_summary.VTree = EVAL_WIRELENGTH_INST->evalTotalVTree();
  total_wirelength_summary.GRWL = EVAL_WIRELENGTH_INST->evalTotalEGRWL() * EVAL_WIRELENGTH_INST->getDesignUnit();
  EVAL_WIRELENGTH_INST->destroyIDB();
  EVAL_WIRELENGTH_INST->destroyEGR();
  EVAL_WIRELENGTH_INST->destroyFlute();

  return total_wirelength_summary;
}

TotalWLSummary WirelengthAPI::totalWLPure()
{
  TotalWLSummary total_wirelength_summary;

  total_wirelength_summary.HPWL = EVAL_WIRELENGTH_INST->evalTotalHPWL();
  total_wirelength_summary.FLUTE = EVAL_WIRELENGTH_INST->evalTotalFLUTE();
  total_wirelength_summary.HTree = EVAL_WIRELENGTH_INST->evalTotalHTree();
  total_wirelength_summary.VTree = EVAL_WIRELENGTH_INST->evalTotalVTree();
  total_wirelength_summary.GRWL = EVAL_WIRELENGTH_INST->evalTotalEGRWL() * EVAL_WIRELENGTH_INST->getDesignUnit();

  return total_wirelength_summary;
}

NetWLSummary WirelengthAPI::netWL(std::string net_name)
{
  NetWLSummary net_wirelength_summary;

  EVAL_WIRELENGTH_INST->initIDB();
  EVAL_WIRELENGTH_INST->initEGR();
  EVAL_WIRELENGTH_INST->initFlute();

  net_wirelength_summary = netWL(EVAL_WIRELENGTH_INST->getNetPointSet(net_name));
  net_wirelength_summary.GRWL
      = EVAL_WIRELENGTH_INST->evalNetEGRWL(EVAL_WIRELENGTH_INST->getEGRDirPath() + "/early_router/route.guide", net_name)
        * EVAL_WIRELENGTH_INST->getDesignUnit();

  EVAL_WIRELENGTH_INST->destroyIDB();
  EVAL_WIRELENGTH_INST->destroyEGR();
  EVAL_WIRELENGTH_INST->destroyFlute();

  return net_wirelength_summary;
}

NetWLSummary WirelengthAPI::netWL(PointSet point_set)
{
  NetWLSummary net_wirelength_summary;

  net_wirelength_summary.HPWL = EVAL_WIRELENGTH_INST->evalNetHPWL(point_set);
  net_wirelength_summary.FLUTE = EVAL_WIRELENGTH_INST->evalNetFLUTE(point_set);
  net_wirelength_summary.HTree = EVAL_WIRELENGTH_INST->evalNetHTree(point_set);
  net_wirelength_summary.VTree = EVAL_WIRELENGTH_INST->evalNetVTree(point_set);

  return net_wirelength_summary;
}

PathWLSummary WirelengthAPI::pathWL(PointSet point_set, PointPair point_pair)
{
  PathWLSummary path_wirelength_summary;

  path_wirelength_summary.HPWL = EVAL_WIRELENGTH_INST->evalPathHPWL(point_set, point_pair);
  path_wirelength_summary.FLUTE = EVAL_WIRELENGTH_INST->evalPathFLUTE(point_set, point_pair);
  path_wirelength_summary.HTree = EVAL_WIRELENGTH_INST->evalPathHTree(point_set, point_pair);
  path_wirelength_summary.VTree = EVAL_WIRELENGTH_INST->evalPathVTree(point_set, point_pair);

  return path_wirelength_summary;
}

double WirelengthAPI::totalEGRWL(std::string guide_path)
{
  return EVAL_WIRELENGTH_INST->evalTotalEGRWL(guide_path);
}

float WirelengthAPI::netEGRWL(std::string guide_path, std::string net_name)
{
  return EVAL_WIRELENGTH_INST->evalNetEGRWL(guide_path, net_name);
}

float WirelengthAPI::pathEGRWL(std::string guide_path, std::string net_name, std::string load_name)
{
  return EVAL_WIRELENGTH_INST->evalPathEGRWL(guide_path, net_name, load_name);
}

void WirelengthAPI::evalNetInfo()
{
  EVAL_WIRELENGTH_INST->initIDB();
  EVAL_WIRELENGTH_INST->initEGR();
  EVAL_WIRELENGTH_INST->initFlute();

  EVAL_WIRELENGTH_INST->evalNetInfo();

  EVAL_WIRELENGTH_INST->destroyIDB();
  EVAL_WIRELENGTH_INST->destroyEGR();
  EVAL_WIRELENGTH_INST->destroyFlute();
}

void WirelengthAPI::evalNetInfoPure()
{
  EVAL_WIRELENGTH_INST->evalNetInfo();
}

int32_t WirelengthAPI::findNetHPWL(std::string net_name)
{
  return EVAL_WIRELENGTH_INST->findHPWL(net_name);
}

int32_t WirelengthAPI::findNetFLUTE(std::string net_name)
{
  return EVAL_WIRELENGTH_INST->findFLUTE(net_name);
}

int32_t WirelengthAPI::findNetGRWL(std::string net_name)
{
  return EVAL_WIRELENGTH_INST->findGRWL(net_name);
}

void WirelengthAPI::evalNetFlute()
{
  EVAL_WIRELENGTH_INST->initIDB();
  EVAL_WIRELENGTH_INST->initFlute();

  EVAL_WIRELENGTH_INST->evalNetFlute();

  EVAL_WIRELENGTH_INST->destroyIDB();
  EVAL_WIRELENGTH_INST->destroyFlute();
}


}  // namespace ieval