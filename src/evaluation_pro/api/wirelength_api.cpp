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

WirelengthAPI::WirelengthAPI()
{
}

WirelengthAPI::~WirelengthAPI()
{
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

float WirelengthAPI::totalEGRWL(std::string guide_path)
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

}  // namespace ieval