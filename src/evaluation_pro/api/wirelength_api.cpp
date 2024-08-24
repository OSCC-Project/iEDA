#include "wirelength_api.h"

#include "wirelength_eval.h"

namespace ieval {

WirelengthAPI::WirelengthAPI()
{
}

WirelengthAPI::~WirelengthAPI()
{
}

TotalWLSummary WirelengthAPI::totalWL(PointSets point_sets)
{
  TotalWLSummary total_wirelength_summary;

  WirelengthEval wirelength_eval;

  total_wirelength_summary.HPWL = wirelength_eval.evalTotalHPWL(point_sets);
  total_wirelength_summary.FLUTE = wirelength_eval.evalTotalFLUTE(point_sets);
  total_wirelength_summary.HTree = wirelength_eval.evalTotalHTree(point_sets);
  total_wirelength_summary.VTree = wirelength_eval.evalTotalVTree(point_sets);

  return total_wirelength_summary;
}

NetWLSummary WirelengthAPI::netWL(PointSet point_set)
{
  NetWLSummary net_wirelength_summary;

  WirelengthEval wirelength_eval;

  net_wirelength_summary.HPWL = wirelength_eval.evalNetHPWL(point_set);
  net_wirelength_summary.FLUTE = wirelength_eval.evalNetFLUTE(point_set);
  net_wirelength_summary.HTree = wirelength_eval.evalNetHTree(point_set);
  net_wirelength_summary.VTree = wirelength_eval.evalNetVTree(point_set);

  return net_wirelength_summary;
}

PathWLSummary WirelengthAPI::pathWL(PointSet point_set)
{
  PathWLSummary path_wirelength_summary;

  WirelengthEval wirelength_eval;

  path_wirelength_summary.HPWL = wirelength_eval.evalPathHPWL(point_set);
  path_wirelength_summary.FLUTE = wirelength_eval.evalPathFLUTE(point_set);
  path_wirelength_summary.HTree = wirelength_eval.evalPathHTree(point_set);
  path_wirelength_summary.VTree = wirelength_eval.evalPathVTree(point_set);

  return path_wirelength_summary;
}

}  // namespace ieval