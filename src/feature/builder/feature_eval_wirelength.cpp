/*
 * @FilePath: feature_eval_wirelength.cpp
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-09-06 11:00:07
 * @Description:
 */

#include "feature_builder.h"
#include "wirelength_api.h"

namespace ieda_feature {

TotalWLSummary FeatureBuilder::buildWirelengthEvalSummary()
{
  TotalWLSummary total_wl_summary;

  ieval::TotalWLSummary eval_total_wl_summary = WIRELENGTH_API_INST->totalWL();
  total_wl_summary.HPWL = eval_total_wl_summary.HPWL;
  total_wl_summary.FLUTE = eval_total_wl_summary.FLUTE;
  total_wl_summary.HTree = eval_total_wl_summary.HTree;
  total_wl_summary.VTree = eval_total_wl_summary.VTree;
  total_wl_summary.GRWL = eval_total_wl_summary.GRWL;

  return total_wl_summary;
}

}  // namespace ieda_feature