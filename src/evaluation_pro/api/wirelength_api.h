/*
 * @FilePath: wirelength_api.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#pragma once

#include "wirelength_db.h"

namespace ieval {

class WirelengthAPI
{
 public:
  WirelengthAPI();
  ~WirelengthAPI();

  TotalWLSummary totalWL(PointSets point_sets);
  NetWLSummary netWL(PointSet point_set);
  PathWLSummary pathWL(PointSet point_set, PointPair point_pair);

  float totalEGRWL(std::string guide_path);
  float netEGRWL(std::string guide_path, std::string net_name);
  float pathEGRWL(std::string guide_path, std::string net_name, std::string load_name);
};
}  // namespace ieval
