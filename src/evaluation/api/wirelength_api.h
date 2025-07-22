/*
 * @FilePath: wirelength_api.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#pragma once

#include "wirelength_db.h"

namespace ieval {

#define WIRELENGTH_API_INST (ieval::WirelengthAPI::getInst())

class WirelengthAPI
{
 public:
  WirelengthAPI();
  ~WirelengthAPI();
  static WirelengthAPI* getInst();
  static void destroyInst();

  TotalWLSummary totalWL();
  TotalWLSummary totalWLPure();
  NetWLSummary netWL(std::string net_name);

  TotalWLSummary totalWL(PointSets point_sets);
  NetWLSummary netWL(PointSet point_set);
  PathWLSummary pathWL(PointSet point_set, PointPair point_pair);

  double totalEGRWL(std::string guide_path);
  float netEGRWL(std::string guide_path, std::string net_name);
  float pathEGRWL(std::string guide_path, std::string net_name, std::string load_name);

  void evalNetInfo();
  void evalNetInfoPure();
  void evalNetFlute();
  int32_t findNetHPWL(std::string net_name);
  int32_t findNetFLUTE(std::string net_name);
  int32_t findNetGRWL(std::string net_name);

 private:
  static WirelengthAPI* _wirelength_api_inst;
};
}  // namespace ieval
