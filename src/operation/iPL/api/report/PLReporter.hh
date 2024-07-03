#ifndef IPL_REPORTER_H
#define IPL_REPORTER_H

#include "ExternalAPI.hh"

namespace ipl {
class PLReporter
{
 public:
  explicit PLReporter(ExternalAPI* _external_api);
  PLReporter(const PLReporter&) = delete;
  PLReporter(PLReporter&&) = delete;
  ~PLReporter();
  PLReporter& operator=(const PLReporter&) = delete;
  PLReporter& operator=(PLReporter&&) = delete;

  void reportPLInfo(std::string target_dir);
  void reportTopoInfo();
  void reportWLInfo(std::ofstream& feed, std::string target_dir);
  void reportSTWLInfo(std::ofstream& feed);
  void reportHPWLInfo(std::ofstream& feed);
  void reportLongNetInfo(std::ofstream& feed);
  void reportViolationInfo(std::ofstream& feed, std::string target_dir);
  void reportBinDensity(std::ofstream& feed);
  int32_t reportOverlapInfo(std::ofstream& feed);
  void reportLayoutWhiteInfo(std::string target_dir);
  void reportTimingInfo(std::ofstream& feed);
  void reportCongestionInfo(std::ofstream& feed);
  void reportPLBaseInfo(std::ofstream& feed);
  
  void printHPWLInfo();
  void printTimingInfo();
  void saveNetPinInfoForDebug(std::string path);
  void savePinListInfoForDebug(std::string path);
  void plotConnectionForDebug(std::vector<std::string> net_name_list, std::string path);
  void plotModuleListForDebug(std::vector<std::string> module_prefix_list, std::string path);
  void plotModuleStateForDebug(std::vector<std::string> special_inst_list, std::string path);

  // tmp for iEDA Evaluation.
  void reportEDAEvaluation();
  void reportEDAFillerEvaluation();
  void reportTDPEvaluation();
 
  ExternalAPI* _external_api;
};

}  // namespace ipl

#endif