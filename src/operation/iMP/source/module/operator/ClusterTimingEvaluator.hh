#pragma once
#include <idm.h>

#include <cstddef>
#include <set>

namespace imp {

class idb::IdbBuilder;
class ista::TimingEngine;

class ClusterTimingEvaluator
{
 public:
  ClusterTimingEvaluator();
  ~ClusterTimingEvaluator() = default;
  void initTimingEngine();
  double reportTNS();
  std::vector<std::tuple<std::string, std::string, double>> getNegativeSlackPaths(
      std::unordered_map<idb::IdbNet*, std::map<std::string, double>>& net_lengths_between_cluster, double percent = 0.5);

 private:
  idb::IdbBuilder* _idb_builder;
  std::string _sdc_file;
  std::vector<std::string> _lib_files;
  ista::TimingEngine* _timing_engine;
};

}  // namespace imp