#pragma once
#include <idm.h>

#include <cstddef>
#include <set>
#include <tuple>
#include <vector>

#include "api/PowerEngine.hh"

namespace imp {

// class idb::IdbBuilder;
// class ista::TimingEngine;
// class ipower::PowerEngine;
// class ipower::ClusterConnection;

class ClusterTimingEvaluator
{
 public:
  ClusterTimingEvaluator();
  ~ClusterTimingEvaluator() = default;
  void initTimingEngine();
  double reportTNS();
  std::vector<std::tuple<std::string, std::string, double>> getNegativeSlackPaths(
      std::unordered_map<idb::IdbNet*, std::map<std::string, double>>& net_lengths_between_cluster, double percent = 0.5);
  void createDataflow(const std::vector<std::set<std::string>>& cluster_instances,
                      const std::set<std::string>& src_instances, size_t max_hop);
  const std::map<std::tuple<size_t, size_t, size_t>, size_t>& get_dataflow_connections() { return _dataflow_connections; }

 private:
  idb::IdbBuilder* _idb_builder;
  std::string _sdc_file;
  std::vector<std::string> _lib_files;
  ista::TimingEngine* _timing_engine;
  ipower::PowerEngine* _power_engine;
  std::map<std::size_t, std::vector<ipower::ClusterConnection>> _dataflow_connection_map;
  // std::vector<std::tuple<size_t, size_t, size_t>> _dataflow_connections;
  std::map<std::tuple<size_t, size_t, size_t>, size_t> _dataflow_connections;
};

}  // namespace imp