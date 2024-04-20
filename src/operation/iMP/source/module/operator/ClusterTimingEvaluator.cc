#include "ClusterTimingEvaluator.hh"

#include "Logger.hpp"
#include "TimingEngine.hh"
#include "TimingIDBAdapter.hh"

namespace imp {

ClusterTimingEvaluator::ClusterTimingEvaluator()
{
  _sdc_file = dmInst->get_config().get_sdc_path();
  _lib_files = dmInst->get_config().get_lib_paths();
}

void ClusterTimingEvaluator::initTimingEngine()
{
  _idb_builder = dmInst->get_idb_builder();
  _timing_engine = ista::TimingEngine::getOrCreateTimingEngine();
  _timing_engine->readLiberty(_lib_files);

  auto db_adapter = std::make_unique<TimingIDBAdapter>(_timing_engine->get_ista());
  db_adapter->set_idb(_idb_builder);
  db_adapter->convertDBToTimingNetlist();
  _timing_engine->set_db_adapter(std::move(db_adapter));
  _timing_engine->readSdc(_sdc_file.c_str());
  _timing_engine->buildGraph();
  _power_engine = ipower::PowerEngine::getOrCreatePowerEngine();
}

void ClusterTimingEvaluator::createDataflow(const std::vector<std::set<std::string>>& cluster_instances,
                                            const std::set<std::string>& src_instances, size_t max_hop)
{

  auto start = std::chrono::high_resolution_clock::now();

  INFO("Cluster num: ", cluster_instances.size());
  _timing_engine->updateTiming();
  _power_engine->creatDataflow();
  _dataflow_connection_map = _power_engine->buildConnectionMap(cluster_instances, src_instances, max_hop);
  _dataflow_connections.clear();
  for (auto&& [src_cluster_id, snk_clusters] : _dataflow_connection_map) {
    for (auto&& snk_cluster : snk_clusters) {
      _dataflow_connections[std::tuple<size_t, size_t, size_t>(src_cluster_id, snk_cluster._dst_cluster_id, snk_cluster._hop)] += 1;
    }
  }

  // INFO("Dataflow_start_id: ", src_cluster_id, " Dataflow_end_id: ", snk_cluster._dst_cluster_id, " hop: ", snk_cluster._hop);
  for (auto&& [k, v] : _dataflow_connections){
    INFO("Dataflow_start_id: ", std::get<0>(k), " Dataflow_end_id: ", std::get<1>(k), " hop: ", std::get<2>(k), " connetion-num: ", v);
  }
  auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = std::chrono::duration<float>(end - start);
    INFO("create dataflow time:", elapsed.count(), "s");
}

std::vector<std::tuple<std::string, std::string, double>> ClusterTimingEvaluator::getNegativeSlackPaths(
    std::unordered_map<idb::IdbNet*, std::map<std::string, double>>& net_lengths_between_cluster, double percent)
{
  INFO("Update from cluster graph start");
  auto start = std::chrono::high_resolution_clock::now();

  for (auto iter = net_lengths_between_cluster.begin(); iter != net_lengths_between_cluster.end(); ++iter) {
    _timing_engine->resetRcTree(_timing_engine->findNet(Str::trimBackslash(iter->first->get_net_name()).c_str()));
    _timing_engine->buildRcTreeAndUpdateRcTreeInfo(Str::trimBackslash(iter->first->get_net_name()).c_str(), iter->second);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> elapsed = std::chrono::duration<float>(end - start);
  INFO("STA update netlength time: ", elapsed.count());

  _timing_engine->updateTiming();
  return _timing_engine->getStartEndSlackPairsOfTopNPercentPaths(percent, ista::AnalysisMode::kMax, ista::TransType::kRise);
}

double ClusterTimingEvaluator::reportTNS()
{
  // _timing_engine->updateTiming();
  double tns = 0;
  for (auto clock : _timing_engine->getClockList()) {
    tns += _timing_engine->reportTNS(clock->get_clock_name(), ista::AnalysisMode::kMax);
  }
  return tns;
}

}  // namespace imp