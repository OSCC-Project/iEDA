#include "ClusterTimingEvaluator.hh"

#include "TimingEngine.hh"
#include "TimingIDBAdapter.hh"

namespace imp {

ClusterTimingEvaluator::ClusterTimingEvaluator()
{
  _sdc_file = dmInst->get_config().get_sdc_path();
  _lib_files = dmInst->get_config().get_lib_paths();
  std::cout << "constructer lib_files_size: " << _lib_files.size() << std::endl;
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
  _timing_engine->buildGraph();
  _timing_engine->readSdc(_sdc_file.c_str());
}

std::vector<std::tuple<std::string, std::string, double>> ClusterTimingEvaluator::getNegativeSlackPaths(
    std::unordered_map<idb::IdbNet*, std::map<std::string, double>>& net_lengths_between_cluster, double percent)
{
  std::cout << "update from cluster graph start " << std::endl;
  clock_t begin = clock();

  for (auto iter = net_lengths_between_cluster.begin(); iter != net_lengths_between_cluster.end(); ++iter) {
    _timing_engine->resetRcTree(_timing_engine->findNet(Str::trimBackslash(iter->first->get_net_name()).c_str()));
    _timing_engine->buildRcTreeAndUpdateRcTreeInfo(Str::trimBackslash(iter->first->get_net_name()).c_str(), iter->second);
  }
  clock_t end = clock();
  std::cout << "sta update netlength time: " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

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