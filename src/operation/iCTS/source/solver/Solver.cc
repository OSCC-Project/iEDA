// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file Solver.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */

#include "Solver.hh"

#include <filesystem>
#include <numeric>
#include <ranges>

#include "BalanceClustering.hh"
#include "CtsDesign.hh"
#include "ThreadPool/ThreadPool.h"
#include "TimingPropagator.hh"
#include "TreeBuilder.hh"
#include "log/Log.hh"
#include "report/CtsReport.hh"
#include "time/Time.hh"
namespace icts {
void Solver::run()
{
  init();
  resolveSinks();
  breakLongWire();
  // report
  levelReport();
}

void Solver::init()
{
  auto* driver_inst = _cts_driver->get_instance();
  auto* inst = new Inst(driver_inst->get_name(), driver_inst->get_location(), InstType::kBuffer);
  _driver = inst->get_driver_pin();
  _driver->set_name(_cts_driver->get_full_name());

  std::ranges::for_each(_cts_pins, [&](CtsPin* pin) {
    auto* cts_inst = pin->get_instance();
    auto type = cts_inst->get_type() == CtsInstanceType::kSink  ? InstType::kSink
                : cts_inst->get_type() == CtsInstanceType::kMux ? InstType::kBuffer
                                                                : InstType::kBuffer;
    auto* inst = new Inst(cts_inst->get_name(), cts_inst->get_location(), type);
    auto* load_pin = inst->get_load_pin();
    load_pin->set_name(pin->get_full_name());
    // update load pin cap
    if (inst->isSink()) {
      TimingPropagator::updatePinCap(load_pin);
      _sink_pins.push_back(load_pin);
    } else {
      // inst->set_cell_master(TimingPropagator::getMinSizeCell());
      // _top_pins.push_back(load_pin);
      auto cell_exist = CTSAPIInst.cellLibExist(cts_inst->get_cell_master());
      inst->set_cell_master(cell_exist ? cts_inst->get_cell_master() : TimingPropagator::getMinSizeCell());
      _sink_pins.push_back(load_pin);  // TBD for mux pin
    }
  });
  TreeBuilder::localPlace(_sink_pins);
}

void Solver::resolveSinks()
{
  if (_sink_pins.empty()) {
    return;
  }
  if (_sink_pins.size() == 1) {
    _top_pins.push_back(_sink_pins.front());
    return;
  }
  // convert to inst
  std::vector<Inst*> insts;
  std::ranges::for_each(_sink_pins, [&](Pin* pin) { insts.push_back(pin->get_inst()); });
  _level_insts.push_back(insts);
  // clustering
  while (insts.size() > 1) {
    auto assign = get_level_assign(_level);
    insts = assignApply(insts, assign);
    _level_insts.push_back(insts);
    std::ranges::for_each(insts, [](Inst* inst) {
      auto* load_pin = inst->get_load_pin();
      // update load pin cap
      TimingPropagator::updatePinCap(load_pin);
    });
    ++_level;
  }
  auto* root = insts.front();
  root->set_cell_master(TimingPropagator::getRootSizeCell());
  auto* root_driver_pin = root->get_driver_pin();
  auto* root_net = root_driver_pin->get_net();
  if (_root_buffer_required) {
    TimingPropagator::update(root_net);
    TimingPropagator::initLoadPinDelay(root->get_load_pin());
    _top_pins.push_back(root->get_load_pin());
    return;
  }
  // if (TimingPropagator::calcLen(_driver, root_driver_pin) + root_driver_pin->get_sub_len() <= TimingPropagator::getMaxLength()) {
  auto load_pins = root_net->get_load_pins();
  if (!_root_buffer_required && _inherit_root) {
    std::ranges::for_each(load_pins, [](Pin* pin) {
      auto* inst = pin->get_inst();
      inst->set_cell_master(TimingPropagator::getRootSizeCell());
    });
  }
  auto net_name = root_net->get_name();
  _nets.erase(std::remove_if(_nets.begin(), _nets.end(), [&](Net* net) { return net == root_net; }), _nets.end());
  TimingPropagator::resetNet(root_net);
  std::ranges::for_each(load_pins, [&](Pin* load_pin) { _top_pins.push_back(load_pin); });
  _level_insts.erase(_level_insts.end() - 1);
  // } else {
  //   _top_pins.push_back(root->get_load_pin());
  // }
}

std::vector<Inst*> Solver::levelProcess(const std::vector<std::vector<Inst*>>& clusters, const std::vector<Point> guide_locs,
                                        const Assign& assign)
{
  auto skew_bound = assign.skew_bound;
  std::vector<Inst*> level_insts;
  if (_max_thread > 1) {
    ThreadPool pool(_max_thread);
    std::vector<std::future<Inst*>> results;
    for (size_t i = 0; i < clusters.size(); ++i) {
      auto cluster = clusters[i];
      auto guide_center = guide_locs[i];
      results.emplace_back(pool.enqueue([&] {
        if (_level > _latency_opt_level) {
          BalanceClustering::latencyOpt(cluster, skew_bound, _local_latency_opt_ratio);
        }
        auto* buf = netAssign(cluster, assign, guide_center, _level > _shift_level);
        return buf;
      }));
    }
    for (auto&& result : results) {
      level_insts.push_back(result.get());
    }
    return level_insts;
  }

  for (size_t i = 0; i < clusters.size(); ++i) {
    auto cluster = clusters[i];
    auto guide_center = guide_locs[i];
    if (_level > _latency_opt_level) {
      BalanceClustering::latencyOpt(cluster, skew_bound, _local_latency_opt_ratio);
    }
    auto* buf = netAssign(cluster, assign, guide_center, _level > _shift_level);
    level_insts.push_back(buf);
  }
  return level_insts;
}

void Solver::breakLongWire()
{
  if (_top_pins.empty()) {
    return;
  }
  std::vector<Pin*> final_load_pins;
  // auto max_len = 0.8 * TimingPropagator::getMaxLength();
  auto max_len = _break_long_wire ? TimingPropagator::getMaxLength() : std::numeric_limits<double>::max();
  std::ranges::for_each(_top_pins, [&](Pin* pin) {
    auto len = TimingPropagator::calcLen(_driver->get_location(), pin->get_location());
    if (len < max_len) {
      final_load_pins.push_back(pin);
      return;
    }
    // break long wire
    int insert_num = std::floor(len / max_len);
    auto delta_loc = (_driver->get_location() - pin->get_location()) / (insert_num + 1);
    auto* load_pin = pin;
    auto driver_loc = pin->get_location();
    while (insert_num--) {
      driver_loc += delta_loc;
      auto buf_name = CTSAPIInst.toString(_net_name, "_break_", pin->get_name(), "_", insert_num);
      auto* buf = TreeBuilder::genBufInst(buf_name, driver_loc);
      buf->set_cell_master(TimingPropagator::getMinSizeCell());
      TreeBuilder::directConnectTree(buf->get_driver_pin(), load_pin);
      auto* net = TimingPropagator::genNet(buf_name, buf->get_driver_pin(), {load_pin});
      TimingPropagator::update(net);
      _nets.push_back(net);
      load_pin = buf->get_load_pin();
    }
    final_load_pins.push_back(load_pin);
  });
  TreeBuilder::shallowLightTree("Salt", _driver, final_load_pins);
  auto* net = TimingPropagator::genNet(_net_name, _driver, final_load_pins);
  TimingPropagator::update(net);
  _nets.push_back(net);
}

Assign Solver::get_level_assign(const int& level) const
{
  auto* config = CTSAPIInst.get_config();
  auto assign = config->query_assign(level);
#ifdef DEBUG_ICTS_SOLVER
  LOG_INFO << "Level " << level << " Assign: " << std::endl;
  LOG_INFO << "max_net_len: " << assign.max_net_len << std::endl;
  LOG_INFO << "max_fanout: " << assign.max_fanout << std::endl;
  LOG_INFO << "max_cap: " << assign.max_cap << std::endl;
  LOG_INFO << "ratio: " << assign.ratio << std::endl;
  LOG_INFO << "skew_bound: " << assign.skew_bound << std::endl;
#endif
  return assign;
}
std::vector<Inst*> Solver::assignApply(const std::vector<Inst*>& insts, const Assign& assign)
{
  LOG_INFO << "| Level: " << _level << " | Bounding HPWL: " << BalanceClustering::calcHPWL(insts) << " um | Inst Num: " << insts.size()
           << " |";
  pinCapDistReport(insts);
  // pre-processing
  auto max_net_len = assign.max_net_len;
  auto max_fanout = assign.max_fanout;
  auto max_cap = assign.max_cap;
  auto cluster_ratio = assign.ratio;
  auto skew_bound = assign.skew_bound;

  auto target_insts = insts;
  if (_level > _latency_opt_level) {
    BalanceClustering::latencyOpt(insts, skew_bound, _global_latency_opt_ratio);
  }

  auto clusters = BalanceClustering::iterClustering(target_insts, max_fanout, 5, 5, cluster_ratio);
  // auto enhanced_clusters = clusters;
  auto enhanced_clusters = BalanceClustering::slackClustering(clusters, max_net_len, max_fanout);
  if (enhanced_clusters.size() < insts.size()) {
    enhanced_clusters
        = BalanceClustering::clusteringEnhancement(enhanced_clusters, max_fanout, max_cap, max_net_len, skew_bound, 200, 0.95, 10000);
  }
  // top guide
  auto guide_centers = BalanceClustering::guideCenter(enhanced_clusters, std::nullopt, TimingPropagator::getMinLength(), 1);
  std::vector<icts::Inst*> level_insts;
  if (_level > 1) {
    // higherDelayOpt(enhanced_clusters, guide_centers, level_insts);
  }
  // if (enhanced_clusters.size() == insts.size() - level_insts.size()) {
  //   std::sort(enhanced_clusters.begin(), enhanced_clusters.end(),
  //             [](const std::vector<Inst*>& cluster1, const std::vector<Inst*>& cluster2) {
  //               auto* inst1 = cluster1.front();
  //               auto* inst2 = cluster2.front();
  //               return inst1->get_driver_pin()->get_max_delay() < inst2->get_driver_pin()->get_max_delay();
  //             });
  //   for (size_t i = 0; i < enhanced_clusters.size(); ++i) {
  //     auto bound = std::ceil(1.0 * enhanced_clusters.size() / 2);
  //     if (i < bound) {
  //       // lower case, insert buf
  //       auto cluster = enhanced_clusters[i];
  //       auto* inst = cluster.front();
  //       auto* single_buf = levelProcess({cluster}, {inst->get_location()}, assign).front();
  //       level_insts.push_back(single_buf);
  //     } else {
  //       // higher case, return to next level
  //       level_insts.push_back(enhanced_clusters[i].front());
  //     }
  //   }
  //   return level_insts;
  // }
  // BalanceClustering::writeClusterPy(enhanced_clusters, "cluster_level_" + std::to_string(_level));

  auto bufs = levelProcess(enhanced_clusters, guide_centers, assign);
  level_insts.insert(level_insts.end(), bufs.begin(), bufs.end());
  return level_insts;
}
std::vector<Inst*> Solver::topGuide(const std::vector<Inst*>& insts, const Assign& assign)
{
  auto max_net_len = assign.max_net_len;
  auto sorted_insts = insts;
  int max_dist = max_net_len * 1.0 / TimingPropagator::getDbUnit();
  int est_net_dist = BalanceClustering::estimateNetLength(insts) * TimingPropagator::getDbUnit();
  while (est_net_dist > max_dist) {
    std::ranges::sort(sorted_insts, [](Inst* inst1, Inst* inst2) {
      return inst1->get_driver_pin()->get_max_delay() < inst2->get_driver_pin()->get_max_delay();
    });
    auto min_delay_inst = sorted_insts.front();
    auto loc = min_delay_inst->get_location();
    sorted_insts.erase(sorted_insts.begin());
    auto center = BalanceClustering::calcBoundCentroid(sorted_insts);

    auto center_dist = TimingPropagator::calcDist(loc, center);
    auto shift_dist = std::min(max_dist / 2, center_dist);
    auto new_loc = (center - loc) * (1.0 * shift_dist / center_dist) + loc;
    auto net_name = CTSAPIInst.toString(_net_name, "_", CTSAPIInst.genId());
    auto* buffer = TreeBuilder::genBufInst(net_name, new_loc);
    buffer->set_cell_master(TimingPropagator::getMinSizeCell());
    auto* load_pin = min_delay_inst->get_load_pin();
    auto* driver_pin = buffer->get_driver_pin();
    TreeBuilder::directConnectTree(driver_pin, load_pin);

    auto* net = TimingPropagator::genNet(net_name, driver_pin, {load_pin});
    TimingPropagator::update(net);
    _nets.push_back(net);
    est_net_dist = BalanceClustering::estimateNetLength(sorted_insts) * TimingPropagator::getDbUnit();
    sorted_insts.push_back(buffer);
  }
  return sorted_insts;
}
Inst* Solver::netAssign(const std::vector<Inst*>& insts, const Assign& assign, const Point& guide_center, const bool& shift)
{
  auto max_net_len = assign.max_net_len;
  auto skew_bound = assign.skew_bound;

  auto guide_loc = guide_center;
  // center shift
  int max_dist = max_net_len * TimingPropagator::getDbUnit();
  int net_dist = BalanceClustering::estimateNetLength(insts) * TimingPropagator::getDbUnit();
  if (shift && net_dist <= max_dist) {
    auto center = BalanceClustering::calcBoundCentroid(insts);
    int center_dist = std::ceil(TimingPropagator::calcDist(center, guide_center));
    auto ratio = 0.1 + (_level - 1) * 0.25;
    ratio = ratio > 1 ? 1 : ratio;
    int allow_center_dist = ratio * center_dist;
    auto shift_dist = std::min(max_dist - net_dist, allow_center_dist);
    guide_loc = center_dist > 0 ? (guide_center - center) * (1.0 * shift_dist / center_dist) + center : center;
  }
  auto net_name = CTSAPIInst.toString(_net_name, "_", CTSAPIInst.genId());
  std::vector<Pin*> cluster_load_pins;
  std::ranges::for_each(insts, [&cluster_load_pins](Inst* inst) {
    auto load_pin = inst->get_load_pin();
    cluster_load_pins.push_back(load_pin);
  });
  // if (!shift) {
  if (cluster_load_pins.size() == 1) {
    auto* buffer = TreeBuilder::genBufInst(net_name, guide_loc);
    // set min size cell master
    buffer->set_cell_master(TimingPropagator::getMinSizeCell());
    // location legitimization
    TreeBuilder::localPlace(buffer, cluster_load_pins);
    auto* driver_pin = buffer->get_driver_pin();
    auto* load_pin = cluster_load_pins.front();
    TreeBuilder::directConnectTree(driver_pin, load_pin);
    auto* net = TimingPropagator::genNet(net_name, driver_pin, cluster_load_pins);
    TimingPropagator::update(net);
    _nets.push_back(net);
    return buffer;
  }

  // build CBS
  auto* buffer = TreeBuilder::shiftCBSTree(net_name, cluster_load_pins, skew_bound, guide_loc, TopoType::kBiPartition,
                                           _level > _shift_level, max_net_len);

  auto* driver_pin = buffer->get_driver_pin();
  auto* cbs_net = TimingPropagator::genNet(net_name, driver_pin, cluster_load_pins);
  buffer->set_cell_master(TimingPropagator::getMinSizeCell());
  // TreeBuilder::iterativeFixSkew(cbs_net, skew_bound, guide_loc); // TBD for testing
  // TreeBuilder::iterativeFixSkew(cbs_net, skew_bound, guide_loc);
  TimingPropagator::update(cbs_net);
  _nets.push_back(cbs_net);
  return buffer;
  // }

  // auto* buffer = TreeBuilder::genBufInst(net_name, guide_loc);
  // // set min size cell master
  // buffer->set_cell_master(TimingPropagator::getMinSizeCell());
  // // location legitimization
  // TreeBuilder::localPlace(buffer, cluster_load_pins);
  // auto* driver_pin = buffer->get_driver_pin();
  // if (cluster_load_pins.size() == 1) {
  //   auto* load_pin = cluster_load_pins.front();
  //   TreeBuilder::directConnectTree(driver_pin, load_pin);
  //   auto* net = TimingPropagator::genNet(net_name, driver_pin, cluster_load_pins);
  //   TimingPropagator::update(net);
  //   _nets.push_back(net);
  //   return buffer;
  // }
  // TreeBuilder::shallowLightTree("Salt", driver_pin, cluster_load_pins);
  // auto* net = TimingPropagator::genNet(net_name, driver_pin, cluster_load_pins);
  // TimingPropagator::update(net);
  // if (TimingPropagator::skewFeasible(driver_pin, skew_bound)) {
  //   _nets.push_back(net);
  //   return buffer;
  // } else {
  //   auto feasible_cell = TreeBuilder::feasibleCell(buffer, skew_bound);
  //   if (!feasible_cell.empty()) {
  //     buffer->set_cell_master(feasible_cell.front());
  //     TimingPropagator::update(net);
  //     _nets.push_back(net);
  //     return buffer;
  //   }
  // }
  // // remove salt
  // TimingPropagator::resetNet(net);

  // // skew violation, try to opt salt
  // auto* opt_salt_net = saltOpt(insts, assign);
  // if (opt_salt_net) {
  //   _nets.push_back(opt_salt_net);
  //   return opt_salt_net->get_driver_pin()->get_inst();
  // }

  // // build DME
  // buffer = TreeBuilder::cbsTree(net_name, cluster_load_pins, skew_bound, guide_loc);

  // driver_pin = buffer->get_driver_pin();
  // auto* bst_net = TimingPropagator::genNet(net_name, driver_pin, cluster_load_pins);
  // buffer->set_cell_master(TimingPropagator::getMinSizeCell());
  // TimingPropagator::update(bst_net);
  // _nets.push_back(bst_net);

  return buffer;
}
Net* Solver::saltOpt(const std::vector<Inst*>& insts, const Assign& assign)
{
  struct Buffering
  {
    Point loc;
    size_t cell_id;
    double skew;
  };
  auto skew_bound = assign.skew_bound;

  int min_x = std::numeric_limits<int>::max();
  int min_y = std::numeric_limits<int>::max();
  int max_x = std::numeric_limits<int>::min();
  int max_y = std::numeric_limits<int>::min();
  std::ranges::for_each(insts, [&](Inst* inst) {
    auto loc = inst->get_location();
    min_x = std::min(min_x, loc.x());
    min_y = std::min(min_y, loc.y());
    max_x = std::max(max_x, loc.x());
    max_y = std::max(max_y, loc.y());
  });
  Point lb = Point(min_x, min_y);
  Point rb = Point(max_x, min_y);
  Point lt = Point(min_x, max_y);
  Point rt = Point(max_x, max_y);
  Point center = BalanceClustering::calcCentroid(insts);
  Point bound_center = BalanceClustering::calcBoundCentroid(insts);

  std::vector<Point> loc_list = {lb, rb, lt, rt, center, bound_center};
  auto lib_list = TimingPropagator::getDelayLibs();

  std::vector<Buffering> feasible_assign;

  std::vector<Pin*> cluster_load_pins;
  std::ranges::for_each(insts, [&cluster_load_pins](Inst* inst) {
    auto load_pin = inst->get_load_pin();
    cluster_load_pins.push_back(load_pin);
  });
  auto net_name = CTSAPIInst.toString(_net_name, "_", CTSAPIInst.genId());
  std::ranges::for_each(loc_list, [&](const Point& loc) {
    for (size_t i = 0; i < lib_list.size(); ++i) {
      auto* lib = lib_list[i];
      auto cell_master = lib->get_cell_master();
      auto* buffer = TreeBuilder::genBufInst(net_name, loc);
      auto* driver_pin = buffer->get_driver_pin();
      buffer->set_cell_master(cell_master);
      TreeBuilder::localPlace(buffer, cluster_load_pins);
      TreeBuilder::shallowLightTree(net_name, driver_pin, cluster_load_pins);
      auto* net = TimingPropagator::genNet(net_name, driver_pin, cluster_load_pins);
      TimingPropagator::update(net);
      if (TimingPropagator::skewFeasible(driver_pin, skew_bound)) {
        auto skew = TimingPropagator::calcSkew(driver_pin);
        feasible_assign.push_back({buffer->get_location(), i, skew});
      }
      TimingPropagator::resetNet(net);
    }
  });

  if (feasible_assign.empty()) {
    return nullptr;
  }
  std::ranges::sort(feasible_assign, [&](const Buffering& assign1, const Buffering& assign2) {
    if (assign1.cell_id == assign2.cell_id) {
      return assign1.skew < assign2.skew;  // sort by skew
    }
    return assign1.cell_id < assign2.cell_id;  // sort by cell size
  });
  // assign
  auto best_assign = feasible_assign.front();
  auto* buffer = TreeBuilder::genBufInst(net_name, best_assign.loc);
  auto* driver_pin = buffer->get_driver_pin();
  auto cell_master = lib_list[best_assign.cell_id]->get_cell_master();
  buffer->set_cell_master(cell_master);
  TreeBuilder::localPlace(buffer, cluster_load_pins);
  TreeBuilder::shallowLightTree(net_name, driver_pin, cluster_load_pins);
  auto* net = TimingPropagator::genNet(net_name, driver_pin, cluster_load_pins);
  TimingPropagator::update(net);
  return net;
}
void Solver::higherDelayOpt(std::vector<std::vector<Inst*>>& clusters, std::vector<Point>& guide_centers,
                            std::vector<Inst*>& level_insts) const
{
  auto calc_max_delay = [](const std::vector<Inst*> cluster) {
    auto max_delay = std::numeric_limits<double>::min();
    std::ranges::for_each(cluster, [&](const Inst* inst) {
      if (!inst->isBuffer()) {
        return;
      }
      auto* driver_pin = inst->get_driver_pin();
      max_delay = std::max(driver_pin->get_max_delay(), max_delay);
    });
    return max_delay;
  };
  std::vector<double> max_delay_list(clusters.size());
  for (size_t i = 0; i < clusters.size(); ++i) {
    max_delay_list[i] = calc_max_delay(clusters[i]);
  }
  auto min_val = std::ranges::min(max_delay_list);
  auto max_val = std::ranges::max(max_delay_list);
  if (max_val - min_val < TimingPropagator::getMinInsertDelay()) {
    return;
  }
  auto bound = min_val + TimingPropagator::getMinInsertDelay();
  std::vector<std::vector<Inst*>> lower_clusters;
  std::vector<Point> lower_guide_centers;
  for (size_t i = 0; i < clusters.size(); ++i) {
    if (max_delay_list[i] <= bound) {
      lower_clusters.push_back(clusters[i]);
      lower_guide_centers.push_back(guide_centers[i]);
    } else {
      level_insts.insert(level_insts.end(), clusters[i].begin(), clusters[i].end());
    }
  }
  clusters = lower_clusters;
  guide_centers = lower_guide_centers;
}
void Solver::writeNetPy(Pin* root, const std::string& save_name) const
{
  LOG_INFO << "Writing net to python file...";
  // write the cluster to python file
  auto* config = CTSAPIInst.get_config();
  auto path = config->get_work_dir();
  std::ofstream ofs(path + "/" + save_name + ".py");
  ofs << "import matplotlib.pyplot as plt" << std::endl;
  ofs << "fig = plt.figure(figsize=(8,6), dpi=300)" << std::endl;
  auto write_node = [&ofs](Node* node) {
    ofs << "x = [";
    ofs << node->get_location().x();
    ofs << "]" << std::endl;
    ofs << "y = [";
    ofs << node->get_location().y();
    ofs << "]" << std::endl;
    ofs << "plt.scatter(x, y)" << std::endl;
    ofs << "plt.text(x[0], y[0], '" << node->get_name() << "')" << std::endl;
    auto parent = node->get_parent();
    if (parent) {
      // add fly line
      ofs << "plt.plot([";
      ofs << node->get_location().x();
      ofs << ", ";
      ofs << parent->get_location().x();
      ofs << "], [";
      ofs << node->get_location().y();
      ofs << ", ";
      ofs << parent->get_location().y();
      ofs << "], color='black', linestyle='-', linewidth=0.5)" << std::endl;
    }
  };
  root->preOrder(write_node);
  ofs << "plt.show()" << std::endl;
  ofs << "plt.savefig('" + save_name + ".png')" << std::endl;
  ofs.close();
}
void Solver::levelReport() const
{
  using Inst_Func = std::function<double(const Inst*)>;
  auto gen_level_rpt = [&](const CtsReportType& rpt_type, const std::string& rpt_tittle, const std::string& file_name,
                           Inst_Func get_val_func, Inst_Func vio_func = nullptr) {
    auto dir = CTSAPIInst.get_config()->get_work_dir() + "/level_log";
    if (!std::filesystem::exists(dir)) {
      std::filesystem::create_directories(dir);
    }

    auto rpt = CtsReportTable::createReportTable("Level " + rpt_tittle + " Log", rpt_type);
    for (size_t level = 1; level < _level_insts.size(); ++level) {
      double min_val = std::numeric_limits<double>::max();
      double max_val = std::numeric_limits<double>::min();
      double avg_val = 0;
      double vio_num = 0;
      auto insts = _level_insts[level];
      std::ranges::for_each(insts, [&min_val, &max_val, &avg_val, &vio_num, &get_val_func, &vio_func](const Inst* inst) {
        auto val = get_val_func(inst);
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        avg_val += val;
        if (vio_func != nullptr) {
          vio_num += vio_func(inst);
        }
      });
      avg_val /= insts.size();
      if (vio_func == nullptr) {
        (*rpt) << level << insts.size() << min_val << max_val << avg_val << "None" << TABLE_ENDLINE;
      } else {
        (*rpt) << level << insts.size() << min_val << max_val << avg_val << vio_num << TABLE_ENDLINE;
      }
    }

    auto save_file_name = _net_name + "_" + file_name + ".rpt";
    auto save_path = dir + "/" + save_file_name;
    std::ofstream outfile(save_path);
    outfile << "Generate the report at " << Time::getNowWallTime() << std::endl;
    outfile << rpt->c_str();
    outfile.close();
  };
  // fanout rpt
  gen_level_rpt(
      CtsReportType::kLevelFanout, "Fanout", "fanout",
      [](const Inst* inst) {
        auto* driver_pin = inst->get_driver_pin();
        auto* net = driver_pin->get_net();
        return net->getFanout();
      },
      [](const Inst* inst) {
        auto* driver_pin = inst->get_driver_pin();
        auto* net = driver_pin->get_net();
        return net->getFanout() > TimingPropagator::getMaxFanout();
      });
  // net len rpt
  gen_level_rpt(
      CtsReportType::kLevelNetLen, "Net Length", "net_len",
      [](const Inst* inst) {
        auto* driver_pin = inst->get_driver_pin();
        return driver_pin->get_sub_len();
      },
      [](const Inst* inst) {
        auto* driver_pin = inst->get_driver_pin();
        return driver_pin->get_sub_len() > TimingPropagator::getMaxLength();
      });
  // cap rpt
  gen_level_rpt(
      CtsReportType::kLevelCap, "Cap", "cap",
      [](const Inst* inst) {
        auto* driver_pin = inst->get_driver_pin();
        return driver_pin->get_cap_load();
      },
      [](const Inst* inst) {
        auto* driver_pin = inst->get_driver_pin();
        return driver_pin->get_cap_load() > TimingPropagator::getMaxCap();
      });
  // slew rpt
  gen_level_rpt(
      CtsReportType::kLevelSlew, "Slew", "slew",
      [](const Inst* inst) {
        auto* load_pin = inst->get_load_pin();
        return load_pin->get_slew_in();
      },
      [](const Inst* inst) {
        auto* load_pin = inst->get_load_pin();
        return load_pin->get_slew_in() > TimingPropagator::getMaxBufTran();
      });
  // min delay rpt
  gen_level_rpt(CtsReportType::kLevelDelay, "Min Delay", "min_delay", [](const Inst* inst) {
    auto* driver_pin = inst->get_driver_pin();
    return driver_pin->get_min_delay();
  });
  // max delay rpt
  gen_level_rpt(CtsReportType::kLevelDelay, "Max Delay", "max_delay", [](const Inst* inst) {
    auto* driver_pin = inst->get_driver_pin();
    return driver_pin->get_max_delay();
  });
  // insert delay rpt
  gen_level_rpt(CtsReportType::kLevelInsertDelay, "Insert Delay", "insert_delay",
                [](const Inst* inst) { return inst->get_insert_delay(); });
  // skew rpt
  gen_level_rpt(
      CtsReportType::kLevelSkew, "Skew", "skew",
      [](const Inst* inst) {
        auto* driver_pin = inst->get_driver_pin();
        return driver_pin->get_max_delay() - driver_pin->get_min_delay();
      },
      [](const Inst* inst) {
        auto* driver_pin = inst->get_driver_pin();
        return !TimingPropagator::skewFeasible(driver_pin);
      });
}
void Solver::pinCapDistReport(const std::vector<Inst*>& insts) const
{
  std::vector<double> cap_list;
  std::ranges::for_each(insts, [&cap_list](const Inst* inst) {
    auto* load_pin = inst->get_load_pin();
    cap_list.push_back(load_pin->get_cap_load());
  });
  std::ranges::sort(cap_list);
  // min, max, avg, median
  auto min_val = cap_list.front();
  auto max_val = cap_list.back();
  auto avg_val = std::accumulate(cap_list.begin(), cap_list.end(), 0.0) / cap_list.size();
  auto median_val = cap_list[cap_list.size() / 2];
  LOG_INFO << ">>> Pin Cap Dist Report: " << std::endl;
  LOG_INFO << ">>> Min: " << min_val << " fF" << std::endl;
  LOG_INFO << ">>> Max: " << max_val << " fF" << std::endl;
  LOG_INFO << ">>> Avg: " << avg_val << " fF" << std::endl;
  LOG_INFO << ">>> Median: " << median_val << " fF" << std::endl;
}
}  // namespace icts