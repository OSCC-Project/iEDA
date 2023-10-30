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
 * @file GOCA.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */

#include "GOCA.hh"

#include <filesystem>
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
void GOCA::run()
{
  init();
  resolveSinks();
  breakLongWire();
  // report
  levelReport();
}

void GOCA::init()
{
  auto* driver_inst = _cts_driver->get_instance();
  auto* inst = new Inst(driver_inst->get_name(), driver_inst->get_location(), InstType::kBuffer);
  _driver = inst->get_driver_pin();
  _driver->set_name(_cts_driver->get_full_name());

  std::ranges::for_each(_cts_pins, [&](CtsPin* pin) {
    auto* cts_inst = pin->get_instance();
    auto type = cts_inst->get_type() == CtsInstanceType::kSink  ? InstType::kSink
                : cts_inst->get_type() == CtsInstanceType::kMux ? InstType::kNoneLib
                                                                : InstType::kBuffer;
    auto* inst = new Inst(cts_inst->get_name(), cts_inst->get_location(), type);
    auto* load_pin = inst->get_load_pin();
    load_pin->set_name(pin->get_full_name());
    // update load pin cap
    if (inst->isSink()) {
      TimingPropagator::updatePinCap(load_pin);
      _sink_pins.push_back(load_pin);
    } else {
      inst->set_cell_master(TimingPropagator::getMinSizeLib()->get_cell_master());
      _top_pins.push_back(load_pin);
    }
  });
}

void GOCA::resolveSinks()
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
  auto assigns = globalAssign();
  // clustering
  const int max_num = 20000;
  while (insts.size() > 1) {
    auto assign = _level > (int) assigns.size() ? assigns.back() : assigns[_level - 1];
    if (insts.size() > max_num) {
      LOG_INFO << "insts divide into " << std::ceil(insts.size() / max_num) << " clusters";
      auto clusters = BalanceClustering::kMeansPlus(insts, std::ceil(insts.size() / max_num), 0, 100);
      insts.clear();
      if (_max_thread > 1) {
        ThreadPool pool(_max_thread);
        std::vector<std::future<std::vector<Inst*>>> results;
        std::ranges::for_each(clusters, [&](const std::vector<Inst*>& cluster) {
          results.emplace_back(pool.enqueue([&]() { return assignApply(cluster, assign); }));
        });
        for (auto&& result : results) {
          auto assign_insts = result.get();
          insts.insert(insts.end(), assign_insts.begin(), assign_insts.end());
        }
      } else {
        std::ranges::for_each(clusters, [&](const std::vector<Inst*>& cluster) {
          auto assign_insts = assignApply(cluster, assign);
          insts.insert(insts.end(), assign_insts.begin(), assign_insts.end());
        });
      }
    } else {
      insts = assignApply(insts, assign);
    }

    _level_insts.push_back(insts);
    std::ranges::for_each(insts, [](Inst* inst) {
      auto* load_pin = inst->get_load_pin();
      // update load pin cap
      TimingPropagator::updateCapLoad<Node>(load_pin);
    });
    ++_level;
  }
  auto* root = insts.front();
  root->set_cell_master(TimingPropagator::getMaxSizeLib()->get_cell_master());
  TimingPropagator::update(root->get_driver_pin()->get_net());
  TimingPropagator::initLoadPinDelay(root->get_load_pin());
  _top_pins.push_back(root->get_load_pin());
}

void GOCA::breakLongWire()
{
  if (_top_pins.empty()) {
    return;
  }
  std::vector<Pin*> final_load_pins;
  // auto max_len = 0.8 * TimingPropagator::getMaxLength();
  auto max_len = std::numeric_limits<double>::max();
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
      buf->set_cell_master(TimingPropagator::getMinSizeLib()->get_cell_master());
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

std::vector<Assign> GOCA::globalAssign()
{
  std::vector<Assign> global_assigns;
  auto max_net_len = TimingPropagator::getMaxLength();
  auto max_fanout = TimingPropagator::getMaxFanout();
  auto max_cap = TimingPropagator::getMaxCap();
  auto skew_bound = TimingPropagator::getSkewBound();
  // global_assigns.push_back({max_net_len * 10000, max_fanout * 10000, max_cap * 10000, skew_bound * 10000, 0.98});
  // level 1 test
  global_assigns.push_back({max_net_len * 8 / 8, max_fanout, max_cap, skew_bound * 1, 1});
  // level 2 test
  global_assigns.push_back({max_net_len * 7 / 8, max_fanout, max_cap, skew_bound * 1, 0.9});
  // level 3 test
  global_assigns.push_back({max_net_len * 8 / 8, max_fanout * 3 / 4, max_cap, skew_bound * 1, 0.9});
  // level 4 test
  global_assigns.push_back({max_net_len * 8 / 8, max_fanout / 2, max_cap, skew_bound * 1, 0.8});
  // level 5 test
  global_assigns.push_back({max_net_len * 8 / 8, max_fanout / 4, max_cap, skew_bound * 1, 0.8});
  // level 6 test
  global_assigns.push_back({max_net_len * 8 / 8, max_fanout / 8, max_cap, skew_bound * 1, 0.8});
  // global_assigns.push_back({max_net_len, max_fanout, max_cap, skew_bound * 1.0, 0.8});
  return global_assigns;
}
std::vector<Inst*> GOCA::assignApply(const std::vector<Inst*>& insts, const Assign& assign)
{
  LOG_INFO << "Bounding HPWL: " << BalanceClustering::calcHPWL(insts) << std::endl;
  // pre-processing
  auto max_net_len = assign.max_net_len;
  auto max_fanout = assign.max_fanout;
  auto max_cap = assign.max_cap;
  auto cluster_ratio = assign.ratio;
  auto skew_bound = assign.skew_bound;

  std::vector<Inst*> level_insts;
  auto target_insts = insts;
  if (_level > 1) {
    BalanceClustering::latencyOpt(insts, skew_bound, 0.3);
  }

  auto clusters = BalanceClustering::iterClustering(target_insts, max_fanout, 5, 5, cluster_ratio);
  // auto enhanced_clusters = clusters;
  auto enhanced_clusters = BalanceClustering::slackClustering(clusters, max_net_len, max_fanout);

  enhanced_clusters
      = BalanceClustering::clusteringEnhancement(enhanced_clusters, max_fanout, max_cap, max_net_len, skew_bound, 200, 0.95, 10000);

  // BalanceClustering::writeClusterPy(enhanced_clusters, "cluster_level_" + std::to_string(_level));

  auto guide_centers = BalanceClustering::guideCenter(enhanced_clusters, 1);
  for (size_t i = 0; i < enhanced_clusters.size(); ++i) {
    auto cluster = enhanced_clusters[i];
    auto guide_center = guide_centers[i];
    if (_level > 1) {
      BalanceClustering::latencyOpt(cluster, skew_bound, 0.4);
    }
    auto buf = netAssign(cluster, assign, guide_center, _level > 1 ? true : false);
    level_insts.push_back(buf);
  }
  return level_insts;
}
std::vector<Inst*> GOCA::topGuide(const std::vector<Inst*>& insts, const Assign& assign)
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
    buffer->set_cell_master(TimingPropagator::getMinSizeLib()->get_cell_master());
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
Inst* GOCA::netAssign(const std::vector<Inst*>& insts, const Assign& assign, const Point& guide_center, const bool& shift)
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
  if (!shift) {
    if (cluster_load_pins.size() == 1) {
      auto* buffer = TreeBuilder::genBufInst(net_name, guide_loc);
      // set min size cell master
      buffer->set_cell_master(TimingPropagator::getMinSizeLib()->get_cell_master());
      // location legitimization
      TreeBuilder::localPlace(buffer, cluster_load_pins);
      auto* driver_pin = buffer->get_driver_pin();
      auto* load_pin = cluster_load_pins.front();
      TreeBuilder::directConnectTree(driver_pin, load_pin);
    }

    // build BEAT
    auto* buffer = TreeBuilder::beatTree(net_name, cluster_load_pins, skew_bound, guide_loc, TopoType::kBiPartition);

    auto* driver_pin = buffer->get_driver_pin();
    auto* beat_net = TimingPropagator::genNet(net_name, driver_pin, cluster_load_pins);
    buffer->set_cell_master(TimingPropagator::getMinSizeLib()->get_cell_master());
    TimingPropagator::update(beat_net);
    _nets.push_back(beat_net);
    return buffer;
  }

  auto* buffer = TreeBuilder::genBufInst(net_name, guide_loc);
  // set min size cell master
  buffer->set_cell_master(TimingPropagator::getMinSizeLib()->get_cell_master());
  // location legitimization
  TreeBuilder::localPlace(buffer, cluster_load_pins);
  auto* driver_pin = buffer->get_driver_pin();
  if (cluster_load_pins.size() == 1) {
    auto* load_pin = cluster_load_pins.front();
    TreeBuilder::directConnectTree(driver_pin, load_pin);
  } else {
    TreeBuilder::shallowLightTree("Salt", driver_pin, cluster_load_pins);
  }
  auto* net = TimingPropagator::genNet(net_name, driver_pin, cluster_load_pins);
  TimingPropagator::update(net);
  if (TimingPropagator::skewFeasible(driver_pin, skew_bound)) {
    _nets.push_back(net);
    return buffer;
  } else {
    auto feasible_cell = TreeBuilder::feasibleCell(buffer, skew_bound);
    if (!feasible_cell.empty()) {
      buffer->set_cell_master(feasible_cell.front());
      TimingPropagator::update(net);
      _nets.push_back(net);
      return buffer;
    }
  }
  // remove salt
  TimingPropagator::resetNet(net);

  // skew violation, try to opt salt
  auto* opt_salt_net = saltOpt(insts, assign);
  if (opt_salt_net) {
    _nets.push_back(opt_salt_net);
    return opt_salt_net->get_driver_pin()->get_inst();
  }

  // build DME
  buffer = TreeBuilder::beatTree(net_name, cluster_load_pins, skew_bound, guide_loc);

  driver_pin = buffer->get_driver_pin();
  auto* bst_net = TimingPropagator::genNet(net_name, driver_pin, cluster_load_pins);
  buffer->set_cell_master(TimingPropagator::getMinSizeLib()->get_cell_master());
  TimingPropagator::update(bst_net);
  _nets.push_back(bst_net);

  return buffer;
}
Net* GOCA::saltOpt(const std::vector<Inst*>& insts, const Assign& assign)
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
void GOCA::writeNetPy(Pin* root, const std::string& save_name) const
{
  LOG_INFO << "Writing net to python file...";
  // write the cluster to python file
  auto* config = CTSAPIInst.get_config();
  auto path = config->get_sta_workspace();
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
void GOCA::levelReport() const
{
  using Inst_Func = std::function<double(const Inst*)>;
  auto gen_level_rpt = [&](const CtsReportType& rpt_type, const std::string& rpt_tittle, const std::string& file_name,
                           Inst_Func get_val_func, Inst_Func vio_func = nullptr) {
    auto dir = CTSAPIInst.get_config()->get_sta_workspace() + "/level_log";
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
}  // namespace icts