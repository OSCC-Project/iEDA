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
 * @file TreeBuilderAux.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include <filesystem>
#include <random>

#include "BoundSkewTree.hh"
#include "TestInterface.hh"
#include "log/Log.hh"

namespace {

using icts::LayerPattern;
using icts::Node;
using icts::SkewTreeFunc;
using icts::SteinerTreeFunc;
using icts::TopoType;
using ieda::Log;

struct TreeInfo
{
  double wirelength;
  double cap;
  double skew;
  double max_wire_delay;
  double max_delay;
};

struct StringPairHash
{
  size_t operator()(const std::pair<std::string, std::string>& p) const
  {
    return std::hash<std::string>()(p.first) ^ std::hash<std::string>()(p.second);
  }
};

using TreeInfoMap = std::unordered_map<std::pair<std::string, std::string>, TreeInfo, StringPairHash>;

std::string TopoTypeToString(const TopoType& topo_type)
{
  switch (topo_type) {
    case TopoType::kGreedyDist:
      return "GreedyDist";
    case TopoType::kGreedyMerge:
      return "GreedyMerge";
    case TopoType::kBiPartition:
      return "BiPartition";
    case TopoType::kBiCluster:
      return "BiCluster";
    case TopoType::kInputTopo:
      return "InputTopo";
    default:
      return "Unknown";
  }
}

class TreeBuilderDataUnit
{
 public:
  TreeBuilderDataUnit(const EnvInfo& env_info, const int& seed, const size_t& pin_num) : _env_info(env_info), _seed(seed), _pin_num(pin_num)
  {
  }
  ~TreeBuilderDataUnit() = default;
  void add_info(const std::string& method, const std::string& topo, const TreeInfo& tree_info)
  {
    LOG_FATAL_IF(_tree_info_map.find(std::make_pair(method, topo)) != _tree_info_map.end()) << "Duplicate method and topo";
    _tree_info_map[std::make_pair(method, topo)] = tree_info;
  }
  EnvInfo get_env_info() const { return _env_info; }
  int get_seed() const { return _seed; }
  size_t get_pin_num() const { return _pin_num; }
  const TreeInfoMap& get_tree_info_map() const { return _tree_info_map; }

 private:
  EnvInfo _env_info;
  int _seed = 0;
  size_t _pin_num = 0;
  TreeInfoMap _tree_info_map;
};

class TreeBuilderDataSet
{
 public:
  TreeBuilderDataSet(const size_t case_num) : _case_num(case_num) {}
  ~TreeBuilderDataSet() = default;

  const size_t& get_case_num() const { return _case_num; }

  void add_data_unit(const TreeBuilderDataUnit& data_unit) { _data_units.push_back(data_unit); }

  void writeCSV(const std::vector<std::string>& method_key, const std::vector<std::string>& topo_type_key, const std::string& dir,
                const std::string& file)
  {
    LOG_INFO << std::endl;
    LOG_INFO << "Write csv...";
    LOG_INFO << "Method: "
             << std::accumulate(method_key.begin(), method_key.end(), std::string(), [](const std::string& a, const std::string& b) {
                  return a + ", " + b;
                }).substr(2);
    LOG_INFO << "Topo type: "
             << std::accumulate(topo_type_key.begin(), topo_type_key.end(), std::string(), [](const std::string& a, const std::string& b) {
                  return a + ", " + b;
                }).substr(2);
    LOG_INFO << "Case num: " << _case_num;
    if (!std::filesystem::exists(dir)) {
      std::filesystem::create_directories(dir);
    }
    auto path = dir + "/" + file;
    std::ofstream ofs(path);
    ofs << "id,method,topo_type,wirelength,cap,skew,max_wire_delay,max_delay,pin_num" << std::endl;
    for (size_t i = 0; i < _data_units.size(); ++i) {
      auto& data_unit = _data_units[i];
      auto& tree_info_map = data_unit.get_tree_info_map();
      std::ranges::for_each(method_key, [&](const std::string& method) {
        std::ranges::for_each(topo_type_key, [&](const std::string& topo_type) {
          auto tree_info = tree_info_map.at(std::make_pair(method, topo_type));
          ofs << i << "," << method << "," << topo_type << "," << tree_info.wirelength << "," << tree_info.cap << "," << tree_info.skew
              << "," << tree_info.max_wire_delay << "," << tree_info.max_delay << "," << data_unit.get_pin_num() << std::endl;
        });
      });
    }
    ofs.close();
    LOG_INFO << "Write csv done...";
    LOG_INFO << "The file is written to csv in the path of " << path;
  }

  void writeReduceCSV(const std::string& target_method_key, const std::string& dir, const std::string& suffix)
  {
    LOG_INFO << std::endl;
    LOG_INFO << "Write reduce csv...";
    LOG_INFO << "Target method: " << target_method_key;
    LOG_INFO << "Case num: " << _case_num;
    auto csv_dir = dir + "/" + target_method_key + "_" + suffix;
    if (!std::filesystem::exists(csv_dir)) {
      std::filesystem::create_directories(csv_dir);
    }
    auto topo_type_list = {TopoType::kGreedyDist, TopoType::kGreedyMerge, TopoType::kBiCluster, TopoType::kBiPartition};
    auto write_csv = [&](const auto& func) {
      auto ref_method_key = TreeBuilder::funcName(func);
      if (target_method_key == ref_method_key) {
        return;
      }
      std::ranges::for_each(topo_type_list, [&](const auto& topo_type) {
        auto topo_type_key = TopoTypeToString(topo_type);
        auto path = csv_dir + "/" + target_method_key + "_cmp2_" + ref_method_key + "_(" + topo_type_key + ")_reduce_" + suffix + ".csv";
        std::ofstream ofs(path);
        ofs << "id,method,topo_type,wirelength,cap,skew,max_wire_delay,max_delay,pin_num" << std::endl;
        for (size_t i = 0; i < _data_units.size(); ++i) {
          auto& data_unit = _data_units[i];
          auto& tree_info_map = data_unit.get_tree_info_map();
          auto target_info = tree_info_map.at(std::make_pair(target_method_key, topo_type_key));
          auto ref_info = tree_info_map.at(std::make_pair(ref_method_key, topo_type_key));
          auto wl_ratio = (ref_info.wirelength - target_info.wirelength) / target_info.wirelength;
          auto cap_ratio = (ref_info.cap - target_info.cap) / target_info.cap;
          auto skew_ratio = (ref_info.skew - target_info.skew) / target_info.skew;
          auto max_wire_delay_ratio = (ref_info.max_wire_delay - target_info.max_wire_delay) / target_info.max_wire_delay;
          auto max_delay_ratio = (ref_info.max_delay - target_info.max_delay) / target_info.max_delay;
          ofs << i << "," << target_method_key << "," << topo_type_key << "," << wl_ratio << "," << cap_ratio << "," << skew_ratio << ","
              << max_wire_delay_ratio << "," << max_delay_ratio << "," << data_unit.get_pin_num() << std::endl;
        }
        ofs.close();
      });
    };
    std::ranges::for_each(TreeBuilder::getSteinerTreeFuncs(), [&](const auto& func) { write_csv(func); });
    std::ranges::for_each(TreeBuilder::getSkewTreeFuncs(), [&](const auto& func) { write_csv(func); });
    LOG_INFO << "Write csv done...";
    LOG_INFO << "The file is written to csv in the path of " << csv_dir;
  }

 private:
  size_t _case_num;
  std::vector<TreeBuilderDataUnit> _data_units;
};

class TreeBuilderAux : public TestInterface
{
 public:
  TreeBuilderAux(const std::string& db_config_path, const std::string& cts_config_path) : TestInterface(db_config_path, cts_config_path)
  {
    if (db_config_path.empty() && cts_config_path.empty()) {
      return;
    }
    LOG_INFO << "Router unit res (H): " << CTSAPIInst.getClockUnitRes(LayerPattern::kH);
    LOG_INFO << "Router unit cap (H): " << CTSAPIInst.getClockUnitCap(LayerPattern::kH);
    LOG_INFO << "Router unit res (V): " << CTSAPIInst.getClockUnitRes(LayerPattern::kV);
    LOG_INFO << "Router unit cap (V): " << CTSAPIInst.getClockUnitCap(LayerPattern::kV);
  }
  ~TreeBuilderAux() = default;

  void runFixedTest(const double& skew_bound) const
  {
    auto load_pins = genFixedPins();
    auto topo_type_list = {TopoType::kGreedyDist, TopoType::kGreedyMerge, TopoType::kBiCluster, TopoType::kBiPartition};
    LOG_INFO << std::endl;
    LOG_INFO << "Run fixed test...";
    LOG_INFO << "Skew bound: " << skew_bound;
    LOG_INFO << "Pin num: " << load_pins.size();
    topoTypeInfo(topo_type_list);
    steinerTreeInfo(TreeBuilder::getSteinerTreeFuncs());
    skewTreeInfo(TreeBuilder::getSkewTreeFuncs());
    std::ranges::for_each(topo_type_list, [&](const TopoType& topo_type) {
      auto guide_loc = getGuideLoc(load_pins, skew_bound, topo_type);
      for (auto func : TreeBuilder::getSteinerTreeFuncs()) {
        treeTest(func, load_pins, skew_bound, topo_type, guide_loc, true);
      }
      for (auto func : TreeBuilder::getSkewTreeFuncs()) {
        treeTest(func, load_pins, skew_bound, topo_type, guide_loc, true);
      }
    });
    std::ranges::for_each(load_pins, [](Pin* pin) { delete pin->get_inst(); });
    LOG_INFO << "Run fixed test done";
  }

  TreeBuilderDataSet runRegressTest(const EnvInfo& env_info, const size_t& case_num, const double& skew_bound) const
  {
    auto topo_type_list = {TopoType::kGreedyDist, TopoType::kGreedyMerge, TopoType::kBiCluster, TopoType::kBiPartition};
    LOG_INFO << std::endl;
    LOG_INFO << "Run regress test...";
    LOG_INFO << "Skew bound: " << skew_bound;
    LOG_INFO << "Case num: " << case_num;
    topoTypeInfo(topo_type_list);
    steinerTreeInfo(TreeBuilder::getSteinerTreeFuncs());
    skewTreeInfo(TreeBuilder::getSkewTreeFuncs());
    TreeBuilderDataSet data_set(case_num);
    for (size_t i = 0; i < case_num; ++i) {
      if (case_num > 10 && (i + 1) % (case_num / 10) == 0) {
        LOG_INFO << "Case num: " << i + 1 << "/" << case_num;
      }
      auto load_pins = genRandomPins(env_info, i);
      TreeBuilderDataUnit data_unit(env_info, i, load_pins.size());
      std::ranges::for_each(topo_type_list, [&](const TopoType& topo_type) {
        auto topo_type_name = TopoTypeToString(topo_type);
        auto guide_loc = getGuideLoc(load_pins, skew_bound, topo_type);
        for (auto func : TreeBuilder::getSteinerTreeFuncs()) {
          auto info = treeTest(func, load_pins, skew_bound, topo_type, guide_loc);
          data_unit.add_info(TreeBuilder::funcName(func), topo_type_name, info);
        }
        for (auto func : TreeBuilder::getSkewTreeFuncs()) {
          auto info = treeTest(func, load_pins, skew_bound, topo_type, guide_loc);
          data_unit.add_info(TreeBuilder::funcName(func), topo_type_name, info);
        }
      });
      std::ranges::for_each(load_pins, [](Pin* pin) { delete pin->get_inst(); });
      data_set.add_data_unit(data_unit);
    }
    LOG_INFO << "Run regress test done";
    return data_set;
  }

  void runEstimationTest(const EnvInfo& env_info, const size_t& case_num, const double& skew_bound, const std::string& dir,
                         const std::string& suffix) const
  {
    LOG_INFO << std::endl;
    LOG_INFO << "Run estimation test...";
    LOG_INFO << "Skew bound: " << skew_bound;
    LOG_INFO << "Case num: " << case_num;
    // write to csv
    if (!std::filesystem::exists(dir)) {
      std::filesystem::create_directories(dir);
    }
    auto path = dir + "/estimation_" + suffix + ".csv";
    std::ofstream ofs(path);
    ofs << "id,pin_num,min_est,max_est,min_est_final,max_est_final,est_skew,min_origin,max_origin,min_origin_final,max_origin_final,origin_"
           "skew"
        << std::endl;
    for (size_t i = 0; i < case_num; ++i) {
      if (case_num > 10 && (i + 1) % (case_num / 10) == 0) {
        LOG_INFO << "Case num: " << i + 1 << "/" << case_num;
      }
      auto load_pins = genRandomPins(env_info, i);

      auto get_delay = [&load_pins](const bool& is_driver, const bool& is_min) {
        double delay = is_min ? std::numeric_limits<double>::max() : std::numeric_limits<double>::min();
        std::ranges::for_each(load_pins, [&](Pin* pin) {
          auto* target = pin;
          if (is_driver) {
            auto* inst = pin->get_inst();
            target = inst->get_driver_pin();
          }
          delay = is_min ? std::min(delay, target->get_min_delay()) : std::max(delay, target->get_max_delay());
        });
        return delay;
      };
      // statistic
      auto min_origin_delay = get_delay(true, true);
      auto max_origin_delay = get_delay(true, false);

      auto* buf = TreeBuilder::boundSkewTree("BoundSkewTree", load_pins, skew_bound, std::nullopt, TopoType::kBiPartition);
      // statistic
      auto min_est_delay = get_delay(false, true);
      auto max_est_delay = get_delay(false, false);

      auto* driver_pin = buf->get_driver_pin();
      driver_pin->preOrder([](Node* node) { node->set_pattern(static_cast<RCPattern>(1 + std::rand() % 2)); });
      buf->set_cell_master(TimingPropagator::getMinSizeCell());
      auto* net = TimingPropagator::genNet("BoundSkewTree", driver_pin, load_pins);
      TimingPropagator::update(net);
      // statistic
      auto min_est_final_delay = get_delay(false, true);
      auto max_est_final_delay = get_delay(false, false);
      auto est_skew = TimingPropagator::calcSkew(driver_pin);

      TimingPropagator::resetNet(net);
      buf = TreeBuilder::noneEstBoundSkewTree("NoneEstBoundSkewTree", load_pins, skew_bound, std::nullopt, TopoType::kBiPartition);
      driver_pin = buf->get_driver_pin();
      driver_pin->preOrder([](Node* node) { node->set_pattern(static_cast<RCPattern>(1 + std::rand() % 2)); });
      buf->set_cell_master(TimingPropagator::getMinSizeCell());
      net = TimingPropagator::genNet("NoneEstBoundSkewTree", driver_pin, load_pins);
      TimingPropagator::update(net);
      // statistic
      auto min_origin_final_delay = get_delay(false, true);
      auto max_origin_final_delay = get_delay(false, false);
      auto origin_skew = TimingPropagator::calcSkew(driver_pin);

      ofs << i << "," << load_pins.size() << "," << min_est_delay << "," << max_est_delay << "," << min_est_final_delay << ","
          << max_est_final_delay << "," << est_skew << "," << min_origin_delay << "," << max_origin_delay << "," << min_origin_final_delay
          << "," << max_origin_final_delay << "," << origin_skew << std::endl;
      // release
      std::ranges::for_each(load_pins, [](Pin* pin) { delete pin->get_inst(); });
    }
    ofs.close();
    LOG_INFO << "Run estimation test done";
  }

  void runIterativeFixSkewTest(const EnvInfo& env_info, const size_t& case_num, const double& skew_bound, const std::string& dir,
                               const std::string& suffix) const
  {
    LOG_INFO << std::endl;
    LOG_INFO << "Run iterative fix skew test...";
    LOG_INFO << "Skew bound: " << skew_bound;
    LOG_INFO << "Case num: " << case_num;
    // write to csv
    if (!std::filesystem::exists(dir)) {
      std::filesystem::create_directories(dir);
    }
    auto path = dir + "/iter_bst_" + suffix + ".csv";
    std::ofstream ofs(path);
    ofs << "id,pin_num,init_skew,iter_1_skew,iter_2_skew,iter_3_skew,iter_4_skew,iter_5_skew,iter_num" << std::endl;
    int seed = 0;
    for (size_t i = 0; i < case_num;) {
      auto load_pins = genRandomPins(env_info, ++seed);

      auto* buf = TreeBuilder::cbsTree("CBS", load_pins, skew_bound, std::nullopt, TopoType::kBiPartition);

      auto* driver_pin = buf->get_driver_pin();
      buf->set_cell_master(TimingPropagator::getMinSizeCell());
      auto* net = TimingPropagator::genNet("CBS", driver_pin, load_pins);
      TimingPropagator::update(net);

      auto est_skew = TimingPropagator::calcSkew(driver_pin);
      if (est_skew <= skew_bound * (1 + 1e-3)) {
        std::ranges::for_each(load_pins, [](Pin* pin) { delete pin->get_inst(); });
        continue;
      }
      if (case_num > 10 && (i + 1) % (case_num / 10) == 0) {
        LOG_INFO << "Case num: " << i + 1 << "/" << case_num;
      }
      ++i;
      ofs << i << "," << load_pins.size() << "," << est_skew;
      size_t iter_num = std::numeric_limits<size_t>::max();
      double iter_skew = std::numeric_limits<double>::max();
      for (size_t n = 0; n < 5; ++n) {
        if (iter_skew > skew_bound * (1 + 1e-3)) {
          TreeBuilder::iterativeFixSkew(net, skew_bound);
          iter_skew = TimingPropagator::calcSkew(driver_pin);
          if (iter_skew <= skew_bound * (1 + 1e-3)) {
            iter_num = std::min(iter_num, n + 1);
          }
        }
        ofs << "," << iter_skew;
      }
      ofs << "," << iter_num << std::endl;
      // release
      std::ranges::for_each(load_pins, [](Pin* pin) { delete pin->get_inst(); });
    }
    ofs.close();
    LOG_INFO << "Run iterative fix skew test done";
  }

 private:
  void topoTypeInfo(const std::initializer_list<TopoType>& vec, const std::string& split = ", ") const
  {
    std::string str;
    for (auto element : vec) {
      str += TopoTypeToString(element) + split;
    }
    LOG_INFO << "Topo Type: " << str.substr(0, str.size() - split.size());
  }

  void steinerTreeInfo(const std::vector<SteinerTreeFunc>& vec, const std::string& split = ", ") const
  {
    std::string str;
    for (auto element : vec) {
      str += TreeBuilder::funcName(element) + split;
    }
    LOG_INFO << "Steiner Tree Method: " << str.substr(0, str.size() - split.size());
  }

  void skewTreeInfo(const std::vector<SkewTreeFunc>& vec, const std::string& split = ", ") const
  {
    std::string str;
    for (auto element : vec) {
      str += TreeBuilder::funcName(element) + split;
    }
    LOG_INFO << "Skew Tree Method: " << str.substr(0, str.size() - split.size());
  }

  Point getGuideLoc(const std::vector<Pin*>& load_pins, const double& skew_bound, const TopoType& topo_type) const
  {
    auto* buf = TreeBuilder::boundSkewTree("BoundSkewTree", load_pins, skew_bound, std::nullopt, topo_type);
    auto* driver_pin = buf->get_driver_pin();
    driver_pin->preOrder([](Node* node) { node->set_pattern(static_cast<RCPattern>(1 + std::rand() % 2)); });
    auto* net = TimingPropagator::genNet("BoundSkewTree", driver_pin, load_pins);
    // TreeBuilder::localPlace(buf, load_pins);
    auto loc = driver_pin->get_location();
    TimingPropagator::resetNet(net);
    return loc;
  }

  template <typename TreeFunc>
  TreeInfo treeTest(TreeFunc func, const std::vector<Pin*>& load_pins, const double& skew_bound, const TopoType& topo_type,
                    const Point& guide_loc, const bool& log = false) const
  {
    auto method_name = TreeBuilder::funcName(func);
    Inst* buf = nullptr;
    if constexpr (std::is_same_v<TreeFunc, SteinerTreeFunc>) {
      buf = TreeBuilder::genBufInst(method_name, guide_loc);
      func(method_name, buf->get_driver_pin(), load_pins);
    } else if constexpr (std::is_same_v<TreeFunc, SkewTreeFunc>) {
      buf = func(method_name, load_pins, skew_bound, guide_loc, topo_type);
    } else {
      LOG_FATAL << "Unknown TreeFunc type";
    }
    auto* driver_pin = buf->get_driver_pin();
    driver_pin->preOrder([](Node* node) { node->set_pattern(static_cast<RCPattern>(1 + std::rand() % 2)); });
    buf->set_cell_master(TimingPropagator::getMinSizeCell());
    auto* net = TimingPropagator::genNet(method_name, driver_pin, load_pins);
    TimingPropagator::update(net);

    // TreeBuilder::writePy(driver_pin, method_name + "_" + TopoTypeToString(topo_type));
    auto topo_type_str = TopoTypeToString(topo_type);
    TreeInfo info{driver_pin->get_sub_len(), driver_pin->get_cap_load(), driver_pin->get_max_delay() - driver_pin->get_min_delay(),
                  driver_pin->get_max_delay() - load_pins.front()->get_inst()->get_insert_delay(), driver_pin->get_max_delay()};
    if (log) {
      LOG_INFO << std::endl;
      LOG_INFO << method_name << "(" << topo_type_str << ")";
      LOG_INFO << "wirelength: " << driver_pin->get_sub_len();
      LOG_INFO << "cap: " << driver_pin->get_cap_load();
      LOG_INFO << "skew: " << driver_pin->get_max_delay() - driver_pin->get_min_delay();
      LOG_INFO << "max wire delay: " << driver_pin->get_max_delay() - load_pins.front()->get_inst()->get_insert_delay();
      LOG_INFO << "max delay: " << driver_pin->get_max_delay();
    }
    TimingPropagator::resetNet(net);
    // CTSAPIInst.resetId();
    return info;
  }

  std::vector<Pin*> genFixedPins() const
  {
    // auto locs
    //     = std::vector<Point>{Point(128000, 154000), Point(90000, 54000),  Point(84000, 158000), Point(98000, 186000), Point(74000,
    //     98000),
    //                          Point(108000, 146000), Point(134000, 60000), Point(80000, 198000), Point(176000, 54000), Point(128000,
    //                          150000), Point(108000, 150000), Point(98000, 158000), Point(98000, 196000), Point(134000, 54000)};
    // auto locs = std::vector<Point>{Point(128000, 154000), Point(90000, 54000),   Point(84000, 158000), Point(98000, 186000),
    //                                Point(54000, 98000),   Point(108000, 146000), Point(134000, 60000), Point(80000, 198000),
    //                                Point(176000, 54000),  Point(198000, 100000), Point(178000, 80000), Point(198000, 158000),
    //                                Point(98000, 196000),  Point(134000, 54000)};
    auto locs = std::vector<Point>{Point(300000, 100000),  Point(1000000, 100000), Point(100000, 200000),  Point(800000, 200000),
                                   Point(300000, 300000),  Point(1100000, 300000), Point(100000, 400000),  Point(900000, 400000),
                                   Point(100000, 900000),  Point(400000, 900000),  Point(800000, 900000),  Point(1100000, 900000),
                                   Point(200000, 1100000), Point(500000, 1100000), Point(700000, 1100000), Point(1000000, 1100000)};
    std::vector<Inst*> load_bufs;
    for (size_t i = 0; i < locs.size(); ++i) {
      auto loc = locs[i];
      auto* buf = TreeBuilder::genBufInst(CTSAPIInst.toString("buf_", i), loc);
      buf->set_cell_master(TimingPropagator::getMinSizeCell());
      load_bufs.push_back(buf);
      auto* load_pin = buf->get_load_pin();
      auto pattern = static_cast<RCPattern>(1 + std::rand() % 2);
      load_pin->set_pattern(pattern);
      TimingPropagator::updatePinCap(load_pin);
      TimingPropagator::initLoadPinDelay(load_pin);
    }
    std::vector<Pin*> load_pins;
    std::ranges::transform(load_bufs, std::back_inserter(load_pins), [](Inst* buf) { return buf->get_load_pin(); });
    return load_pins;
  }
};

}  // namespace