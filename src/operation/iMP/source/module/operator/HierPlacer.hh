#pragma once
#include <cstddef>
#include <functional>
#include <future>
#include <numeric>
#include <set>

#include "Block.hh"
#include "ClusterTimingEvaluator.hh"
#include "Layout.hh"
#include "Logger.hpp"
#include "Net.hh"
#include "Pin.hh"
#include "SAPlacer.hh"
#include "thread"

namespace imp {

template <typename T>
std::vector<std::pair<T, T>> generateDifferentTilings(const std::vector<ShapeCurve<T>>& sub_shape_curves, T core_width, T core_height,
                                                      const std::string& name);

std::vector<std::shared_ptr<imp::Instance>> get_instances(Block& blk);
std::vector<std::shared_ptr<imp::Instance>> get_macros(Block& blk);

template <typename T>
struct SAHierPlacer
{
  ~SAHierPlacer() {}
  void operator()(Block& root_cluster)
  {
    // initialize
    if (init_cluster == true) {
      initialize(root_cluster);
    }

    // sta & add-virtual-net
    std::unordered_map<idb::IdbNet*, std::map<std::string, double>> net_lengths;
    // createDataflow(root_cluster, 2);
    auto negative_slack_paths = timing_evaluator.getNegativeSlackPaths(net_lengths, 1.0);
    addVirtualNet(root_cluster, negative_slack_paths);

    // Hier-place
    auto place_op = [this](Block& blk) -> void {
      if (!blk.isFixed()) {
        this->place(blk);
      }
    };
    root_cluster.parallel_preorder_op(place_op);
  }

  // settings
  float macro_halo_micron = 3.0;
  float dead_space_ratio = 0.7;
  float weight_wl = 1.0;
  float weight_ol = 0.05;
  float weight_ob = 0.01;
  float weight_periphery = 0.01;
  float weight_blk = 0.02;
  float weight_io = 0.0;
  size_t max_iters = 1000;
  double cool_rate = 0.97;
  double init_temperature = 1000;
  double prob_ps = 0.2;
  double prob_ns = 0.2;
  double prob_ds = 0.2;
  double prob_pi = 0.2;
  double prob_ni = 0.2;
  double prob_rs = 0.2;
  float virtual_net_weight = 0.5;
  int seed = 0;

  // data
  T macro_halo;
  double dbu;
  bool init_cluster = true;
  SeqPair<NodeShape<T>> solution_top_level_sp;
  SeqPair<NodeShape<T>> init_top_level_sp = SeqPair<NodeShape<T>>();
  Block* root = nullptr;

  // sta
  ClusterTimingEvaluator timing_evaluator;
  std::unordered_map<std::string, idb::IdbPin*> ista_pin_name2idb_pin;
  std::unordered_map<std::string, idb::IdbPin*> idb_name2pin;
  std::unordered_map<std::string, size_t> inst2cluster;  // top-level cluster
  std::unordered_map<std::string, std::shared_ptr<Instance>> name2inst;

  // functions
  void initialize(Block& root_cluster);
  void place(Block& blk);
  void init_cell_area(Block& root_cluster, T macro_halo);
  void initTimingEvaluator();
  void initIstaPinNameMap();
  void initInstanceInfo();
  void addVirtualNet(Block& root_cluster, std::vector<std::tuple<std::string, std::string, double>> negative_slack_paths);
  void createDataflow(Block& root_cluster, size_t max_hop);
  std::set<std::string> get_boundary_instances(Block& root_cluster);
  std::string fullPinName(idb::IdbPin* idb_pin);
  const SeqPair<NodeShape<T>>& get_sp_solution() const { return solution_top_level_sp; }

  template <typename getPackingShapes>
  std::enable_if_t<std::is_invocable_v<getPackingShapes, std::vector<ShapeCurve<T>>, T, T, std::string>, void> coarseShaping(
      Block& root_cluster, getPackingShapes get_packing_shapes)
  {
    // calculate cluster's discrete shapes based on children's discrete shapes recursively, only called on root node

    auto [core_width, core_height] = get_core_size();
    auto coarse_shape_op = [get_packing_shapes, core_width, core_height](imp::Block& blk) -> void {
      // calculate current node's discrete_shape_curve based on children node's discrete shapes, only concerns macros
      // assume children node's shape has been calculated..
      if (blk.isRoot() || blk.is_io_cluster() || blk.is_stdcell_cluster() || blk.is_io_cluster()
          || blk.isFixed()) {  // root cluster's shape is core-size
        return;
      }
      if (blk.netlist().vSize() == 1) {  // single macro cluster, set its shape as child-shape
        auto macro = std::static_pointer_cast<Instance, Object>(blk.netlist().vertex_at(0).property());
        blk.set_shape_curve({{macro->get_halo_width(), macro->get_halo_height()}}, 0, true);  // use halo width & height
        return;
      }

      std::vector<ShapeCurve<T>> sub_shape_curves;
      for (auto&& i : blk.netlist().vRange()) {
        auto sub_obj = i.property();
        if (sub_obj->isInstance()) {
          auto sub_inst = std::static_pointer_cast<Instance, Object>(sub_obj);
          if (!sub_inst->get_cell_master().isMacro()) {
            ERROR("Instance ", sub_inst->get_name(), " in in cluster hierarchy ", blk.get_name());
            throw std::runtime_error("Instance in cluster hierarchy");
          }
          // create discrete shape-curve for macro
          auto macro_shape_curve = ShapeCurve<T>();
          // macro_shape_curve.setShapes({{sub_inst->get_width(), sub_inst->get_height()}}, 0, false);
          macro_shape_curve.setShapes({{sub_inst->get_halo_width(), sub_inst->get_halo_height()}}, 0, false);
          sub_shape_curves.push_back(std::move(macro_shape_curve));
        } else {
          auto sub_block = std::static_pointer_cast<Block, Object>(sub_obj);
          if (sub_block->is_macro_cluster() || sub_block->is_mixed_cluster()) {
            sub_shape_curves.push_back(sub_block->get_shape_curve());
          }
        }
      }

      // calculate possible tilings of children clusters & update discrete-shape curve
      auto possible_discrete_shapes = get_packing_shapes(sub_shape_curves, core_width, core_height, blk.get_name());
      blk.set_shape_curve(possible_discrete_shapes, 0, true);
      INFO(blk.get_name(), " clipped shape width: ", blk.get_shape_curve().get_width(), " height: ", blk.get_shape_curve().get_height(),
           " shape_curve_size: ", blk.get_shape_curve().get_width_list().size(), " area: ", blk.get_shape_curve().get_area());
    };
    root_cluster.postorder_op(coarse_shape_op);
  }

  std::pair<T, T> get_core_size() const
  {
    return std::make_pair(root->get_shape_curve().get_width(), root->get_shape_curve().get_height());
  }
};

template <typename T>
std::vector<std::pair<T, T>> generateDifferentTilings(const std::vector<ShapeCurve<T>>& sub_shape_curves, T core_width, T core_height,
                                                      const std::string& name);
template <typename T>
void SAHierPlacer<T>::initialize(Block& root_cluster)
{
  root = &root_cluster;
  dbu = root_cluster.netlist().property()->get_database_unit();
  macro_halo = dbu * macro_halo_micron;

  // init shape-curve
  std::cout << "init cluster area && coarse shaping..." << std::endl;
  init_cell_area(root_cluster, macro_halo);                  // init stdcell-area && macro-area
  coarseShaping(root_cluster, generateDifferentTilings<T>);  // init discrete-shapes bottom-up (coarse-shaping, only considers macros)

  // init timing-engine;
  initTimingEvaluator();
}

template <typename T>
void SAHierPlacer<T>::initTimingEvaluator()
{
  timing_evaluator.initTimingEngine();
  initInstanceInfo();
  initIstaPinNameMap();
}

template <typename T>
void SAHierPlacer<T>::place(Block& blk)
{
  void clipChildrenShapes(Block & blk);
  void addChildrenStdcellArea(Block & blk, float dead_space_ratio);

  if (blk.netlist().vSize() == 0 || blk.isFixed() || blk.is_stdcell_cluster() || blk.is_io_cluster()) {
    return;  // only place cluster with macros..
  }

  // only one children nodes, place it at min_corner of parent cluster
  if (blk.netlist().vSize() == 1) {
    auto sub_obj = blk.netlist().vertex_at(0).property();
    if (sub_obj->isBlock()) {
      sub_obj->set_min_corner(blk.get_min_corner());
    } else {
      // place Instance's halo_min_corner at min_corner of parent cluster
      std::static_pointer_cast<Instance, Object>(sub_obj)->set_halo_min_corner(blk.get_min_corner());
    }
  }

  else {
    if (init_cluster) {
      std::cout << "add children shapes" << std::endl;
      clipChildrenShapes(blk);                        // clip discrete-shapes larger than parent-clusters bounding-box
      addChildrenStdcellArea(blk, dead_space_ratio);  // add stdcell area
    }
    // INFO("start placing cluster ", blk.get_name(), ", node_num: ", blk.netlist().vSize());
    // auto th = std::thread(SAPlace<T>(_weight_wl, _weight_ol, _weight_ob, _weight_periphery, _max_iters, _cool_rate, _init_temperature),
    //                       std::ref(blk));
    // th.join();
    if (blk.isRoot()) {
      SAPlace<T> placer{.seed = seed,
                        .weight_wl = weight_wl,
                        .weight_ol = weight_ol,
                        .weight_ob = weight_ob,
                        .weight_periphery = weight_periphery,
                        .weight_blk = weight_blk,
                        .weight_io = weight_io,
                        .max_iters = max_iters,
                        .cool_rate = cool_rate,
                        .init_temperature = init_temperature,
                        .prob_ps_op = prob_ps,
                        .prob_ns_op = prob_ns,
                        .prob_ds_op = prob_ds,
                        .prob_pi_op = prob_pi,
                        .prob_ni_op = prob_ni,
                        .prob_rs_op = prob_rs};

      solution_top_level_sp = placer(blk);
    } else {
      SAPlace<T> placer{.seed = seed,
                        .weight_wl = weight_wl,
                        .weight_ol = weight_ol,
                        .weight_ob = weight_ob,
                        .weight_periphery = weight_periphery,
                        .weight_blk = weight_blk,
                        .weight_io = weight_io,
                        .max_iters = max_iters,
                        .cool_rate = cool_rate,
                        .init_temperature = init_temperature,
                        .prob_ps_op = prob_ps,
                        .prob_ns_op = prob_ns,
                        .prob_ds_op = prob_ds,
                        .prob_pi_op = prob_pi,
                        .prob_ni_op = prob_ni,
                        .prob_rs_op = prob_rs,
                        .init_represent = init_top_level_sp};
      auto not_used_sp = placer(blk);
    }
  }
}

template <typename T>
void SAHierPlacer<T>::init_cell_area(Block& root_cluster, T macro_halo)
{
  auto area_op = [macro_halo](imp::Block& obj) -> void {
    obj.set_macro_area(0.);
    obj.set_stdcell_area(0.);

    size_t macro_num = 0;
    float macro_area = 0, stdcell_area = 0, io_area = 0;
    for (auto&& i : obj.netlist().vRange()) {
      auto sub_obj = i.property();
      if (sub_obj->isInstance()) {  // add direct instance child area
        auto sub_inst = std::static_pointer_cast<Instance, Object>(sub_obj);
        if (sub_inst->get_cell_master().isMacro()) {
          macro_num++;
          macro_area += sub_inst->get_area();
          // add halo for macros
          sub_inst->set_extend_left(macro_halo);
          sub_inst->set_extend_right(macro_halo);
          sub_inst->set_extend_bottom(macro_halo);
          sub_inst->set_extend_top(macro_halo);
        } else if (sub_inst->get_cell_master().isIOCell()) {
          io_area += 1;  // assume io-cluster has area 1
        } else {
          stdcell_area += sub_inst->get_area();
        }
      } else {  // add block children's instance area
        auto sub_block = std::static_pointer_cast<Block, Object>(sub_obj);
        macro_num += sub_block->get_macro_num();
        macro_area += sub_block->get_macro_area();
        stdcell_area += sub_block->get_stdcell_area();
        io_area += sub_block->get_io_area();
      }
    }
    obj.set_macro_num(macro_num);
    obj.set_macro_area(macro_area);
    obj.set_stdcell_area(stdcell_area);
    obj.set_io_area(io_area);

    // set io-cluster's location and fix it.
    if (obj.is_io_cluster()) {
      float mean_x = 0, mean_y = 0;
      for (auto&& i : obj.netlist().vRange()) {
        auto min_corner = i.property()->get_min_corner();
        mean_x += min_corner.x();
        mean_y += min_corner.y();
      }
      mean_x /= obj.netlist().vSize();
      mean_y /= obj.netlist().vSize();
      obj.set_min_corner(mean_x, mean_y);
      obj.set_shape_curve(geo::make_box(0, 0, 0, 0));  // io-cluster 0 area
      obj.set_fixed();
    }
    // INFO(obj.get_name(), " macro_area: ", macro_area, " stdcell area: ", stdcell_area, " io_area: ", io_area);
    return;
  };
  root_cluster.postorder_op(area_op);
  // INFO("total macro area: ", get_root_cluster().get_macro_area());
  // INFO("total stdcell area: ", get_root_cluster().get_stdcell_area());
}

template <typename T>
void SAHierPlacer<T>::initIstaPinNameMap()
{
  // init idb_pin_name_map
  ista_pin_name2idb_pin.clear();
  for (idb::IdbNet* idb_net : dmInst->get_idb_design()->get_net_list()->get_net_list()) {
    for (auto pin : idb_net->get_instance_pin_list()->get_pin_list()) {
      idb_name2pin[pin->get_pin_name()] = pin;
      std::string pin_name = fullPinName(pin);
      std::string pin_name_trim_slash = Str::trimBackslash(pin_name);
      ista_pin_name2idb_pin[pin_name_trim_slash] = pin;
    }
    if (idb_net->has_io_pins()) {
      for (auto pin : idb_net->get_io_pins()->get_pin_list()) {
        idb_name2pin[pin->get_pin_name()] = pin;
        std::string pin_name = fullPinName(pin);
        std::string pin_name_trim_slash = Str::trimBackslash(pin_name);
        ista_pin_name2idb_pin[pin_name_trim_slash] = pin;
      }
    }
  }
}

template <typename T>
void SAHierPlacer<T>::addVirtualNet(Block& root_cluster, std::vector<std::tuple<std::string, std::string, double>> negative_slack_paths)
{
  size_t valid_paths = 0;
  size_t pin_not_found = 0;
  size_t inst_not_found = 0;
  size_t small_weight = 0;
  size_t same_cluster = 0;
  std::vector<double> initial_slacks;
  std::vector<double> smooth_weights;
  std::map<std::pair<size_t, size_t>, double> total_slack_between_cluster;
  float total_outlier_slack = 0;
  size_t outlier_slack_num = 0;
  double min_slack = -2.0;
  double max_slack = 0.1;
  double max_weight = 2.0;
  double min_weight = 0.05;
  for (size_t i = 0; i < negative_slack_paths.size(); ++i) {
    auto [start_pin_name, end_pin_name, slack] = negative_slack_paths[i];
    // std::cout << "oslack: " << slack << ", ";
    if (slack < -5) {
      ++outlier_slack_num;
      total_outlier_slack += outlier_slack_num;
      // continue;  // some wrong paths ?
    }
    if (slack > max_slack) {
      continue;
    } else if (slack < min_slack) {
      slack = min_slack;
    }

    std::replace(start_pin_name.begin(), start_pin_name.end(), ':', '/');
    std::replace(end_pin_name.begin(), end_pin_name.end(), ':', '/');
    if (ista_pin_name2idb_pin.find(start_pin_name) == ista_pin_name2idb_pin.end()
        || ista_pin_name2idb_pin.find(end_pin_name) == ista_pin_name2idb_pin.end()) {
      std::cout << "sta-pin not found..." << std::endl;
      std::cout << "start_pin: " << start_pin_name << std::endl;
      std::cout << "end_pin: " << end_pin_name << std::endl;
      pin_not_found++;
      continue;
    }
    // float net_weight = smooth_weight(slack);
    // float net_weight = std::min(std::max(pow(-(slack - 0.2), 0.5), min_weight), max_weight);
    float net_weight = virtual_net_weight * std::min(std::max(pow(-(slack - max_slack), 1.0), min_weight), max_weight);
    // if (net_weight < _min_net_weight) {
    //   small_weight++;
    //   continue;
    // }
    size_t start_cluster_id = 0, end_cluster_id = 0;
    auto start_pin = ista_pin_name2idb_pin.at(start_pin_name);
    auto end_pin = ista_pin_name2idb_pin.at(end_pin_name);
    if (start_pin->is_io_pin()) {
      start_cluster_id = inst2cluster.at(start_pin->get_pin_name());
    } else if (start_pin->get_instance() != nullptr) {
      start_cluster_id = inst2cluster.at(start_pin->get_instance()->get_name());
    } else {
      inst_not_found++;
      continue;
    }

    if (end_pin->is_io_pin()) {
      end_cluster_id = inst2cluster.at(end_pin->get_pin_name());
    } else if (end_pin->get_instance() != nullptr) {
      end_cluster_id = inst2cluster.at(end_pin->get_instance()->get_name());
    } else {
      inst_not_found++;
      continue;
    }

    if (start_cluster_id == end_cluster_id) {
      same_cluster++;
      continue;  // start && end in same cluster
    }

    if (slack < -5) {
      std::cout << "wrong path: " << root_cluster.netlist().vertex_at(start_cluster_id).property()->get_name();
      std::cout << ", " << root_cluster.netlist().vertex_at(end_cluster_id).property()->get_name() << std::endl;
      continue;
    }

    initial_slacks.push_back(slack);
    smooth_weights.push_back(net_weight);
    total_slack_between_cluster[std::make_pair(start_cluster_id, end_cluster_id)] += net_weight;  // count virtual nets between clusters
    // std::cout << "start id: " << start_cluster_id << ", end id: " << end_cluster_id << " slack:" << slack << std::endl;
    // addVirtualNet(root_cluster, start_cluster_id, end_cluster_id, net_weight);
    valid_paths++;
  }

  void addVirtualNet(Block & parent_blk, size_t sub_obj_pos1, size_t sub_obj_pos2, float net_weight);
  for (auto& [k, v] : total_slack_between_cluster) {
    addVirtualNet(root_cluster, k.first, k.second, v);  //
  }

  std::vector<std::pair<float, std::pair<size_t, size_t>>> total_slack_between_cluster_sort;
  for (auto& [k, v] : total_slack_between_cluster) {
    total_slack_between_cluster_sort.push_back(std::make_pair(v, k));
  }
  std::sort(total_slack_between_cluster_sort.begin(), total_slack_between_cluster_sort.end(),
            [](const std::pair<float, std::pair<size_t, size_t>>& x1, const std::pair<float, std::pair<size_t, size_t>>& x2) -> bool {
              return x1.first < x2.first;
            });
  for (size_t i = 0; i < total_slack_between_cluster_sort.size(); ++i) {
    std::cout << "begin: " << total_slack_between_cluster_sort[i].second.first
              << ", end: " << total_slack_between_cluster_sort[i].second.second
              << ", total_net_weight: " << total_slack_between_cluster_sort[i].first << std::endl;
    auto begin = std::static_pointer_cast<Block, Object>(
        root_cluster.netlist().vertex_at(total_slack_between_cluster_sort[i].second.first).property());
    auto end = std::static_pointer_cast<Block, Object>(
        root_cluster.netlist().vertex_at(total_slack_between_cluster_sort[i].second.second).property());
    std::string begin_type, end_type;
    if (begin->is_io_cluster()) {
      begin_type = "io";
    } else if (begin->is_macro_cluster()) {
      begin_type = "macro";
    } else {
      begin_type = "stdcell";
    }
    if (end->is_io_cluster()) {
      end_type = "io";
    } else if (end->is_macro_cluster()) {
      end_type = "macro";
    } else {
      end_type = "stdcell";
    }
    std::cout << "begin: " << begin_type << ", " << begin->get_name() << std::endl;
    std::cout << "end: " << end_type << ", " << end->get_name() << std::endl;
  }

  std::cout << "outlier_slack_num: " << outlier_slack_num << std::endl;
  std::cout << "total_outlier_slack: " << total_outlier_slack << std::endl;
  std::cout << "pin_not_found: " << pin_not_found << std::endl;
  std::cout << "inst_not_found: " << inst_not_found << std::endl;
  std::cout << "small_weight: " << small_weight << std::endl;
  std::cout << "same_cluster: " << same_cluster << std::endl;
  std::cout << "add " << valid_paths << " path" << std::endl;
  std::cout << "add " << total_slack_between_cluster.size() << " virtual nets" << std::endl;
  for (size_t i = 0; i < initial_slacks.size(); ++i) {
    std::cout << "slack: " << initial_slacks[i] << ", smooth_weight: " << smooth_weights[i] << ", ";
  }
  std::cout << std::endl;
}

template <typename T>
void SAHierPlacer<T>::initInstanceInfo()
{
  std::vector<std::shared_ptr<imp::Instance>> get_instances(Block&);

  name2inst.clear();
  inst2cluster.clear();
  for (size_t v_id = 0; v_id < root->netlist().vSize(); ++v_id) {
    auto sub_obj = root->netlist().vertex_at(v_id).property();
    auto sub_blk = std::static_pointer_cast<Block, Object>(sub_obj);
    std::vector<std::shared_ptr<imp::Instance>> instances = get_instances(*sub_blk);
    for (auto inst : instances) {
      // if (inst->get_cell_master().isIOCell()) {
      //   continue;
      // }
      // IO-CELL uses pin-name as instance-name
      name2inst[inst->get_name()] = inst;
      inst2cluster[inst->get_name()] = v_id;
    }
  }
}
template <typename T>
std::string SAHierPlacer<T>::fullPinName(idb::IdbPin* idb_pin)
{
  if (idb_pin->get_instance() != nullptr) {
    return idb_pin->get_instance()->get_name() + "/" + idb_pin->get_pin_name();
  } else {
    return idb_pin->get_pin_name();
  }
}

template <typename T>
void SAHierPlacer<T>::createDataflow(Block& root_cluster, size_t max_hop)
{
  std::vector<std::set<std::string>> cluster_instances;
  std::vector<std::set<std::string>> cluster_src_instances;

  std::set<std::string> inst_count;
  // std::set<std::string> boundary_inst_names = get_boundary_instances(root_cluster);
  for (size_t i = 0; i < root_cluster.netlist().vSize(); ++i) {
    std::set<std::string> inst_set;
    std::set<std::string> src_inst_set;
    auto sub_blk = *(std::static_pointer_cast<Block, Object>(root_cluster.netlist().vertex_at(i).property()));

    auto inst_list = get_instances(sub_blk);
    for (auto&& inst : inst_list) {
      std::string inst_name = Str::trimBackslash(inst->get_name());
      if (inst_count.count(inst_name) != 0) {
        ERROR("Error, same instance in multi-clusters");
      }
      inst_set.insert(inst_name);
      if (inst->get_cell_master().isMacro()) {
        src_inst_set.insert(inst_name);
      }
    }

    cluster_instances.push_back(std::move(inst_set));
    cluster_src_instances.push_back(std::move(src_inst_set));

  }
  timing_evaluator.createDataflow(cluster_instances, cluster_src_instances, max_hop);
}

// template <typename T>
// std::set<std::string> SAHierPlacer<T>::get_boundary_instances(Block& root_cluster)
// {
//   std::set<std::string> boundary_inst_names;
//   auto& net2idb = ParserInst()->get_net2idb();
//   for (auto&& he : root_cluster.netlist().heRange()) {
//     auto idb_net = net2idb.at(he.property());
//     for (auto inst : idb_net->get_instance_list()->get_instance_list()) {
//       boundary_inst_names.insert(Str::trimBackslash(inst->get_name()));
//     }
//   }
//   return boundary_inst_names;
// }

template <typename T>
std::vector<std::pair<T, T>> generateDifferentTilings(const std::vector<ShapeCurve<T>>& sub_shape_curves, T core_width, T core_height,
                                                      const std::string& name)
{
  if (sub_shape_curves.empty()) {
    throw std::runtime_error("no shapes to place!");
  }
  if (sub_shape_curves.size() == 1) {
    // maybe only one child macro-cluster, return it's possbile-discrete-shapes
    INFO("only one shapes here! ");
    return sub_shape_curves[0].get_discrete_shapes();
  }

  size_t num_runs = 10;
  std::vector<T> outline_width_list;
  std::vector<T> outline_height_list;
  float width_unit = float(core_width) / num_runs;
  float height_unit = float(core_height) / num_runs;
  for (size_t i = 1; i <= num_runs; ++i) {
    outline_width_list.emplace_back(core_width);
    outline_height_list.emplace_back(width_unit * i);  // vary outline-height
    outline_height_list.emplace_back(core_height);
    outline_width_list.emplace_back(height_unit * i);  // vary outline-width
  }

  std::vector<std::thread> threads;
  threads.reserve(outline_width_list.size());
  std::vector<std::promise<std::pair<T, T>>> promises(outline_width_list.size());
  std::vector<std::future<std::pair<T, T>>> futures;
  for (size_t i = 0; i < outline_width_list.size(); ++i) {
    futures.push_back(promises[i].get_future());
    // t.detach();
    threads.emplace_back([&promises, &sub_shape_curves, &outline_width_list, &outline_height_list, name, i] {
      std::pair<T, T> shape
          = calMacroTilings<T>(sub_shape_curves, outline_width_list[i], outline_height_list[i], name + "_run" + std::to_string(i));
      promises[i].set_value(shape);
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  std::set<std::pair<T, T>> tilings_set;  // remove same tilings
  for (auto&& future : futures) {
    tilings_set.insert(future.get());
  }
  std::vector<std::pair<T, T>> tilings;
  tilings.reserve(tilings_set.size());
  for (auto& shape : tilings_set) {
    tilings.push_back(shape);
  }

  INFO("child cluster num: ", sub_shape_curves.size(), ", generated tilings num: ", tilings.size());
  return tilings;
}

void clipChildrenShapes(Block& blk)
{
  // remove child cluster's shapes larger than current-node's bounding-box,
  auto bound_width = blk.get_shape_curve().get_width();
  auto bound_height = blk.get_shape_curve().get_height();
  for (auto&& i : blk.netlist().vRange()) {
    auto sub_obj = i.property();
    if (sub_obj->isInstance()) {  // instance not supported..
      throw std::runtime_error("try to clip instance");
    }

    // only clip clusters with macros
    auto sub_block = std::static_pointer_cast<Block, Object>(sub_obj);
    if (!(sub_block->is_macro_cluster() || sub_block->is_mixed_cluster())) {
      continue;
    }
    auto clipped_shape_curve = sub_block->get_shape_curve();
    clipped_shape_curve.clip(bound_width, bound_height);
    sub_block->set_shape_curve(clipped_shape_curve);
  }
}

void addChildrenStdcellArea(Block& blk, float dead_space_ratio)
{
  float bound_area = blk.get_shape_curve().get_area();
  float mixed_cluster_stdcell_area = 0;
  float stdcell_cluster_area = 0;
  float macro_area = 0;

  for (auto&& i : blk.netlist().vRange()) {
    auto sub_obj = i.property();
    if (sub_obj->isInstance()) {  // 目前不考虑中间层有单独stdcell情况
      throw std::runtime_error("Instance in cluster hierarchy");
    }

    auto sub_block = std::static_pointer_cast<Block, Object>(sub_obj);
    if (sub_block->is_macro_cluster()) {
      macro_area += sub_block->get_shape_curve().get_area();  // shape-curve has only macro area now..
    } else if (sub_block->is_mixed_cluster()) {
      macro_area += sub_block->get_shape_curve().get_area();  // shape-curve has only macro area now..
      mixed_cluster_stdcell_area += sub_block->get_stdcell_area();
    } else if (sub_block->is_stdcell_cluster()) {
      stdcell_cluster_area += sub_block->get_stdcell_area();
    }
  }

  // 假设每一层级，剩余空间的一半用来膨胀单元，一半用来留空。(先用mixed-cluster和 stdcell-cluster相同膨胀率)
  // 考虑mixed-cluster需要后续布局，让它膨胀率为stdcell 2倍吧)
  float area_left = bound_area - macro_area - stdcell_cluster_area - mixed_cluster_stdcell_area;
  if (area_left < 0) {
    INFO("------- fine-shaping cluster ", blk.get_name(), "--------");
    INFO("bound_area: ", bound_area);
    INFO("macro_cluster_area: ", macro_area);
    INFO("real macro area: ", blk.get_macro_area());
    INFO("stdcell_cluster_area: ", stdcell_cluster_area);
    INFO("mixed_cluster_stdcell_area: ", mixed_cluster_stdcell_area);
    INFO("area left: ", area_left);
    throw std::runtime_error("Error: Not enough area left...");
  }
  float inflate_area_for_stdcell = area_left * (1 - dead_space_ratio);
  float stdcell_inflate_ratio = inflate_area_for_stdcell / (2 * mixed_cluster_stdcell_area + stdcell_cluster_area);
  float mixed_cluster_stdcell_inflate_ratio = 2 * stdcell_inflate_ratio;

  for (auto&& i : blk.netlist().vRange()) {
    auto sub_obj = i.property();
    // add stdcell area to discrete-shape-curve
    if (sub_obj->isInstance()) {  // 目前不考虑中间层有instance
      throw std::runtime_error("Instance in cluster hierarchy");
    }
    auto sub_block = std::static_pointer_cast<Block, Object>(sub_obj);
    if (sub_block->is_mixed_cluster()) {
      auto new_shape_curve = sub_block->get_shape_curve();
      new_shape_curve.add_continous_area((1 + mixed_cluster_stdcell_inflate_ratio) * sub_block->get_stdcell_area());
      sub_block->set_shape_curve(new_shape_curve);
    } else if (sub_block->is_stdcell_cluster()) {
      auto discrete_shapes = std::vector<
          std::pair<decltype(sub_block->get_shape_curve().get_width()), decltype(sub_block->get_shape_curve().get_width())>>();
      sub_block->set_shape_curve(discrete_shapes, (1 + stdcell_inflate_ratio) * sub_block->get_stdcell_area());
    }
  }
}

void preorder_out(Block& blk, std::ofstream& out)
{
  // if (blk.isRoot()) {
  //   auto fence = blk.get_fence_region();
  //   out << fence.min_corner().x() << "," << fence.min_corner().y() << "," << fence.max_corner().x() - fence.min_corner().x() << ","
  //       << fence.max_corner().y() - fence.min_corner().y() << "," << blk.get_name() << ","
  //       << "fence" << std::endl;
  // }
  out << blk.get_min_corner().x() << "," << blk.get_min_corner().y() << "," << blk.get_shape_curve().get_width() << ","
      << blk.get_shape_curve().get_height() << "," << blk.get_name() << ","
      << "cluster" << std::endl;
  if (blk.is_stdcell_cluster()) {
    return;
  }
  for (auto&& i : blk.netlist().vRange()) {
    auto obj = i.property();
    if (obj->isInstance()) {
      std::string type;
      auto inst = std::static_pointer_cast<Instance, Object>(obj);
      if (inst->get_cell_master().isMacro()) {
        type = "macro";
      } else if (inst->get_cell_master().isIOCell()) {
        type = "io";
      } else {
        continue;
      }
      // print macro & io info
      out << inst->get_min_corner().x() << "," << inst->get_min_corner().y() << "," << inst->get_width() << "," << inst->get_height() << ","
          << blk.get_name() << "," << type << std::endl;
    }
    if (!obj->isBlock())
      continue;
    auto sub_block = std::static_pointer_cast<Block, Object>(obj);
    preorder_out(*sub_block, out);
  }
}

void preorder_get_macros(Block& blk, std::vector<std::shared_ptr<imp::Instance>>& macros)
{
  for (auto&& i : blk.netlist().vRange()) {
    auto sub_obj = i.property();
    if (sub_obj->isInstance()) {
      auto sub_inst = std::static_pointer_cast<Instance, Object>(sub_obj);
      if (sub_inst->get_cell_master().isMacro()) {
        macros.push_back(sub_inst);
      }
    } else {
      auto sub_block = std::static_pointer_cast<Block, Object>(sub_obj);
      preorder_get_macros(*sub_block, macros);
    }
  }
}

std::vector<std::shared_ptr<imp::Instance>> get_macros(Block& blk)
{
  std::vector<std::shared_ptr<imp::Instance>> macros;
  preorder_get_macros(blk, macros);
  return macros;
}

void writePlacement(Block& root_cluster, std::string file_name)
{
  std::ofstream out(file_name);
  auto core = root_cluster.netlist().property()->get_core_shape();
  auto core_min_corner = core.min_corner();
  auto core_max_corner = core.max_corner();
  out << core_min_corner.x() << "," << core_min_corner.y() << "," << core_max_corner.x() - core_min_corner.x() << ","
      << core_max_corner.y() - core_min_corner.y() << std::endl;
  preorder_out(root_cluster, out);
}

std::string orientToInnovusStr(const Orient& orient)
{
  switch (orient) {
    case Orient::kNone:
      return "None";
    case Orient::kN_R0:
      return "R0";
    case Orient::kW_R90:
      return "R90";
    case Orient::kS_R180:
      return "R180";
    case Orient::kE_R270:
      return "R270";
    case Orient::kFN_MY:
      return "MY";
    case Orient::kFE_MY90:
      return "MY90";
    case Orient::kFS_MX:
      return "MX";
    case Orient::kFW_MX90:
      return "MX90";
  }
  return "None";
}

void writePlacementTcl(Block& blk, std::string file_name, int32_t dbu)
{
  auto macros = get_macros(blk);
  std::ofstream out(file_name, std::ios::binary);
  if (!out) {
    ERROR("Cannot create file " + file_name);
  }

  for (const auto& macro : macros) {
    out << "placeInstance " << macro->get_name() << " " << macro->get_min_corner().x() / dbu << " " << macro->get_min_corner().y() / dbu
        << " " << orientToInnovusStr(macro->get_orient()) << std::endl;
    out << "setInstancePlacementStatus -status "
        << "fixed"
        << " -name " << macro->get_name() << std::endl;
  }
  INFO(file_name, " write success");
}

void preorder_get_instances(Block& blk, std::vector<std::shared_ptr<imp::Instance>>& instances)
{
  for (auto&& i : blk.netlist().vRange()) {
    auto sub_obj = i.property();
    if (sub_obj->isInstance()) {
      auto sub_inst = std::static_pointer_cast<Instance, Object>(sub_obj);
      instances.push_back(sub_inst);
    } else {
      auto sub_block = std::static_pointer_cast<Block, Object>(sub_obj);
      preorder_get_instances(*sub_block, instances);
    }
  }
}

std::vector<std::shared_ptr<imp::Instance>> get_instances(Block& blk)
{
  std::vector<std::shared_ptr<imp::Instance>> instances;
  preorder_get_instances(blk, instances);
  return instances;
}

void addVirtualNet(Block& parent_blk, size_t sub_obj_pos1, size_t sub_obj_pos2, float net_weight = 1.0)
{
  if (sub_obj_pos1 > parent_blk.netlist().vSize() || sub_obj_pos2 > parent_blk.netlist().vSize()) {
    throw std::runtime_error("Error, sub_obj_pos invalid!");
  }
  if (sub_obj_pos1 == sub_obj_pos2) {
    return;
  }
  std::vector<std::shared_ptr<imp::Pin>> pins;
  std::vector<size_t> sub_obj_pos = {sub_obj_pos1, sub_obj_pos2};
  // pins.push_back(std::make_shared<imp::Pin>("virtual_pin_" + std::to_string(_virtual_pin_id++)));
  // pins.push_back(std::make_shared<imp::Pin>("virtual_pin_" + std::to_string(_virtual_pin_id++)));
  pins.push_back(std::make_shared<imp::Pin>("virtual_pin"));
  pins.push_back(std::make_shared<imp::Pin>("virtual_pin"));
  // for (size_t i = 0; i < pins.size(); ++i) {
  //   pins[i]->set_offset(0);  // not setting pin-offset currently
  //   // pins[i]->set_pin_type(PIN_TYPE::kInstancePort);
  //   // pin_ptr->set_pin_io_type(PIN_IO_TYPE::kNone);
  // }

  auto net_ptr = std::make_shared<Net>("virtual_net");
  // auto net_ptr = std::make_shared<Net>("virtual_net_" + std::to_string(_virtual_net_id));
  net_ptr->set_net_type(NET_TYPE::kFakeNet);
  net_ptr->set_net_weight(net_weight);
  add_net(parent_blk.netlist(), sub_obj_pos, pins, net_ptr);
}

}  // namespace imp