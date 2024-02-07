#pragma once
#include <cstddef>
#include <functional>
#include <future>
#include <numeric>
#include <set>

#include "Block.hh"
#include "Layout.hh"
#include "Logger.hpp"
#include "SAPlacer.hh"
#include "thread"
namespace imp {

class HierPlacer
{
 public:
  explicit HierPlacer(Block& root_cluster) : _root_cluster(root_cluster) {}
  virtual void initialize(){};                // do some initilaize operations
  virtual void placeCluster(Block& blk) = 0;  // place a given cluster
  void hierPlace(bool parallel = true)
  {
    initialize();
    auto place_op = [this](Block& blk) -> void {
      if (!blk.isFixed()) {
        this->placeCluster(blk);
      }
    };
    if (parallel) {
      _root_cluster.parallel_preorder_op(place_op);
    } else {
      _root_cluster.preorder_op(place_op);
    }
  }
  ~HierPlacer() = default;

 protected:
  Block& _root_cluster;
};

template <typename T>
class SAHierPlacer : public HierPlacer
{
 public:
  explicit SAHierPlacer(Block& root_cluster, float macro_halo_micron = 2.0, float dead_space_ratio = 0.5, float weight_wl = 1.0,
                        float weight_ol = 0.05, float weight_area = 0.01, float weight_periphery = 0.05, float max_iters = 500,
                        float cool_rate = 0.97, float init_temperature = 1000.0)
      : HierPlacer(root_cluster),
        _dead_space_ratio(dead_space_ratio),
        _weight_wl(weight_wl),
        _weight_ol(weight_ol),
        _weight_area(weight_area),
        _weight_periphery(weight_periphery),
        _max_iters(max_iters),
        _cool_rate(cool_rate),
        _init_temperature(init_temperature)
  {
    _dbu = _root_cluster.netlist().property()->get_database_unit();
    _macro_halo = _dbu * macro_halo_micron;
  }
  ~SAHierPlacer() = default;

  void initialize() override
  {
    init_cell_area(_macro_halo);                 // init stdcell-area && macro-area
    coarse_shaping(generate_different_tilings);  // init discrete-shapes bottom-up (coarse-shaping, only considers macros)
  }

  void placeCluster(Block& blk)
  {
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
      clipChildrenShapes(blk);                         // clip discrete-shapes larger than parent-clusters bounding-box
      addChildrenStdcellArea(blk, _dead_space_ratio);  // add stdcell area
      INFO("start placing cluster ", blk.get_name(), ", node_num: ", blk.netlist().vSize());
      auto th = std::thread(SAPlace<T>, std::ref(blk), _weight_wl, _weight_ol, _weight_area, _weight_periphery, _max_iters, _cool_rate,
                            _init_temperature);
      th.join();
      // SAPlace<T>(blk, _weight_wl, _weight_ol, _weight_area, _weight_periphery, _max_iters, _cool_rate, _init_temperature);
    }
    blk.set_fixed();  // set placed cluster fixed
  }

  void writePlacement(Block& blk, std::string file_name)
  {
    std::ofstream out(file_name);
    auto core = _root_cluster.netlist().property()->get_core_shape();
    auto core_min_corner = core.min_corner();
    auto core_max_corner = core.max_corner();
    out << core_min_corner.x() << "," << core_min_corner.y() << "," << core_max_corner.x() - core_min_corner.x() << ","
        << core_max_corner.y() - core_min_corner.y() << std::endl;
    preorder_out(blk, out);
  }

 private:
  T _macro_halo;
  double _dbu;
  float _dead_space_ratio;
  float _weight_wl;
  float _weight_ol;
  float _weight_area;
  float _weight_periphery;
  size_t _max_iters;
  float _cool_rate;
  float _init_temperature;

  void init_cell_area(T macro_halo)
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
      INFO(obj.get_name(), " macro_area: ", macro_area, " stdcell area: ", stdcell_area, " io_area: ", io_area);
      return;
    };
    _root_cluster.postorder_op(area_op);
    INFO("total macro area: ", _root_cluster.get_macro_area());
    INFO("total stdcell area: ", _root_cluster.get_stdcell_area());
  }

  static std::vector<std::pair<T, T>> generate_different_tilings(const std::vector<ShapeCurve<T>>& sub_shape_curves, T core_width,
                                                                 T core_height, const std::string& name)
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
    // auto [core_width, core_height] = get_core_size();
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

  template <typename getPackingShapes>
  std::enable_if_t<std::is_invocable_v<getPackingShapes, std::vector<ShapeCurve<T>>, T, T, std::string>, void> coarse_shaping(
      getPackingShapes get_packing_shapes)
  // void coarse_shaping(std::function<std::vector<std::pair<T, T>>(std::vector<ShapeCurve<T>>&, T, T, const std::string&)>
  // get_packing_shapes)
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
        // blk.set_shape_curve({{macro->get_width(), macro->get_height()}}, 0, true);
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
    _root_cluster.postorder_op(coarse_shape_op);
  }

  static void clipChildrenShapes(Block& blk)
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

  static void addChildrenStdcellArea(Block& blk, float dead_space_ratio)
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
        sub_block->set_shape_curve(std::vector<std::pair<T, T>>(), (1 + stdcell_inflate_ratio) * sub_block->get_stdcell_area());
      }
    }
  }

  void preorder_out(Block& blk, std::ofstream& out)
  {
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
        out << inst->get_min_corner().x() << "," << inst->get_min_corner().y() << "," << inst->get_width() << "," << inst->get_height()
            << "," << blk.get_name() << "," << type << std::endl;
      }
      if (!obj->isBlock())
        continue;
      auto sub_block = std::static_pointer_cast<Block, Object>(obj);
      preorder_out(*sub_block, out);
    }
  }

  std::pair<T, T> get_core_size() const
  {
    return std::make_pair(_root_cluster.get_shape_curve().get_width(), _root_cluster.get_shape_curve().get_height());
  }
};

}  // namespace imp