#pragma once
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

#include "../formula/Hpwl.hh"
#include "../representation/SeqPair.hh"
#include "../search_algoritm/Evaluator.hh"
#include "../search_algoritm/Operator.hh"
#include "../search_algoritm/SimulateAnneal.hh"
#include "Block.hh"
#include "HyperGraphAlgorithm.hh"
#include "Logger.hpp"

namespace imp {

struct Prop
{
};

template <typename T>
struct NodeShape
{
  NodeShape() : width(0), height(0), _shape_curve(nullptr) {}
  NodeShape(T w, T h) : width(w), height(h), _shape_curve(nullptr) {}
  explicit NodeShape(const ShapeCurve<T>* shape_curve)
  {
    // shape with shape-curve
    _shape_curve = shape_curve;
    width = _shape_curve->get_width();
    height = _shape_curve->get_height();
  }

  bool resize(std::mt19937& generator)
  {
    if (_shape_curve != nullptr && _shape_curve->isResizable()) {
      std::uniform_real_distribution<float> distribution(0., 1.);
      auto [w, h] = _shape_curve->generateRandomShape(distribution, generator);
      width = w;
      height = h;
      return true;
    }
    return false;
  }

  bool is_rotate = false;
  T width;
  T height;

 private:
  const ShapeCurve<T>* _shape_curve;
};

struct Coordinate
{
  int32_t width{0};
  int32_t height{0};
  std::vector<int32_t> x{};
  std::vector<int32_t> y{};
};

template <typename T>
bool SAPlace(Block& cluster, float weight_wl = 1.0, float weight_ol = 0.1, float weight_area = 0.1)
{
  /**
   * @brief place given cluster based on SA method
   *
   * @param cluster parent cluster
   * @param weight_wl cost-weight of wirelength
   * @param weight_ol cost-weight of outline
   * @param weight_area cost-weight of area
   * @return success flag
   */
  if (cluster.netlist().vSize() == 0) {
    throw std::runtime_error("try to place cluster with 0 nodes...");
  }
  if (cluster.netlist().vSize() == 1) {  // single-node cluster, place it at the min_corner of parent-cluster
    auto sub_obj = cluster.netlist().vertex_at(0).property();
    // update shape && locaton
    sub_obj->set_min_corner(cluster.get_min_corner());
  }

  using DimFunc = FastPackSP<NodeShape<T>, Coordinate>::DimFunc;
  using IgnoreFunc = FastPackSP<NodeShape<T>, Coordinate>::IgnoreFunc;
  using Decoder = Evaluator<SeqPair<NodeShape<T>>, Coordinate>::Decoder;
  using CostFunc = Evaluator<SeqPair<NodeShape<T>>, Coordinate>::CostFunc;
  using PerturbFunc = Perturb<SeqPair<NodeShape<T>>, void>::PerturbFunc;

  // initialize input data
  auto start = std::chrono::high_resolution_clock::now();
  size_t num_vertices = cluster.netlist().vSize();
  size_t num_edges = cluster.netlist().heSize();
  std::vector<NodeShape<T>> blk_shapes(num_vertices);
  std::vector<bool> ignore(num_vertices, false);
  std::vector<int32_t> initial_lx(num_vertices, 0);
  std::vector<int32_t> initial_ly(num_vertices, 0);

  // place-outline is based on parent cluster's shape & location,  assume parent cluster's shape & location has been decided
  auto outline_min_corner = cluster.get_min_corner();
  int32_t outline_lx = outline_min_corner.get<0>();
  int32_t outline_ly = outline_min_corner.get<1>();
  int32_t outline_width = cluster.get_shape_curve().get_width();
  int32_t outline_height = cluster.get_shape_curve().get_height();

  for (auto v_iter = cluster.netlist().vbegin(); v_iter != cluster.netlist().vend(); ++v_iter) {
    auto sub_obj = (*v_iter).property();
    auto v_pos = (*v_iter).pos();

    if (sub_obj->isBlock()) {
      auto sub_block = std::static_pointer_cast<Block, Object>(sub_obj);
      blk_shapes[v_pos] = NodeShape(&(sub_block->get_shape_curve()));
      if (sub_block->isFixed()) {
        ignore[v_pos] = true;
        continue;
      }
    } else {
      auto sub_inst = std::static_pointer_cast<Instance, Object>(sub_obj);
      blk_shapes[v_pos] = NodeShape(sub_inst->get_cell_master().get_width(), sub_inst->get_cell_master().get_height());
      if (sub_inst->isFixed()) {
        ignore[v_pos] = true;
        continue;
      }
    }
    auto min_corner = sub_obj->get_min_corner();
    initial_lx[v_pos] = min_corner.get<0>();
    initial_ly[v_pos] = min_corner.get<1>();
  }

  auto&& [eptr, eind] = vectorize(cluster.netlist());
  std::vector<int32_t> net_weight(num_edges, 1);
  Coordinate packing_result{.x = initial_lx, .y = initial_ly};

  auto end = std::chrono::high_resolution_clock::now();
  INFO("SAPlace data initialize time: ", std::chrono::duration<double>(end - start).count(), "s");
  int seed = 0;
  std::mt19937 gen(seed);
  SeqPair<NodeShape<T>> sp(blk_shapes, gen);

  DimFunc get_blk_width = [&](size_t i, const NodeShape<T>& b) { return b.is_rotate ? b.height : b.width; };
  DimFunc get_blk_height = [&](size_t i, const NodeShape<T>& b) { return b.is_rotate ? b.width : b.height; };
  IgnoreFunc is_ignore = [&](size_t i, const NodeShape<T>&) { return ignore[i]; };

  // packing function to get coordinate
  FastPackSP<NodeShape<T>, Coordinate> pack(outline_lx, outline_ly, get_blk_width, get_blk_height, is_ignore);
  Decoder decoder = pack;

  // only use outline && area cost

  Hpwl<int32_t> hpwl(eptr, eind, {}, {}, {}, 1);
  CostFunc wl = EvalWirelength(outline_width, outline_height, hpwl, net_weight);
  CostFunc ol = EvalOutline(outline_width, outline_height);
  CostFunc area = [&](const Coordinate& packing_result) {
    return (float) packing_result.width * (float) packing_result.height / (float) outline_width / (float) outline_height;
  };
  Evaluator<SeqPair<NodeShape<T>>, Coordinate> eval(decoder, {weight_wl, weight_ol, weight_area}, {wl, ol, area});

  PerturbFunc ps_op = PosSwap();
  PerturbFunc ns_op = NegSwap();
  PerturbFunc ds_op = DoubleSwap();
  PerturbFunc pi_op = PosInsert();
  PerturbFunc ni_op = NegInsert();
  PerturbFunc rs_op = Resize();
  PerturbFunc rotate_op = [](SeqPair<NodeShape<T>>& sp, std::mt19937& gen) {
    std::uniform_int_distribution<size_t> get_ridx(0, sp.size - 1);
    size_t id = get_ridx(gen);
    sp.properties[id].is_rotate = !sp.properties[id].is_rotate;
  };

  Perturb<SeqPair<NodeShape<T>>, void> perturb(seed, {0.2, 0.2, 0.2, 0.2, 0.2, 0.2}, {ps_op, ns_op, ds_op, pi_op, ni_op, rs_op});

  // intalize the norm cost and inital product
  eval.initalize(packing_result, sp, perturb, num_vertices * 15 / 10);

  SimulateAnneal solve{.seed = seed, .num_perturb = num_vertices * 2, .cool_rate = 0.98, .inital_temperature = 1000};
  auto sa_start = std::chrono::high_resolution_clock::now();
  sp = solve(sp, eval, perturb, [](auto&& x) { std::cout << x << std::endl; });
  auto sa_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> sa_elapsed = std::chrono::duration<double>(sa_end - sa_start);
  INFO("SAPlace time: ", sa_elapsed.count(), "s");

  // get location
  pack(sp, packing_result);

  // update sub_cluster's location
  for (size_t v_pos = 0; v_pos < initial_lx.size(); ++v_pos) {
    auto sub_obj = cluster.netlist().vertex_at(v_pos).property();
    if (ignore[v_pos]) {
      continue;
    }
    // update shape && locaton
    if (sub_obj->isBlock()) {  // only block has shape-curve, update shape
      auto sub_block = std::static_pointer_cast<Block, Object>(sub_obj);
      sub_block->get_shape_curve().set_width(sp.properties[v_pos].width);
      sub_block->get_shape_curve().set_height(sp.properties[v_pos].height);
    }
    sub_obj->set_min_corner(packing_result.x[v_pos], packing_result.y[v_pos]);
  }

  if (packing_result.width > outline_width || packing_result.height > outline_height) {
    return false;
  } else {
    return true;
  }
}

template <typename T>
std::pair<T, T> calMacroTilings(const std::vector<ShapeCurve<T>>& sub_shape_curves, T outline_width, T outline_height,
                                const std::string& name)
{
  /**
   * @brief calculate children-macro-cluster's possible tilings with minimal area
   *
   * @param cluster parent cluster
   * @param runs num sa_runs to generate different tilings
   * @param core_width Core-Width of Chip (Not parent cluster!)
   * @param core_height Core-Height of Chip (Not parent cluster!)
   * @return possible macro tilings (discrete shapes)
   */

  using DimFunc = FastPackSP<Prop, Coordinate>::DimFunc;
  using IgnoreFunc = FastPackSP<Prop, Coordinate>::IgnoreFunc;
  using Decoder = Evaluator<SeqPair<Prop>, Coordinate>::Decoder;
  using CostFunc = Evaluator<SeqPair<Prop>, Coordinate>::CostFunc;
  using PerturbFunc = Perturb<SeqPair<Prop>, void>::PerturbFunc;

  // initialize input data
  if (sub_shape_curves.empty()) {
    throw std::runtime_error("no shapes to place!");
  }
  // if (sub_shape_curves.size() == 1) {
  //   // maybe only one child macro-cluster, return it's possbile-discrete-shapes
  //   std::cout << "only one shapes here! " << std::endl;
  //   return sub_shape_curves[0].get_discrete_shapes();
  // }

  auto bench_begin = std::chrono::high_resolution_clock::now();
  size_t num_vertices = sub_shape_curves.size();
  std::vector<Prop> prop(num_vertices);
  std::vector<bool> ignore(num_vertices, false);
  std::vector<T> blk_widths(num_vertices);
  std::vector<T> blk_heights(num_vertices);
  std::vector<T> initial_lx(num_vertices, 0);
  std::vector<T> initial_ly(num_vertices, 0);
  for (size_t i = 0; i < sub_shape_curves.size(); ++i) {
    blk_widths[i] = sub_shape_curves[i].get_width();
    blk_heights[i] = sub_shape_curves[i].get_height();
  }
  T outline_lx = 0;
  T outline_ly = 0;
  Coordinate packing_result{.x = initial_lx, .y = initial_ly};

  auto bench_end = std::chrono::high_resolution_clock::now();
  INFO("cal macro tiling initialize time: ", std::chrono::duration<double>(bench_end - bench_begin).count(), "s");
  int seed = 0;
  std::mt19937 gen(seed);
  SeqPair<Prop> sp(prop, gen);

  DimFunc get_blk_width = [&](size_t i, const Prop& b) { return blk_widths[i]; };
  DimFunc get_blk_height = [&](size_t i, const Prop& b) { return blk_heights[i]; };
  IgnoreFunc is_ignore = [&](size_t i, const Prop&) { return false; };

  // packing function to get coordinate
  FastPackSP<Prop, Coordinate> pack(outline_lx, outline_ly, get_blk_width, get_blk_height, is_ignore);
  Decoder decoder = pack;

  // only use outline && area cost
  CostFunc ol = EvalOutline(outline_width, outline_height);
  CostFunc area = [&](const Coordinate& packing_result) {
    return (float) packing_result.width * (float) packing_result.height / (float) outline_width / (float) outline_height;
  };
  Evaluator<SeqPair<Prop>, Coordinate> eval(decoder, {1, 2}, {area, ol});

  PerturbFunc ps_op = PosSwap();
  PerturbFunc ns_op = NegSwap();
  PerturbFunc ds_op = DoubleSwap();
  PerturbFunc pi_op = PosInsert();
  PerturbFunc ni_op = NegInsert();

  Perturb<SeqPair<Prop>, void> perturb(seed, {0.2, 0.2, 0.2, 0.2, 0.2}, {ps_op, ns_op, ds_op, pi_op, ni_op});

  // intalize the norm cost and inital product
  eval.initalize(packing_result, sp, perturb, num_vertices * 15 / 10);

  INFO("start calculate macro tiling..., num_vertices = ", num_vertices);
  auto sa_start = std::chrono::high_resolution_clock::now();
  SimulateAnneal solve{.seed = seed, .num_perturb = num_vertices * 2, .cool_rate = 0.98, .inital_temperature = 1000};
  sp = solve(sp, eval, perturb, [](auto&& x) {});
  auto sa_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> sa_elapsed = std::chrono::duration<double>(sa_end - sa_start);

  // update solution
  pack(sp, packing_result);
  INFO("cal macro tiling SA time: ", sa_elapsed.count(), "s");
  // writePlacement<T>("/home/liuyuezuo/iEDA-master/build/output/" + name + ".txt", packing_result.x, packing_result.y, blk_widths,
  //                   blk_heights, outline_width, outline_height);
  return std::make_pair(packing_result.width, packing_result.height);
}

}  // namespace imp