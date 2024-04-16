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
#include "Net.hh"
#include "Object.hh"

namespace imp {

struct Prop
{
};

template <typename T>
struct NodeShape
{
  NodeShape() : width(0), height(0), _shape_curve(nullptr) {}
  NodeShape(T w, T h) : width(w), height(h), _shape_curve(nullptr) {}
  NodeShape(const NodeShape& other) = default;
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
      float inflate_rate = 1.0;
      // auto rand = distribution(generator) * 3;
      // if (rand >= 2.0) {
      //   inflate_rate = 1.2;
      // } else if (rand >= 1.0) {
      //   inflate_rate = 1.4;
      // } else {
      //   inflate_rate = 1.0;
      // }
      width = w * inflate_rate;
      height = h * inflate_rate;
      return true;
    }
    return false;
  }

  bool is_rotate = false;
  T width;
  T height;

  //  private:
  const ShapeCurve<T>* _shape_curve;
};

struct Coordinate
{
  int32_t width{0};
  int32_t height{0};
  std::vector<int32_t> x{};
  std::vector<int32_t> y{};
  std::vector<int32_t> dx{};
  std::vector<int32_t> dy{};
};

template <typename T>
struct SAPlace
{
  using Represent = SeqPair<NodeShape<T>>;
  using DimFunc = FastPackSP<NodeShape<T>, Coordinate>::DimFunc;
  using IgnoreFunc = FastPackSP<NodeShape<T>, Coordinate>::IgnoreFunc;
  using Decoder = Evaluator<Represent, Coordinate>::Decoder;
  using CostFunc = Evaluator<Represent, Coordinate>::CostFunc;
  using PerturbFunc = Perturb<Represent, void>::PerturbFunc;

  Represent operator()(Block& cluster);
  ~SAPlace();

  // settings
  int seed = 0;
  float weight_wl = 1.0;
  float weight_ol = 0.05;
  float weight_ob = 0.01;
  float weight_periphery = 0.01;
  float weight_blk = 0.01;
  float weight_io = 0.0;
  size_t max_iters = 1000;
  double cool_rate = 0.97;
  double init_temperature = 1000;
  double prob_ps_op = 0.2;
  double prob_ns_op = 0.2;
  double prob_ds_op = 0.2;
  double prob_pi_op = 0.2;
  double prob_ni_op = 0.2;
  double prob_rs_op = 0.2;
  int num_threads = 4;
  Represent init_represent;

  // data
  size_t num_vertices;
  size_t num_edges;
  std::vector<NodeShape<T>> blk_shapes;
  std::vector<bool> ignore;
  std::vector<T> initial_lx;
  std::vector<T> initial_ly;
  std::vector<size_t> blk_macro_nums;
  std::vector<size_t> macro_blk_indices;  // indices of blocks which contains macros
  std::vector<size_t> eptr;
  std::vector<size_t> eind;
  std::vector<float> net_weight;
  size_t total_macro_num;
  geo::point<int32_t> outline_min_corner;
  T outline_lx;
  T outline_ly;
  T outline_width;
  T outline_height;
  T outline_ux;
  T outline_uy;
  Coordinate packing_result;
  Represent* represent{nullptr};
  Perturb<Represent, void>* perturb{nullptr};
  Evaluator<Represent, Coordinate>* eval{nullptr};
  Represent& get_represent() { return *represent; }
  Perturb<Represent, void>& get_perturb_func() { return *perturb; }
  Evaluator<Represent, Coordinate>& get_evaluator() { return *eval; }
  Decoder decoder;

  void initPlaceData(Block& cluster);
  void initPerturbFunc();
  void initRepresent(Block& cluster);
  void initEvaluator(Block& cluster);
  Decoder createPackFunc();
  void updateNetlist(Block& netlist, const Represent& solution_represent);

  static float overlapArea(T lx1, T ly1, T ux1, T uy1, T lx2, T ly2, T ux2, T uy2)
  {
    T overlap_lx = std::max(lx1, lx2);
    T overlap_ly = std::max(ly1, ly2);
    T overlap_ux = std::min(ux1, ux2);
    T overlap_uy = std::min(uy1, uy2);
    if (overlap_lx > overlap_ux || overlap_ly > overlap_uy) {
      return 0;
    };
    T width = overlap_ux - overlap_lx;
    T height = overlap_uy - overlap_ly;
    return float(width) * height;
  }

  struct EvalIODense
  {
    double operator()(const Coordinate& packing_result);
    EvalIODense(Block& root_cluster, std::vector<size_t>& macro_ind, double clip_ratio, int num_threads = 4);

    int num_threads = 4;
    size_t grid_num;
    double grid_w;
    double grid_h;
    double outline_lx;
    double outline_ly;
    double outline_ux;
    double outline_uy;
    double outline_width;
    double outline_height;
    double clip_value;
    std::vector<std::vector<double>> dense_map;
    std::vector<size_t>& macro_indices;

    size_t get_x_grid(double x);
    size_t get_y_grid(double y);
    double dist(bool horizontal, double pin_x, double pin_y, double loc_x, double loc_y);
  };
};

template <typename T>
std::pair<T, T> calMacroTilings(const std::vector<ShapeCurve<T>>& sub_shape_curves, T outline_width, T outline_height,
                                const std::string& name);

template <typename T>
SAPlace<T>::~SAPlace()
{
  if (represent != nullptr)
    delete represent;
  if (perturb != nullptr)
    delete perturb;
  if (eval != nullptr)
    delete eval;
}

template <typename T>
SAPlace<T>::Represent SAPlace<T>::operator()(Block& cluster)
{
  if (cluster.netlist().vSize() == 0) {
    ERROR("Try to place cluster with 0 nodes...");
  }
  if (cluster.netlist().vSize() == 1) {  // single-node cluster, place it at the min_corner of parent-cluster
    auto sub_obj = cluster.netlist().vertex_at(0).property();
    // update shape && locaton
    sub_obj->set_min_corner(cluster.get_min_corner());
    return init_represent;
  }

  initPlaceData(cluster);
  initRepresent(cluster);
  initPerturbFunc();
  initEvaluator(cluster);

  auto sa_start = std::chrono::high_resolution_clock::now();
  SimulateAnneal solve{.seed = seed,
                       .max_iters = max_iters,
                       .num_perturb = num_vertices * 4,
                       .cool_rate = cool_rate,
                       .inital_temperature = init_temperature};
  Represent solution_represent = solve(get_represent(), get_evaluator(), get_perturb_func(), [](auto&& x) { std::cout << x << std::endl; });
  auto sa_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> sa_elapsed = std::chrono::duration<float>(sa_end - sa_start);
  INFO("SAPlace time: ", sa_elapsed.count(), "s");

  updateNetlist(cluster, solution_represent);
  return solution_represent;
}

template <typename T>
void SAPlace<T>::initPlaceData(Block& cluster)
{
  // initialize input data
  auto start = std::chrono::high_resolution_clock::now();
  num_vertices = cluster.netlist().vSize();
  num_edges = cluster.netlist().heSize();
  blk_shapes = std::vector<NodeShape<T>>(num_vertices);
  ignore = std::vector<bool>(num_vertices, false);
  initial_lx = std::vector<T>(num_vertices, 0);
  initial_ly = std::vector<T>(num_vertices, 0);
  blk_macro_nums = std::vector<size_t>(num_vertices, 0);
  macro_blk_indices.clear();
  total_macro_num = cluster.get_macro_num();

  // place-outline is based on parent cluster's shape & location,  assume parent cluster's shape & location has been decided
  outline_min_corner = cluster.get_min_corner();
  outline_lx = outline_min_corner.x();
  outline_ly = outline_min_corner.y();
  outline_width = cluster.get_shape_curve().get_width();
  outline_height = cluster.get_shape_curve().get_height();
  outline_ux = outline_lx + outline_width;
  outline_uy = outline_ly + outline_height;

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
      if (sub_block->is_macro_cluster()) {
        macro_blk_indices.push_back(v_pos);
        blk_macro_nums[v_pos] = sub_block->get_macro_num();
      }
    } else {
      auto sub_inst = std::static_pointer_cast<Instance, Object>(sub_obj);
      blk_shapes[v_pos] = NodeShape(sub_inst->get_width(), sub_inst->get_height());
      if (sub_inst->isFixed()) {
        ignore[v_pos] = true;
        continue;
      }
      if (sub_inst->get_cell_master().isMacro()) {
        macro_blk_indices.push_back(v_pos);
        blk_macro_nums[v_pos] = 1;
      }
    }
    auto min_corner = sub_obj->get_min_corner();
    initial_lx[v_pos] = min_corner.x();
    initial_ly[v_pos] = min_corner.y();
    packing_result.x = initial_lx;
    packing_result.y = initial_ly;
  }

  // vectorized net & netweight
  auto get_net_weight = [](std::shared_ptr<Net> net) -> float { return net->get_net_weight(); };
  auto&& [temp_eptr, temp_eind, temp_vweight, temp_heweight]
      = vectorize(cluster.netlist(), NoneWeight<std::shared_ptr<Object>>, get_net_weight);
  if (num_edges != temp_heweight.size()) {
    ERROR("net weight length error!");
  }
  eptr = temp_eptr;
  eind = temp_eind;
  net_weight = temp_heweight;

  auto end = std::chrono::high_resolution_clock::now();
  INFO("SAPlace data initialize time: ", std::chrono::duration<float>(end - start).count(), "s");
}

template <typename T>
void SAPlace<T>::initRepresent(Block& cluster)
{
  std::mt19937 gen(seed);
  if (represent != nullptr) {
    delete represent;
  }

  if (init_represent.size != 0) {
    if (init_represent.size != cluster.netlist().vSize()) {
      ERROR("Error, inital represent doesn't match netlist vertex num!");
    }
    represent = new Represent(init_represent);
    INFO("Using given initial represent solution");
  } else {
    represent = new Represent(blk_shapes, gen);
    INFO("Using random initial represent solution");
  }
}

template <typename T>
void SAPlace<T>::initPerturbFunc()
{
  PerturbFunc ps_op = PosSwap();
  PerturbFunc ns_op = NegSwap();
  PerturbFunc ds_op = DoubleSwap();
  PerturbFunc pi_op = PosInsert();
  PerturbFunc ni_op = NegInsert();
  PerturbFunc rs_op = Resize();
  if (perturb != nullptr) {
    delete perturb;
  }
  perturb = new Perturb(seed, {prob_ps_op, prob_ns_op, prob_ds_op, prob_pi_op, prob_ni_op, prob_rs_op},
                        {ps_op, ns_op, ds_op, pi_op, ni_op, rs_op});
}

template <typename T>
SAPlace<T>::Decoder SAPlace<T>::createPackFunc()
{
  DimFunc get_blk_width = [](size_t i, const NodeShape<T>& b) { return b.is_rotate ? b.height : b.width; };
  DimFunc get_blk_height = [](size_t i, const NodeShape<T>& b) { return b.is_rotate ? b.width : b.height; };
  IgnoreFunc is_ignore = [this](size_t i, const NodeShape<T>&) { return this->ignore[i]; };

  // packing function to get coordinate
  return FastPackSPWithShape<NodeShape<T>, Coordinate>(outline_lx, outline_ly, get_blk_width, get_blk_height, is_ignore);
}

template <typename T>
void SAPlace<T>::initEvaluator(Block& cluster)
{
  decoder = createPackFunc();
  // Hpwl<T> hpwl(eptr, eind, {}, {}, {}, 1);
  // CostFunc wl = EvalWirelength(outline_width, outline_height, hpwl, net_weight);
  Hpwl2<T> hpwl(eptr, eind, net_weight, num_threads);
  CostFunc wl = EvalWirelength2(outline_width, outline_height, hpwl, net_weight);  // EvalWirelength2 uses with Hpwl2
  CostFunc ol = EvalOutline(outline_width, outline_height);
  CostFunc ob = EvalOutOfBound(outline_width, outline_height, outline_lx, outline_ly);
  CostFunc periphery = [this](const Coordinate& packing_result) {
    // T min_dist = 0;
    const auto& outline_width = this->outline_width;
    const auto& outline_height = this->outline_height;
    const auto& outline_lx = this->outline_lx;
    const auto& outline_ly = this->outline_ly;
    const auto& total_macro_num = this->outline_ly;
    const auto& blk_macro_nums = this->blk_macro_nums;
    const auto& macro_blk_indices = this->macro_blk_indices;

    double periphery_cost = 0;
    for (size_t id : macro_blk_indices) {
      T min_dist = 0;
      T dist_left = packing_result.x[id] - outline_lx;
      T dist_bottom = packing_result.y[id] - outline_ly;
      min_dist = std::min(dist_left, dist_bottom);
      T dist_top = std::abs(outline_width - dist_left - packing_result.dx[id]);
      min_dist = std::min(min_dist, dist_top);
      T dist_right = std::abs(outline_height - dist_bottom - packing_result.dy[id]);
      min_dist = std::min(min_dist, dist_right);
      periphery_cost += std::pow(double(min_dist), 2) * blk_macro_nums[id];
    }
    periphery_cost /= total_macro_num;
    return periphery_cost;
  };

  auto blkOverlap = [&cluster, this](const Coordinate& packing_result) -> double {
    double blk_overlap = 0;
    const auto& blockages = cluster.get_blockages();
    const auto& macro_blk_indices = this->macro_blk_indices;
    for (size_t id : macro_blk_indices) {
      for (auto&& blockage : blockages) {
        blk_overlap += overlapArea(blockage.min_corner().x(), blockage.min_corner().y(), blockage.max_corner().x(),
                                   blockage.max_corner().y(), packing_result.x[id], packing_result.y[id],
                                   packing_result.x[id] + packing_result.dx[id], packing_result.y[id] + packing_result.dy[id]);
      }
    }
    blk_overlap = blk_overlap / cluster.get_shape_curve().get_width() / cluster.get_shape_curve().get_height();
    return blk_overlap;
  };

  EvalIODense io_dense(cluster, macro_blk_indices, 0.03, num_threads);

  if (eval != nullptr) {
    delete eval;
  }

  eval = new Evaluator<SeqPair<NodeShape<T>>, Coordinate>(decoder, {weight_wl, weight_ol, weight_ob, weight_periphery, weight_blk, 0.05},
                                                          {wl, ol, ob, periphery, blkOverlap, io_dense});

  // intalize the norm cost and inital product
  eval->initalize(packing_result, get_represent(), get_perturb_func(), num_vertices * 15);
}

template <typename T>
void SAPlace<T>::updateNetlist(Block& cluster, const Represent& solution_represent)
{
  // get location
  auto pack = createPackFunc();
  pack(solution_represent, packing_result);

  // update sub_cluster's location
  for (size_t v_pos = 0; v_pos < initial_lx.size(); ++v_pos) {
    auto sub_obj = cluster.netlist().vertex_at(v_pos).property();
    if (ignore[v_pos]) {
      continue;
    }
    // update shape && locaton
    if (sub_obj->isBlock()) {  // only block has shape-curve, update shape
      auto sub_block = std::static_pointer_cast<Block, Object>(sub_obj);
      sub_block->get_shape_curve().set_width(solution_represent.properties[v_pos].width);
      sub_block->get_shape_curve().set_height(solution_represent.properties[v_pos].height);
    }
    sub_obj->set_min_corner(packing_result.x[v_pos], packing_result.y[v_pos]);
  }
}

template <typename T>
double SAPlace<T>::EvalIODense::operator()(const Coordinate& packing_result)
{
  double lx, ly, ux, uy;
  double cost = 0;
  double grid_area = grid_w * grid_h;
  int chunk_size = std::max(macro_indices.size() / num_threads, size_t(1));
  // #pragma omp parallel for num_threads(num_threads) schedule(static, chunk_size) reduction(+ : cost)
  for (size_t id : macro_indices) {
    double macro_cost = 0;
    lx = packing_result.x[id];
    ly = packing_result.y[id];
    ux = lx + packing_result.dx[id];
    uy = ly + packing_result.dy[id];

    size_t grid_start_x = get_x_grid(lx);
    size_t grid_end_x = get_x_grid(ux);
    size_t grid_start_y = get_y_grid(ly);
    size_t grid_end_y = get_y_grid(uy);

    for (size_t i = grid_start_x; i <= grid_end_x; ++i) {
      for (size_t j = grid_start_y; j <= grid_end_y; ++j) {
        macro_cost += dense_map[i][j];
      }
    }
    if (ux > outline_ux || uy > outline_uy) {
      double d1 = std::min(std::max(ux - outline_ux, 0.0), double(packing_result.dx[id]));
      double d2 = std::min(std::max(uy - outline_uy, 0.0), double(packing_result.dy[id]));
      double out_area = d1 * packing_result.dy[id] + d2 * packing_result.dx[id] - d1 * d2;
      macro_cost += out_area / grid_area * clip_value;
    }
    cost += macro_cost;
  }
  return cost;
}

template <typename T>
SAPlace<T>::EvalIODense::EvalIODense(Block& root_cluster, std::vector<size_t>& macro_ind, double clip_ratio, int num_threads)
    : macro_indices(macro_ind), num_threads(num_threads)
{
  auto outline_min_corner = root_cluster.get_min_corner();
  outline_lx = outline_min_corner.x();
  outline_ly = outline_min_corner.y();
  outline_width = root_cluster.get_shape_curve().get_width();
  outline_height = root_cluster.get_shape_curve().get_height();
  outline_ux = outline_lx + outline_width;
  outline_uy = outline_ly + outline_height;

  std::vector<std::shared_ptr<Object>> left_pins;
  std::vector<std::shared_ptr<Object>> right_pins;
  std::vector<std::shared_ptr<Object>> top_pins;
  std::vector<std::shared_ptr<Object>> bottom_pins;

  for (auto&& i : root_cluster.netlist().vRange()) {
    auto obj = i.property();
    auto blk = std::static_pointer_cast<Block, Object>(obj);
    if (blk->is_io_cluster()) {
      auto pin_x = blk->get_min_corner().x();
      auto pin_y = blk->get_min_corner().y();
      if (pin_x <= outline_lx) {
        left_pins.push_back(obj);
      } else if (pin_x >= outline_ux) {
        right_pins.push_back(obj);
      } else if (pin_y <= outline_ly) {
        bottom_pins.push_back(obj);
      } else if (pin_y >= outline_uy) {
        top_pins.push_back(obj);
      } else {
        ERROR("Error, io-pin in core!");
      }
    }
  }
  grid_num = 128;
  grid_w = double(outline_width) / grid_num;
  grid_h = double(outline_height) / grid_num;
  dense_map = std::vector<std::vector<double>>(grid_num, std::vector<double>(grid_num, 0));

  auto pin_num = left_pins.size() + right_pins.size() + top_pins.size() + bottom_pins.size();
  std::vector<double> sorted_dense;
  sorted_dense.reserve(grid_num * grid_num);
  for (size_t i = 0; i < grid_num; ++i) {
    for (size_t j = 0; j < grid_num; ++j) {
      double scale = std::max(outline_height, outline_width);
      double x = outline_lx + grid_w / 2 + grid_w * i;
      double y = outline_ly + grid_h / 2 + grid_h * j;

      x /= scale;
      y /= scale;

      for (auto pin : left_pins) {
        double pin_x = pin->get_min_corner().x();
        double pin_y = pin->get_min_corner().y();
        pin_x /= scale;
        pin_y /= scale;
        // dense_map[j][i] += pin_dense(true, x, y, pin_x, pin_y);
        dense_map[i][j] += dist(true, x, y, pin_x, pin_y) / pin_num;
      }
      for (auto pin : right_pins) {
        double pin_x = pin->get_min_corner().x();
        double pin_y = pin->get_min_corner().y();
        pin_x /= scale;
        pin_y /= scale;
        // dense_map[j][i] += pin_dense(true, x, y, pin_x, pin_y);
        dense_map[i][j] += dist(true, x, y, pin_x, pin_y) / pin_num;
      }
      for (auto pin : bottom_pins) {
        double pin_x = pin->get_min_corner().x();
        double pin_y = pin->get_min_corner().y();
        pin_x /= scale;
        pin_y /= scale;
        // dense_map[j][i] += pin_dense(false, x, y, pin_x, pin_y);
        dense_map[i][j] += dist(false, x, y, pin_x, pin_y) / pin_num;
      }
      for (auto pin : top_pins) {
        double pin_x = pin->get_min_corner().x();
        double pin_y = pin->get_min_corner().y();
        pin_x /= scale;
        pin_y /= scale;
        // dense_map[j][i] += pin_dense(false, x, y, pin_x, pin_y);
        dense_map[i][j] += dist(false, x, y, pin_x, pin_y) / pin_num;
      }
      sorted_dense.push_back(dense_map[i][j]);
    }
  }
  // sort descending
  std::sort(sorted_dense.begin(), sorted_dense.end(), [](double x1, double x2) -> bool { return x1 > x2; });
  // double clip_ratio = 0.05;
  clip_value = sorted_dense[size_t(sorted_dense.size() * clip_ratio)];
  // clip large densities
  for (size_t i = 0; i < dense_map.size(); ++i) {
    for (size_t j = 0; j < dense_map[0].size(); ++j) {
      dense_map[i][j] = std::min(dense_map[i][j], clip_value);
    }
  }
}

template <typename T>
size_t SAPlace<T>::EvalIODense::get_x_grid(double x)
{
  x = std::min(outline_ux, x);
  size_t x_grid = std::min(size_t((x - outline_lx) / grid_w), grid_num - 1);
  assert(x_grid <= grid_num - 1);
  return x_grid;
}

template <typename T>
size_t SAPlace<T>::EvalIODense::get_y_grid(double y)
{
  y = std::min(outline_uy, y);
  size_t y_grid = std::min(size_t((y - outline_ly) / grid_h), grid_num - 1);
  assert(y_grid <= grid_num - 1);
  return y_grid;
}

template <typename T>
double SAPlace<T>::EvalIODense::dist(bool horizontal, double pin_x, double pin_y, double loc_x, double loc_y)
{
  double small = 1;
  double large = 2.0;
  double dist;
  if (abs(pin_x - loc_x) > 2 || abs(pin_y - loc_y) > 2) {
    std::cout << ">1" << std::endl;
  }
  if (horizontal) {
    dist = pow(pow(abs(pin_x - loc_x), large) + pow(abs(pin_y - loc_y), small), 0.5);
  } else {
    dist = pow(pow(abs(pin_x - loc_x), small) + pow(abs(pin_y - loc_y), large), 0.5);
  }
  dist = 0.01 / std::max(dist, 0.01);
  return pow(dist, 1.5);
}

template <typename T>
std::pair<T, T> calMacroTilings(const std::vector<ShapeCurve<T>>& sub_shape_curves, T outline_width, T outline_height,
                                const std::string& name)
{
  /**
   * @brief calculate children-macro-cluster's possible tilings with minimal area
   *
   * @param cluster parent cluste
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
    ERROR("no shapes to place!");
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
  INFO("cal macro tiling initialize time: ", std::chrono::duration<float>(bench_end - bench_begin).count(), "s");
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
  std::chrono::duration<float> sa_elapsed = std::chrono::duration<float>(sa_end - sa_start);

  // update solution
  pack(sp, packing_result);
  INFO("cal macro tiling SA time: ", sa_elapsed.count(), "s");
  // writePlacement<T>("/home/liuyuezuo/iEDA-master/build/output/" + name + ".txt", packing_result.x, packing_result.y, blk_widths,
  //                   blk_heights, outline_width, outline_height);
  return std::make_pair(packing_result.width, packing_result.height);
}

}  // namespace imp