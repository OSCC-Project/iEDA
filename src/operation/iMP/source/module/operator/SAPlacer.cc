#include "SAPlacer.hh"

namespace imp {

template <typename T>
SeqPair<NodeShape<T>> SAPlace<T>::operator()(Block& cluster)
{
  if (cluster.netlist().vSize() == 0) {
    throw std::runtime_error("try to place cluster with 0 nodes...");
  }
  if (cluster.netlist().vSize() == 1) {  // single-node cluster, place it at the min_corner of parent-cluster
    auto sub_obj = cluster.netlist().vertex_at(0).property();
    // update shape && locaton
    sub_obj->set_min_corner(cluster.get_min_corner());
    // return true;
    return SeqPair<NodeShape<T>>();
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
  std::vector<T> initial_lx(num_vertices, 0);
  std::vector<T> initial_ly(num_vertices, 0);
  std::vector<size_t> blk_macro_nums(num_vertices, 0);
  std::vector<size_t> macro_blk_indices;  // indices of blocks which contains macros
  size_t total_macro_num = cluster.get_macro_num();

  // place-outline is based on parent cluster's shape & location,  assume parent cluster's shape & location has been decided
  auto outline_min_corner = cluster.get_min_corner();
  T outline_lx = outline_min_corner.x();
  T outline_ly = outline_min_corner.y();
  T outline_width = cluster.get_shape_curve().get_width();
  T outline_height = cluster.get_shape_curve().get_height();
  T outline_ux = outline_lx + outline_width;
  T outline_uy = outline_ly + outline_height;

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
  }

  auto get_net_weight = [](std::shared_ptr<Net> net) -> float { return net->get_net_weight(); };

  auto&& [eptr, eind, vweight, heweight] = vectorize(cluster.netlist(), NoneWeight<std::shared_ptr<Object>>, get_net_weight);

  std::vector<float> net_weight(num_edges, 1);
  if (net_weight.size() != heweight.size()) {
    throw std::runtime_error("net weight length error!");
  }

  std::cout << std::endl;

  Coordinate packing_result{.x = initial_lx, .y = initial_ly};

  auto end = std::chrono::high_resolution_clock::now();
  INFO("SAPlace data initialize time: ", std::chrono::duration<float>(end - start).count(), "s");
  std::mt19937 gen(_seed);
  SeqPair<NodeShape<T>> sp;
  if (_init_sp.size != 0) {
    if (_init_sp.size != cluster.netlist().vSize()) {
      throw std::runtime_error("Error, inital sp doesn't match netlist vertex num!");
    }
    sp = _init_sp;
    INFO("using given initial sp solution");
  } else {
    sp = SeqPair<NodeShape<T>>(blk_shapes, gen);
    INFO("using random initial sp solution");
  }
  // SeqPair<NodeShape<T>> sp(blk_shapes, gen);

  DimFunc get_blk_width = [&](size_t i, const NodeShape<T>& b) { return b.is_rotate ? b.height : b.width; };
  DimFunc get_blk_height = [&](size_t i, const NodeShape<T>& b) { return b.is_rotate ? b.width : b.height; };
  IgnoreFunc is_ignore = [&](size_t i, const NodeShape<T>&) { return ignore[i]; };

  // packing function to get coordinate
  FastPackSPWithShape<NodeShape<T>, Coordinate> pack(outline_lx, outline_ly, get_blk_width, get_blk_height, is_ignore);
  Decoder decoder = pack;

  // Hpwl<T> hpwl(eptr, eind, {}, {}, {}, 1);
  // CostFunc wl = EvalWirelength(outline_width, outline_height, hpwl, net_weight);
  Hpwl2<T> hpwl(eptr, eind, net_weight, num_threads);
  CostFunc wl = EvalWirelength2(outline_width, outline_height, hpwl, net_weight);  // EvalWirelength2 uses with Hpwl2
  CostFunc ol = EvalOutline(outline_width, outline_height);
  CostFunc ob = EvalOutOfBound(outline_width, outline_height, outline_lx, outline_ly);
  CostFunc periphery = [outline_width, outline_height, outline_lx, outline_ly, total_macro_num, &blk_macro_nums,
                        &macro_blk_indices](const Coordinate& packing_result) {
    // T min_dist = 0;
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

  auto blockages = cluster.get_blockages();

  auto blkOverlap = [blockages, &macro_blk_indices, &cluster](const Coordinate& packing_result) -> double {
    double blk_overlap = 0;
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

  Evaluator<SeqPair<NodeShape<T>>, Coordinate> eval(decoder, {_weight_wl, _weight_ol, _weight_ob, _weight_periphery, _weight_blk, 0.05},
                                                    {wl, ol, ob, periphery, blkOverlap, io_dense});

  PerturbFunc ps_op = PosSwap();
  PerturbFunc ns_op = NegSwap();
  PerturbFunc ds_op = DoubleSwap();
  PerturbFunc pi_op = PosInsert();
  PerturbFunc ni_op = NegInsert();
  PerturbFunc rs_op = Resize();

  Perturb<SeqPair<NodeShape<T>>, void> perturb(_seed, {_prob_ps_op, _prob_ns_op, _prob_ds_op, _prob_pi_op, _prob_ni_op, _prob_rs_op},
                                               {ps_op, ns_op, ds_op, pi_op, ni_op, rs_op});

  // intalize the norm cost and inital product
  eval.initalize(packing_result, sp, perturb, num_vertices * 15);
  SeqPair<NodeShape<T>> solution_sp;
  auto sa_start = std::chrono::high_resolution_clock::now();

  SimulateAnneal solve{.seed = _seed,
                       .max_iters = _max_iters,
                       .num_perturb = num_vertices * 4,
                       .cool_rate = _cool_rate,
                       .inital_temperature = _init_temperature};
  solution_sp = solve(sp, eval, perturb, [](auto&& x) { std::cout << x << std::endl; });

  auto sa_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> sa_elapsed = std::chrono::duration<float>(sa_end - sa_start);
  INFO("SAPlace time: ", sa_elapsed.count(), "s");

  // get location
  pack(solution_sp, packing_result);

  // update sub_cluster's location
  for (size_t v_pos = 0; v_pos < initial_lx.size(); ++v_pos) {
    auto sub_obj = cluster.netlist().vertex_at(v_pos).property();
    if (ignore[v_pos]) {
      continue;
    }
    // update shape && locaton
    if (sub_obj->isBlock()) {  // only block has shape-curve, update shape
      auto sub_block = std::static_pointer_cast<Block, Object>(sub_obj);
      sub_block->get_shape_curve().set_width(solution_sp.properties[v_pos].width);
      sub_block->get_shape_curve().set_height(solution_sp.properties[v_pos].height);
    }
    sub_obj->set_min_corner(packing_result.x[v_pos], packing_result.y[v_pos]);
  }

  // if (packing_result.width > outline_width || packing_result.height > outline_height) {
  //   return false;
  // } else {
  //   return true;
  // }

  return solution_sp;
}

}  // namespace imp