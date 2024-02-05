#include "BlkClustering.hh"

#include "Block.hh"
#include "Hmetis.hh"
#include "HyperGraphAlgorithm.hh"
#include "Logger.hpp"
namespace imp {
void BlkClustering::operator()(Block& block)
{
  auto netlist = block.netlist();
  size_t nparts = block.level() == 1 ? l1_nparts : l2_nparts;
  if (netlist.vSize() <= nparts || block.level() > 2)
    return;

  auto&& [eptr, eind] = vectorize(netlist);
  HMetis partition{.seed = 0};
  auto parts = partition(block.get_name(), eptr, eind, nparts);
  int i = 0;

  auto sub_block = [&](const Netlist& graph, const std::vector<size_t>& sub_vertices) {
    auto&& [sub_netlist, cuts] = sub_graph(graph, sub_vertices);
    int64_t sum_area = std::accumulate(sub_netlist.vbegin(), sub_netlist.vend(), int64_t(0),
                                       [](auto&& a, auto&& b) { return a + (int64_t) geo::area(b.property()->boundingbox()); });
    int32_t w = std::sqrt(sum_area);
    int32_t h = w;
    auto new_block = std::make_shared<imp::Block>(block.get_name() + "_" + std::to_string(i++),
                                                  std::make_shared<imp::Netlist>(std::move(sub_netlist)), block.shared_from_this());
    new_block->set_shape_curve(imp::geo::make_box(0, 0, w, h));
    INFO(new_block->get_name(), " num_v: ", new_block->netlist().vSize(), " num_cuts: ", cuts.size(),
         " num_e: ", new_block->netlist().heSize());
    return new_block;
  };

  auto clusters = clustering(netlist, parts, sub_block);
  block.set_netlist(std::make_shared<Netlist>(std::move(clusters)));
}

void BlkClustering2::operator()(Block& block)
{
  auto netlist = block.netlist();
  size_t nparts = block.level() == 1 ? l1_nparts : l2_nparts;
  if (netlist.vSize() <= nparts || block.level() > level_num)
    return;

  auto&& [eptr, eind] = vectorize(netlist);
  HMetis partition{.seed = 0};
  auto parts = partition(block.get_name(), eptr, eind, nparts);

  // extract io-cell as single cluster at level 1
  size_t single_cluster_id = nparts;
  if (block.level() == 1) {
    size_t num_terminals = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
      auto vertex_prop = block.netlist().vertex_at(i).property();
      if (vertex_prop->isInstance()) {
        auto& inst = dynamic_cast<Instance&>(*vertex_prop);
        if (inst.get_cell_master().isIOCell()) {
          num_terminals++;
          parts[i] = single_cluster_id++;  // extract io as a single cluster;
        }
      }
    }
    INFO("num_terminals : ", num_terminals);
  }

  // extract macro as single cluster at last level
  if (block.level() == level_num) {
    // size_t single_macro_cluster_id = nparts;
    for (size_t i = 0; i < parts.size(); ++i) {
      auto vertex_prop = block.netlist().vertex_at(i).property();
      if (vertex_prop->isInstance()) {
        auto& inst = dynamic_cast<Instance&>(*vertex_prop);
        if (inst.get_cell_master().isMacro()) {
          parts[i] = single_cluster_id++;  // extract macro as a single cluster;
        }
      }
    }
  }

  int i = 0;

  auto sub_block = [&](const Netlist& graph, const std::vector<size_t>& sub_vertices) {
    auto&& [sub_netlist, cuts] = sub_graph(graph, sub_vertices);
    int64_t sum_area = std::accumulate(sub_netlist.vbegin(), sub_netlist.vend(), int64_t(0),
                                       [](auto&& a, auto&& b) { return a + (int64_t) geo::area(b.property()->boundingbox()); });
    int32_t w = std::sqrt(sum_area);
    int32_t h = w;
    auto new_block = std::make_shared<imp::Block>(block.get_name() + "_" + std::to_string(i++),
                                                  std::make_shared<imp::Netlist>(std::move(sub_netlist)), block.shared_from_this());
    // new_block->set_shape(imp::geo::make_box(0, 0, w, h));
    INFO(new_block->get_name(), " num_v: ", new_block->netlist().vSize(), " num_cuts: ", cuts.size(),
         " num_e: ", new_block->netlist().heSize());
    return new_block;
  };

  auto clusters = clustering(netlist, parts, sub_block);
  block.set_netlist(std::make_shared<Netlist>(std::move(clusters)));
}

}  // namespace imp
