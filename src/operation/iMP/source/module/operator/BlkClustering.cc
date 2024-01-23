#include "BlkClustering.hh"

#include "Block.hh"
#include "Hmetis.hh"
#include "HyperGraphAlgorithm.hh"
#include "Logger.hpp"
namespace imp {
void BlkClustering::operator()(imp::Block& block)
{
  auto netlist = block.netlist();
  size_t nparts = block.level() == 1 ? l1_nparts : l2_nparts;
  if (netlist.vSize() <= nparts || block.level() > 2)
    return;

  auto&& [eptr, eind] = imp::vectorize(netlist);
  imp::HMetis partition{0};
  auto parts = partition(eptr, eind, nparts);
  int i = 0;

  auto sub_block = [&](const imp::Netlist& graph, const std::vector<size_t>& sub_vertices) {
    auto&& [sub_netlist, cuts] = imp::sub_graph(graph, sub_vertices);
    int64_t sum_area = std::accumulate(sub_netlist.vbegin(), sub_netlist.vend(), int64_t(0),
                                       [](auto&& a, auto&& b) { return a + (int64_t) imp::geo::area(b.property()->boundingbox()); });
    int32_t w = std::sqrt(sum_area);
    int32_t h = w;
    auto new_block = std::make_shared<imp::Block>(block.get_name() + "/" + std::to_string(i++),
                                                  std::make_shared<imp::Netlist>(std::move(sub_netlist)), block.shared_from_this());
    // new_block->set_shape(imp::geo::make_box(0, 0, w, h));
    INFO(new_block->get_name(), " num_v: ", new_block->netlist().vSize(), " num_cuts: ", cuts.size(),
         " num_e: ", new_block->netlist().heSize());
    return new_block;
  };

  auto clusters = imp::clustering(netlist, parts, sub_block);
  block.set_netlist(std::make_shared<imp::Netlist>(std::move(clusters)));
}

}  // namespace imp
