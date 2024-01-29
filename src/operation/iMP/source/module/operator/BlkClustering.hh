#pragma once
#include <numeric>
#include <cstddef>
namespace imp {
class Block;
class HMetis;
struct BlkClustering
{
  void operator()(imp::Block& block);

  size_t l1_nparts{std::numeric_limits<size_t>::max()};
  size_t l2_nparts{std::numeric_limits<size_t>::max()};
};
}  // namespace imp
