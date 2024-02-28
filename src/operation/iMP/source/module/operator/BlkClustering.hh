#pragma once
#include <cstddef>
#include <numeric>
namespace imp {
class Block;
class HMetis;
struct BlkClustering
{
  void operator()(imp::Block& block);

  size_t l1_nparts{std::numeric_limits<size_t>::max()};
  size_t l2_nparts{std::numeric_limits<size_t>::max()};
};

struct BlkClustering2
{
  void operator()(imp::Block& block);

  size_t l1_nparts{std::numeric_limits<size_t>::max()};
  size_t l2_nparts{std::numeric_limits<size_t>::max()};
  size_t level_num = 2;
};

}  // namespace imp
