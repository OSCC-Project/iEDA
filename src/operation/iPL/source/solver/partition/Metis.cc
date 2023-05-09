#include "Metis.hh"

#include <time.h>

namespace ipl {
void Metis::partition(const std::vector<std::vector<int>> adjacent_edge_list)
{
  clock_t start = clock();
  _xadj.clear();
  _adjncy.clear();
  *_nvtxs = adjacent_edge_list.size();
  _xadj.emplace_back(0);
  _part.resize(adjacent_edge_list.size());

  for (std::vector<int> adjacent_edge : adjacent_edge_list) {
    for (int index : adjacent_edge) {
      _adjncy.emplace_back(index);
    }
    _xadj.emplace_back(_adjncy.size());
  }

  LOG_INFO << "num of edge: " << _adjncy.size();

  idx_t* xadj = _xadj.data();
  idx_t* adjncy = _adjncy.data();
  idx_t* options = &(_options[0]);
  idx_t* part = _part.data();

  // call metis
  LOG_INFO << "call metis";
  METIS_API(int)
  result = METIS_PartGraphRecursive(_nvtxs, _ncon, xadj, adjncy, NULL, NULL, NULL, _nparts, NULL, NULL, options, _objval, part);

  if (1 == result) {
    LOG_INFO << "metis succeed" ;
  }

  LOG_INFO << "the edge-cut: " << *_objval ;
  LOG_INFO << "size of _part: " << _part.size() ;
  LOG_INFO << "partition time consume: " << double(clock() - start) / CLOCKS_PER_SEC << "s" ;
}
}  // namespace ipl