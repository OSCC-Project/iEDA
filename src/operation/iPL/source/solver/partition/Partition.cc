
#include "Partition.hh"

#include <time.h>

#include <iostream>

namespace ipl {
void Partition::MetisPartition(MetisParam* metis_param)
{
  clock_t start = clock();
  idx_t* _nvtxs = metis_param->_nvtxs;  // 1
  idx_t* _ncon = metis_param->_ncon;                     // 2
  idx_t* _xadj = metis_param->_xadj.data();              // 3
  idx_t* _adjncy = metis_param->_adjncy.data();          // 4
  idx_t* _nparts = metis_param->_nparts;                 // 8
  idx_t* _options = &(metis_param->_options[0]);         // 11
  idx_t* _objval = metis_param->_objval;                 // 12
  idx_t* _part = metis_param->_part.data();

  // call metis
  std::cout << "call metis" << std::endl;
  METIS_API(int)
  result = METIS_PartGraphRecursive(_nvtxs, _ncon, _xadj, _adjncy, NULL, NULL, NULL, _nparts, NULL, NULL, _options, _objval, _part);

  if (result == 1) {
    std::cout << "metis succeed" << std::endl;
  }

  std::cout << "the edge-cut: " << *_objval << std::endl;
  std::cout << "end metis" << std::endl;
  std::cout << "size of _part: " << metis_param->_part.size() << std::endl;
  std::cout << "partition time consume: " << double(clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
}
}  // namespace ipl