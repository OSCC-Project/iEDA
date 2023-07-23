// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#include "Metis.hh"

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
    LOG_INFO << "metis succeed";
  }

  LOG_INFO << "the edge-cut: " << *_objval;
  LOG_INFO << "size of _part: " << _part.size();
  LOG_INFO << "partition time consume: " << double(clock() - start) / CLOCKS_PER_SEC << "s";
}

std::vector<int> Metis::get_result()
{
  std::vector<int> result;
  for (idx_t part_index : _part) {
    result.emplace_back(int(part_index));
  }
  return result;
}

}  // namespace ipl