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
#include "Hmetis.hh"

namespace ipl {

void Hmetis::partition(int vertex_num, const std::vector<std::vector<int>>& hyper_edge_list)
{
  // input
  std::string hgraph_file_name = _output_path + "/input.txt";
  std::ofstream hgraph_file(hgraph_file_name);
  hgraph_file << hyper_edge_list.size() << " " << vertex_num << std::endl;
  for (const std::vector<int>& hyper_edge : hyper_edge_list) {
    for (size_t i = 0; i < hyper_edge.size() - 1; ++i) {
      hgraph_file << hyper_edge[i] + 1 << " ";
    }
    hgraph_file << hyper_edge[hyper_edge.size() - 1] + 1 << std::endl;
  }
  hgraph_file.close();

  // call hmetis
  LOG_INFO << "hgraph_file_name: " << hgraph_file_name;
  std::string hmetis_command = _hmetis_path + " " + hgraph_file_name + " " + std::to_string(_nparts);
  hmetis_command += " -ptype=" + _ptype;
  hmetis_command += " -ctype=" + _ctype;
  hmetis_command += " -rtype=" + _rtype;
  hmetis_command += " -otype=" + _otype;
  hmetis_command += " -ufactor=" + std::to_string(_ufactor);
  hmetis_command += " -nruns=" + std::to_string(_nruns);
  hmetis_command += " -dbglvl=" + std::to_string(_dbglvl);
  hmetis_command += " -seed=0" + std::to_string(_seed);
  if (_reconst) {
    hmetis_command += " -reconst";
  }
  int status = system(hmetis_command.c_str());
  if (-1 != status && WIFEXITED(status) && 0 == WEXITSTATUS(status) ) {
    LOG_INFO << "hmetis partition succeed..";
  } else {
    LOG_ERROR << "hmetis partition fail, system return " << status;
  }
  

  // read result
  std::string solution_file = hgraph_file_name + ".part." + std::to_string(_nparts);
  std::ifstream result_file(solution_file);
  int cluster_idx;

  _partition_result.clear();
  while (result_file >> cluster_idx) {
    _partition_result.push_back(cluster_idx);
  }
}
}  // namespace ipl