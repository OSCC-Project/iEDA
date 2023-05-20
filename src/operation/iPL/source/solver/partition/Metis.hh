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
#pragma once
#include <time.h>

#include <vector>

#include "metis.h"
#include "module/logger/Log.hh"

namespace ipl {
class Metis
{
 public:
  Metis()
  {
    _options[METIS_OPTION_PTYPE] = METIS_PTYPE_RB;       // METIS_PTYPE_RB or METIS_PTYPE_KWAY
    _options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;  // METIS_OBJTYPE_CUT or METIS_OBJTYPE_VOL
    _options[METIS_OPTION_CTYPE] = METIS_CTYPE_RM;       // METIS_CTYPE_RM or METIS_CTYPE_SHEM
    _options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_GROW;   //
    _options[METIS_OPTION_RTYPE] = METIS_RTYPE_FM;       //
    _options[METIS_OPTION_DBGLVL] = 0;                   //
    _options[METIS_OPTION_NITER] = 10;                   //
    _options[METIS_OPTION_NCUTS] = 1;                    //
    _options[METIS_OPTION_SEED] = 0;                     //
    _options[METIS_OPTION_NO2HOP] = 0;                   //
    _options[METIS_OPTION_MINCONN] = 0;                  //
    _options[METIS_OPTION_CONTIG] = 0;                   //
    _options[METIS_OPTION_COMPRESS] = 0;                 //
    _options[METIS_OPTION_CCORDER] = 0;                  //
    _options[METIS_OPTION_PFACTOR] = 0;                  //
    _options[METIS_OPTION_NSEPS] = 1;                    //
    _options[METIS_OPTION_UFACTOR] = 400;                // importance
    _options[METIS_OPTION_NUMBERING] = 0;                //
  }

  void set_ncon(int ncon) { *_ncon = ncon; }
  void set_nparts(int nparts) { *_nparts = nparts; }
  void set_ufactor(int ufactor) { _options[METIS_OPTION_UFACTOR] = ufactor; }

  void partition(const std::vector<std::vector<int>> adjacent_edge_list);  // Index of adjacent node
  std::vector<int> get_result();

 private:
  // data
  idx_t* _nvtxs = new idx_t(1);       // number of node
  idx_t* _ncon = new idx_t(5);        // Maximum difference node for each part
  std::vector<idx_t> _xadj;           // edge pointers
  std::vector<idx_t> _adjncy;         // edge index
  idx_t* _nparts = new idx_t(2);      // number of part
  idx_t _options[METIS_NOPTIONS];     // metis options
  idx_t* _vertices_weight = nullptr;  // node weights
  idx_t* _edges_weight = nullptr;     // net weights
  idx_t* _objval = new idx_t(0);      // edge-cut
  std::vector<idx_t> _part;           // result of metis
  idx_t* _net_weight = nullptr;       // net weights
};

}  // namespace ipl
