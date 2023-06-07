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
#include <fstream>
#include <string>
#include <vector>

#include "module/logger/Log.hh"

namespace ipl {

class Hmetis
{
 public:
  void set_hmetis_path(std::string path) { _hmetis_path = path; }
  void set_output_path(std::string path) { _output_path = path; }
  void set_nparts(int num_parts) { _nparts = num_parts; }
  void set_ptype(std::string type) { _ptype = type; }
  void set_ctype(std::string type) { _ctype = type; }
  void set_rtype(std::string type) { _rtype = type; }
  void set_otype(std::string type) { _otype = type; }
  void set_ufactor(float factor) { _ufactor = factor; }
  void set_nruns(int num_runs) { _nruns = num_runs; }
  void set_nvcycles(int cycles) { _nvcycles = cycles; }
  void set_cmaxnet(int max_net) { _cmaxnet = max_net; }
  void set_reconst(bool reconst) { _reconst = reconst; }
  void set_kwayrefine(bool refine) { _kwayrefine = refine; }
  void set_fixed(std::string fixed) { _fixed = fixed; }
  void set_seed(int seed) { _seed = seed; }
  void set_dbglvl(int dbglvl) { _dbglvl = dbglvl; }

  void partition(int vertex_num, const std::vector<std::vector<int>>& hyper_edge_list); // Index of each hyperedge node
  std::vector<int> get_result() { return _partition_result; }

 private:
  std::string _hmetis_path = "../src/third_party/hmetis/hmetis2.0pre1";
  std::string _output_path = "./result/pl";
  int _nparts = 2;

  // hmetis option
  std::string _ptype = "rb";
  std::string _ctype = "gfc1";
  std::string _rtype = "moderate";
  std::string _otype = "cut";

  float _ufactor = 2.0;
  int _nruns = 1;
  int _nvcycles = 1;
  int _cmaxnet = 50;
  int _rmaxnet = 50;
  bool _reconst = false;
  bool _kwayrefine = false;
  std::string _fixed = "rb";
  int _seed = 0;
  int _dbglvl = 0;

  // result
  std::vector<int> _partition_result;
};

}  // namespace ipl

//  Optional parameters
//   -ptype=string
//      Specifies the scheme to be used for computing the k-way partitioning.
//      The possible values are:
//         rb       - Recursive bisection [default]
//         kway     - Direct k-way partitioning.

//   -ctype=string                 (rb, kway)
//      Specifies the scheme to be used for coarsening.
//      The possible values are:
//         fc1        - First-choice scheme
//         gfc1       - Greedy first-choice scheme [default]
//         fc2        - Alternate implementation of the fc scheme
//         gfc2       - Alternate implementation of the gfc scheme
//         h1         - Alternates between fc1 and gfc1 [default if nruns<20]
//         h2         - Alternates between fc2 and gfc2
//         h12        - Alternates between fc1, gfc1, fc2, gfc2 [default otherwise]
//         edge1      - Edge-based coarsening
//         gedge1     - Greedy edge-based coarsening
//         edge2      - Alternate implementation of the edge scheme
//         gedge2     - Alternate implementation of the gedge scheme

//   -rtype=string                 (rb, kway)
//      Specifies the scheme to be used for refinement.
//      The possible values and the partitioning types where they apply are:
//         fast        - Fast FM-based refinement (rb) [default]
//         moderate    - Moderate FM-based refinement (rb)
//         slow        - Slow FM-based refinement (rb)
//         krandom     - Random k-way refinement (kway) [default for ptype=kway]
//         kpfast      - Pairwise k-way FAST refinement (kway)
//         kpmoderate  - Pairwise k-way MODERATE refinement (kway)
//         kpslow      - Pairwise k-way SLOW refinement (kway)

//   -otype=string                 (kway)
//      Specifies the objective function to use for k-way partitioning.
//      The possible values are:
//         cut      - Minimize the cut [default]
//         soed     - Minimize the sum-of-external-degrees

//   -ufactor=float                (rb, kway)
//      Specifies the unbalance factor. The meaning of this parameters depends
//      on the partitioning scheme (ptype):
//        For ptype=rb
//          Specifies the maximum difference between each successive bisection.
//          For instance, a value of 5 leads to a 45-55 split at each bisection.
//        For ptype=kway
//          Specifies the maximum load imbalance among all the partitions. For
//          instance, a value of 5 produces partitions in which the weight of
//          the largest partition over the average weight is bounded by 5%.

//   -nruns=int                    (rb, kway)
//      Specifies the number of different bisections to be computed
//      at each level. The final bisection corresponds to the one that
//      has the smallest cut

//   -nvcycles=int                 (rb, kway)
//      Specifies the number of solutions to be further refined using
//      V-cycle refinement. If the supplied number is k, then the best
//      k bisections are further refined using V-cycling. This number
//      should be less or equal to that specified for -nruns. The default
//      value is one.

//   -cmaxnet=int                  (rb, kway)
//      Specifies the size of the largest net to be considered during
//      coarsening. Any nets larger than that are ignored when determing
//      which cells to merge together. Default value for that is 50.

//   -rmaxnet=int                  (rb, kway)
//      Specifies the size of the largest net to be considered during
//      refinement. Any nets larger than that are ignored during the
//      multilevel partitioning phase, and are dealt once the partitioning
//      has been computed. Default value for that is 50.

//   -reconst                      (rb)
//      Instructs hmetis to create partial nets within each partition
//      representing the nets that were cut during the bisection.

//   -kwayrefine                   (rb)
//      Instructs hmetis to perform a final k-way refinement once the
//      partitioning has been computed using recursive bisection

//   -fixed=string                 (rb, kway)
//      Instructs hmetis to read the file specified as the argument
//      of this parameter for specifying the groups of cells to be
//      placed in the same partition

//   -seed=int                     (rb, kway)
//      Selects the seed of the random number generator.

//   -dbglvl=int                   (rb, kway)
//      Selects the dbglvl. The value is obtained by adding the following
//      codes:
//         1       - Show coarsening progress
//         2       - Show the initial cuts
//         4       - Show the refinement progress
//         8       - Show bisection statistics
//        16       - Show bisection statistics
//        32       - Print timing statistics
//       128       - Show detailed bisection statistics
//       256       - Show detailed info about cell moves