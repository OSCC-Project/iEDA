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
#ifndef IMP_HMETIS_H
#define IMP_HMETIS_H
#include <string>
#include <vector>
namespace imp {
struct HMetis
{
  // Function to execute hmetis
  std::vector<size_t> operator()(const std::string name, const std::vector<size_t>& eptr, const std::vector<size_t>& eind, size_t nparts,
                                 const std::vector<int>& vwgt = {}, const std::vector<int>& hewgt = {});

  int seed{-1};
  float ufactor{5};
  int nruns{10};
  int ctype{1};
  int rtype{0};
  int nvcycles{1};
  bool reconst{false};
  int dbglvl{8};
  int ptype{0};
  int otype{0};
  int cmaxnet{50};
  int rmaxnet{50};
  bool kwayrefine{true};

  static const std::string ctypes[11];
  static const std::string rtypes[7];
  static const std::string ptypes[2];
  static const std::string otypes[2];
  static const std::string khmetis_binary_path;
};
// Optional parameters
/**
 * @brief -ctype=string                 (rb, kway)
    Specifies the scheme to be used for coarsening.
    The possible values are:
    0  fc1        - First-choice scheme
    1  gfc1       - Greedy first-choice scheme [default]
    2  fc2        - Alternate implementation of the fc scheme
    3  gfc2       - Alternate implementation of the gfc scheme
    4  h1         - Alternates between fc1 and gfc1 [default if nruns<20]
    5  h2         - Alternates between fc2 and gfc2
    6  h12        - Alternates between fc1, gfc1, fc2, gfc2 [default otherwise]
    7  edge1      - Edge-based coarsening
    8  gedge1     - Greedy edge-based coarsening
    9  edge2      - Alternate implementation of the edge scheme
    10 gedge2     - Alternate implementation of the gedge scheme
 */
inline const std::string HMetis::ctypes[11] = {"fc1", "gfc1", "fc2", "gfc2", "h1", "h2", "h12", "edge1", "gedge1", "edge2", "gedge2"};
/**
 * @brief -rtype=string                 (rb, kway)
    Specifies the scheme to be used for refinement.
    The possible values and the partitioning types where they apply are:
    0  fast        - Fast FM-based refinement (rb) [default]
    1  moderate    - Moderate FM-based refinement (rb)
    2  slow        - Slow FM-based refinement (rb)
    3  krandom     - Random k-way refinement (kway) [default for ptype=kway]
    4  kpfast      - Pairwise k-way FAST refinement (kway)
    5  kpmoderate  - Pairwise k-way MODERATE refinement (kway)
    6  kpslow      - Pairwise k-way SLOW refinement (kway)
 */
inline const std::string HMetis::rtypes[7] = {"fast", "moderate", "slow", "krandom", "kpfast", "kpmoderate", "kpslow"};
/**
 * @brief -ptype=string
    Specifies the scheme to be used for computing the k-way partitioning.
    The possible values are:
    0  rb       - Recursive bisection [default]
    1  kway     - Direct k-way partitioning.
 */
inline const std::string HMetis::ptypes[2] = {"rb", "kway"};
/**
 * @brief -otype=string                 (kway)
    Specifies the objective function to use for k-way partitioning.
    The possible values are:
    0  cut      - Minimize the cut [default]
    1  soed     - Minimize the sum-of-external-degrees
 */
inline const std::string HMetis::otypes[2] = {"cut", "soed"};
/*
 -ufactor=float                (rb, kway)
    Specifies the unbalance factor. The meaning of this parameters depends
    on the partitioning scheme (ptype):
      For ptype=rb
        Specifies the maximum difference between each successive bisection.
        For instance, a value of 5 leads to a 45-55 split at each bisection.
      For ptype=kway
        Specifies the maximum load imbalance among all the partitions. For
        instance, a value of 5 produces partitions in which the weight of
        the largest partition over the average weight is bounded by 5%.

 -nruns=int                    (rb, kway)
    Specifies the number of different bisections to be computed
    at each level. The final bisection corresponds to the one that
    has the smallest cut

 -nvcycles=int                 (rb, kway)
    Specifies the number of solutions to be further refined using
    V-cycle refinement. If the supplied number is k, then the best
    k bisections are further refined using V-cycling. This number
    should be less or equal to that specified for -nruns. The default
    value is one.

 -cmaxnet=int                  (rb, kway)
    Specifies the size of the largest net to be considered during
    coarsening. Any nets larger than that are ignored when determing
    which cells to merge together. Default value for that is 50.

 -rmaxnet=int                  (rb, kway)
    Specifies the size of the largest net to be considered during
    refinement. Any nets larger than that are ignored during the
    multilevel partitioning phase, and are dealt once the partitioning
    has been computed. Default value for that is 50.

 -reconst                      (rb)
    Instructs hmetis to create partial nets within each partition
    representing the nets that were cut during the bisection.

 -kwayrefine                   (rb)
    Instructs hmetis to perform a final k-way refinement once the
    partitioning has been computed using recursive bisection

 -fixed=string                 (rb, kway)
    Instructs hmetis to read the file specified as the argument
    of this parameter for specifying the groups of cells to be
    placed in the same partition

 -seed=int                     (rb, kway)
    Selects the seed of the random number generator.

 -dbglvl=int                   (rb, kway)
    Selects the dbglvl. The value is obtained by adding the following
    codes:
       1       - Show coarsening progress
       2       - Show the initial cuts
       4       - Show the refinement progress
       8       - Show bisection statistics
      16       - Show bisection statistics
      32       - Print timing statistics
     128       - Show detailed bisection statistics
     256       - Show detailed info about cell moves
  *
  */
// extern "C" void HMETIS_PartRecursive(int nvtxs, int nhedges, int* vwgts, int* eptr, int* eind, int* hewgts, int nparts, int ubfactor,
//  int* options, int* part, int* edgecut);
}  // namespace imp
#endif