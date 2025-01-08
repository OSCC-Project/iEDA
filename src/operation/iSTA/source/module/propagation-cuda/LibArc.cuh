// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file LibArc.cuh
 * @author longshy (longshy@pcl.ac.cn)
 * @brief The file gpu implement lookup table.
 * @version 0.1
 * @date 2025-01-03
 *
 */

#pragma once

#include <cuda_runtime.h>

#include <vector>

namespace ista {

/**
 * @brief The struct of LibTableGPU.
 *
 */
struct LibTableGPU {
  double* _x;
  double* _y;
  unsigned _num_x;
  unsigned _num_y;
  double* _values;
  unsigned _num_values;
  unsigned _type;  //!< normal(slew->cap),invert(cap->slew),slew,cap,and so on.

  double findValue(double slew, double constrain_slew_or_load);
};

/**
 * @brief The struct of LibArcGPU.
 *
 */
struct LibArcGPU {
  LibTableGPU* _table;
  unsigned _num_table;  //!< number of tables.(SSTA:12 tables.)
};

/**
 * @brief The struct of LibData.
 *
 */
struct LibData {
  LibArcGPU* _arcs_gpu;  //!< points to GPU arc datas.
  unsigned _num_arc;     //!< GPU arc datas.

  std::vector<LibArcGPU> _arcs;  //!< CPU arc datas.
};

}  // namespace ista