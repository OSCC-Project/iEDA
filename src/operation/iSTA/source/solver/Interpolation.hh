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
/**
 * @file Interpolation.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The utility function of ista.
 * @version 0.1
 * @date 2021-06-29
 *
 *
 */

#pragma once

#include <cstdlib>

#include "include/Type.hh"

namespace ista {

double LinearInterpolate(double x1, double x2, double y1, double y2, double x);
double BilinearInterpolation(double q11, double q12, double q21, double q22,
                             double x1, double x2, double y1, double y2,
                             double x, double y);
}  // namespace ista
