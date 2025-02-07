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
 * @file Interpolation.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The utility function implemention of ista.
 * @version 0.1
 * @date 2021-06-29
 */

#include "Interpolation.hh"

#include <cassert>
#include <limits>

#include "Type.hh"

namespace ista {

/**
 * @brief The one dimension interpolation.
 *
 * @param x1
 * @param x2
 * @param y1
 * @param y2
 * @param x
 * @return double
 */
double LinearInterpolate(double x1, double x2, double y1, double y2, double x) {
  assert(!IsDoubleEqual(x1, x2));

  if (x >= std::numeric_limits<double>::max() ||
      x <= std::numeric_limits<double>::lowest()) {
    return x;
  }

  double slope = (y2 - y1) / (x2 - x1);
  double ret_val;

  if (x < x1) {
    ret_val = y1 - (x1 - x) * slope;  // Extrapolation.
  } else if (x > x2) {
    ret_val = y2 + (x - x2) * slope;  // Extrapolation.
  } else if (IsDoubleEqual(x, x1)) {
    ret_val = y1;  // Boundary case.
  } else if (IsDoubleEqual(x, x2)) {
    ret_val = y2;  // Boundary case.
  } else {
    ret_val = y1 + (x - x1) * slope;  // Interpolation.
  }

  return ret_val;
}

/**
 * @brief The two dimension interpolation.
 * // From
 * https://helloacm.com/cc-function-to-compute-the-bilinear-interpolation/
 * @param q11 x1, y1 value
 * @param q12 x1, y2 value
 * @param q21 x2, y1 value
 * @param q22 x2, y2 value
 * @param x1
 * @param x2
 * @param y1
 * @param y2
 * @param x
 * @param y
 * @return double
 */
double BilinearInterpolation(double q11, double q12, double q21, double q22,
                             double x1, double x2, double y1, double y2,
                             double x, double y) {
  const double x2x1 = x2 - x1;
  const double y2y1 = y2 - y1;
  const double x2x = x2 - x;
  const double y2y = y2 - y;
  const double yy1 = y - y1;
  const double xx1 = x - x1;
  return 1.0 / (x2x1 * y2y1) *
         (q11 * x2x * y2y + q21 * xx1 * y2y + q12 * x2x * yy1 +
          q22 * xx1 * yy1);
}

}  // namespace ista
