/**
 * @file Propagation.cuh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2025-01-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <vector>

namespace ista {

/**
 * @brief The struct of LibTableGPU.
 *
 */
struct LibTableGPU {
  double* _x;
  double* _y;
  unsigned _num_x = 0;
  unsigned _num_y = 0;
  double* _values;
  unsigned _num_values = 0;
  unsigned _type =
      UINT_MAX;  //!< normal(slew->cap),invert(cap->slew),slew,cap,and so on.
};

/**
 * @brief The struct of LibArcGPU.
 *
 */
struct LibArcGPU {
  LibTableGPU* _table;
  unsigned _num_table;  //!< number of tables.(first case:SSTA:12 tables;second
                        //!< case:4 tables; third case(check arc):2 tables.
                        //!< ps:the table's index order is the same as the
                        //!< LibTable::TableType(124 Line in Lib.cc ).)
};

/**
 * @brief The struct of LibDataGPU.
 *
 */
struct LibDataGPU {
  LibArcGPU* _arcs_gpu;  //!< points to GPU arc datas.
  unsigned _num_arcs;    //!< GPU arc datas.

  std::vector<LibArcGPU> _arcs;  //!< CPU arc datas.
};

}  // namespace ista
