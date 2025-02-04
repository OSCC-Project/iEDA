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
 * @brief The struct of Lib_Table_GPU.
 *
 */
struct Lib_Table_GPU {
  float* _x = nullptr;
  float* _y = nullptr;
  unsigned _num_x = 0;
  unsigned _num_y = 0;
  float* _values = nullptr;
  unsigned _num_values = 0;
  unsigned _type =
      UINT_MAX;  //!< 0(x axis denotes slew), 1(x axis denotes
                 //!< constrain_slew_or_load), , 2(x axis denotes slew, y axis
                 //!< denotes constrain_slew_or_load), 3(x axis denotes
                 //!< constrain_slew_or_load, y axis denotes slew.)
};

/**
 * @brief The struct of Lib_Arc_GPU.
 *
 */
struct Lib_Arc_GPU {
  int _file_id = 0; //!< for debug file info.
  int _line_no = 0; //!< for debug arc info.
  Lib_Table_GPU* _table = nullptr;
  unsigned _num_table = 0;  //!< number of tables.(first case:SSTA:12 tables;second
                        //!< case(delay arc):4 tables; third case(check arc):2
                        //!< tables. ps:the table's index order is the same as
                        //!< the LibTable::TableType(124 Line in Lib.cc ).)
};

/**
 * @brief The struct of Lib_Data_GPU.
 *
 */
struct Lib_Data_GPU {
  Lib_Arc_GPU* _arcs_gpu = nullptr;  //!< points to GPU arc datas.
  unsigned _num_arcs = 0;    //!< GPU arc datas.
};

}  // namespace ista
