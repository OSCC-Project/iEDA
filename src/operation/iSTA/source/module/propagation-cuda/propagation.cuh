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
 * @brief The cap unit of the lib.
 * 
 */
enum Lib_Cap_unit {
  kFF = 0,
  kPF
};

enum Lib_Time_unit {
  kNS = 0,
  kPS,
  kFS
};

/**
 * @brief The struct of Lib_Arc_GPU.
 *
 */
struct Lib_Arc_GPU {
  int _file_id = 0; //!< for debug file info.
  int _line_no = 0; //!< for debug arc info.
  Lib_Cap_unit _cap_unit = Lib_Cap_unit::kFF; //!< The cap load unit.
  Lib_Time_unit _time_unit = Lib_Time_unit::kNS; //!< The time unit.
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
