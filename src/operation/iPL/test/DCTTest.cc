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

#include <iostream>
#include <string>

#include "gtest/gtest.h"
#include "module/evaluator/density/dct_process/DCT.hh"
#include "module/evaluator/density/dct_process/FFT.hh"

namespace ipl {

class DCTTestInterface : public testing::Test
{
  void SetUp() {}
  void TearDown() final {}
};

TEST_F(DCTTestInterface, diff_test)
{
  static const int matrix_size = 4;
  float test_matrix[matrix_size][matrix_size] = {{1.9968, 8.0812, 4.6889, 5.5983},
                                                 {9.7861, 4.2651, 9.3779, 8.6842},
                                                 {5.7838, 4.1267, 5.0219, 9.0164},
                                                 {4.1550, 5.3115, 7.4724, 3.0068}};

  FFT origin_fft(matrix_size, matrix_size, 2, 2);
  origin_fft.updateDensity(0, 0, test_matrix[0][0]);
  origin_fft.updateDensity(0, 1, test_matrix[0][1]);
  origin_fft.updateDensity(0, 2, test_matrix[0][2]);
  origin_fft.updateDensity(0, 3, test_matrix[0][3]);
  origin_fft.updateDensity(1, 0, test_matrix[1][0]);
  origin_fft.updateDensity(1, 1, test_matrix[1][1]);
  origin_fft.updateDensity(1, 2, test_matrix[1][2]);
  origin_fft.updateDensity(1, 3, test_matrix[1][3]);
  origin_fft.updateDensity(2, 0, test_matrix[2][0]);
  origin_fft.updateDensity(2, 1, test_matrix[2][1]);
  origin_fft.updateDensity(2, 2, test_matrix[2][2]);
  origin_fft.updateDensity(2, 3, test_matrix[2][3]);
  origin_fft.updateDensity(3, 0, test_matrix[3][0]);
  origin_fft.updateDensity(3, 1, test_matrix[3][1]);
  origin_fft.updateDensity(3, 2, test_matrix[3][2]);
  origin_fft.updateDensity(3, 3, test_matrix[3][3]);

  origin_fft.set_thread_nums(1);
  origin_fft.doFFT(false);

  for (int i = 0; i < matrix_size; i++) {
    for (int j = 0; j < matrix_size; j++) {
      auto pair = origin_fft.get_electro_force(j, i);
      std::cout << "(" << pair.first << " , " << pair.second << ") ";
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;

  DCT modify_dct(matrix_size, matrix_size, 2, 2);
  modify_dct.updateDensity(0, 0, test_matrix[0][0]);
  modify_dct.updateDensity(0, 1, test_matrix[0][1]);
  modify_dct.updateDensity(0, 2, test_matrix[0][2]);
  modify_dct.updateDensity(0, 3, test_matrix[0][3]);
  modify_dct.updateDensity(1, 0, test_matrix[1][0]);
  modify_dct.updateDensity(1, 1, test_matrix[1][1]);
  modify_dct.updateDensity(1, 2, test_matrix[1][2]);
  modify_dct.updateDensity(1, 3, test_matrix[1][3]);
  modify_dct.updateDensity(2, 0, test_matrix[2][0]);
  modify_dct.updateDensity(2, 1, test_matrix[2][1]);
  modify_dct.updateDensity(2, 2, test_matrix[2][2]);
  modify_dct.updateDensity(2, 3, test_matrix[2][3]);
  modify_dct.updateDensity(3, 0, test_matrix[3][0]);
  modify_dct.updateDensity(3, 1, test_matrix[3][1]);
  modify_dct.updateDensity(3, 2, test_matrix[3][2]);
  modify_dct.updateDensity(3, 3, test_matrix[3][3]);

  modify_dct.set_thread_nums(1);
  modify_dct.doDCT(false);

  for (int i = 0; i < matrix_size; i++) {
    for (int j = 0; j < matrix_size; j++) {
      auto pair = modify_dct.get_electro_force(j, i);
      std::cout << "(" << pair.first << " , " << pair.second << ") ";
    }
    std::cout << std::endl;
  }

  // test_matrix;
}

}  // namespace ipl