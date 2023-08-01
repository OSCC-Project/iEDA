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

#ifndef IPL_FFT_H
#define IPL_FFT_H

#include <vector>

#include "fftsg.h"

namespace ipl {

class FFT
{
 public:
  FFT();
  FFT(int binCnt_x, int binCnt_y, int binSize_x, int binSize_y);
  ~FFT();

  void init();
  void updateDensity(int x, int y, float density);

  void doFFT(bool is_calculate_phi);

  void set_thread_nums(int thread_nums) { _thread_nums = thread_nums; }

  // return func

  float** get_density_2d_ptr() const { return _bin_density; }
  float** get_electro_x_2d_ptr() const { return _electroForce_x; }
  float** get_electro_y_2d_ptr() const { return _electroForce_y; }
  float** get_phi_2d_ptr() const { return _electro_phi; }

  std::pair<float, float> get_electro_force(int x, int y);
  float get_electro_phi(int x, int y);

 private:
  // 2D array; width: binCntX_, height: binCntY_
  // No hope to use Vector at this moment...
  float** _bin_density;
  float** _electro_phi;
  float** _electroForce_x;
  float** _electroForce_y;

  // cos/sin table (prev: w_2d)
  // length:  max(binCntY, binCntX) * 3 / 2
  std::vector<float> _cs_table;

  // wx. length:  binCntX_
  std::vector<float> _wx;
  std::vector<float> _wx_square;

  // wy. length:  binCntY_
  std::vector<float> _wy;
  std::vector<float> _wy_square;

  // work area for bit reversal (prev: ip)
  // length: round(sqrt( max(binCntY_, binCntX_) )) + 2
  std::vector<int> _work_area;

  int _binCnt_x;
  int _binCnt_y;
  int _binSize_x;
  int _binSize_y;

  int _thread_nums;
};

}  // namespace ipl

#endif  // IPL_FFT_H