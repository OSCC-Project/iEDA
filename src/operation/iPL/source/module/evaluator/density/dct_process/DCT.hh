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

#ifndef IPL_DCT_H
#define IPL_DCT_H

#include <complex>
#include <vector>

#include "fftsg.h"

namespace ipl {

class DCT
{
 public:
  DCT(int bin_cnt_y, int bin_cnt_x, int bin_size_y, int bin_size_x);
  ~DCT();
  void init();
  void updateDensity(int y, int x, float density);

  void doDCT(bool is_calculate_phi);

  void set_thread_nums(int thread_nums) { _thread_nums = thread_nums; }


  // return function

  float** get_density_2d_ptr() const { return _bin_density;}
  float** get_electro_x_2d_ptr() const { return _electroForce_x;}
  float** get_electro_y_2d_ptr() const { return _electroForce_y;}
  float** get_phi_2d_ptr() const { return _electro_phi;}

  std::pair<float, float> get_electro_force(int y, int x);
  float get_electro_phi(int y, int x);

 private:
  float** _bin_density;
  float** _electro_phi;
  float** _electroForce_x;
  float** _electroForce_y;
  float** _buf_sequence;

  std::vector<std::complex<float>> _expk_x;
  std::vector<std::complex<float>> _expk_y;
  std::vector<std::vector<std::complex<float>>> _complex_2d_list;

  // cos/sin table (prev: w_2d)
  // length:  max(_bin_cnt_y / 2, _bin_cnt_x / 4) + _bin_cnt_x /4
  std::vector<float> _cs_table;

  // wx. length:  _bin_cnt_x
  std::vector<float> _wx;
  std::vector<float> _wx_square;

  // wy. length:  _bin_cnt_y
  std::vector<float> _wy;
  std::vector<float> _wy_square;

  // work area for bit reversal (prev: ip)
  // length: round(sqrt( max(_bin_cnt_y, _bin_cnt_x / 2) )) + 2
  std::vector<int> _work_area;

  int _bin_cnt_y;
  int _bin_cnt_x;
  int _bin_size_y;
  int _bin_size_x;

  int _thread_nums;

  void dct2UseFFT2Process(float** sequence);
  void dct2Preprocess(float** sequence, float** buf_sequence);
  void dct2Postprocess(float** buf_sequence, float** sequence);

  void idct2UseFFT2Process(float** sequence);
  void idct2Preprocess(float** sequence, float** buf_sequence);
  void idct2Postprocess(float** buf_sequence, float** sequence);

  void idctAndIdxstProcess(float** sequence);
  void idctAndIdxstPreprocess(float** sequence, float** buf_sequence);
  void idctAndIdxstPostprocess(float** buf_sequence, float** sequence);

  void idxstAndIdctProcess(float** sequence);
  void idxstAndIdctPreprocess(float** sequence, float** buf_sequence);
  void idxstAndIdctPostprocess(float** buf_sequence, float** sequence);

  void resetForInverseMatrix(float** sequence);

  int obtainIndex(int h_id, int w_id, int column_cnt);
  float obtainRealPartMultiply(std::complex<float>& x, std::complex<float>& y);
  float obtainImaginaryPartMultiply(std::complex<float>& x, std::complex<float>& y);
  std::complex<float> obtainComplexAdd(std::complex<float>& x, std::complex<float>& y);
  std::complex<float> obtainComplexSubtract(std::complex<float>& x, std::complex<float>& y);
  std::complex<float> obtainComplexMultiply(std::complex<float>& x, std::complex<float>& y);
  std::complex<float> obtainComplexConjugate(std::complex<float>& x);

  void resetBufSequence();
  void resetComplex2DList();
  void convertBufSequenceToComplex2DList();
  void convertComplex2DListToBufSequence();
};

}  // namespace ipl

#endif
