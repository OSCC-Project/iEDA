
#include "DCT.hh"

#include <cassert>
#include <cmath>
#include <ctime>
#include <iostream>

#include "omp.h"

#define DCT_PI 3.141592653589793238462L

namespace ipl {

DCT::DCT(int bin_cnt_y, int bin_cnt_x, int bin_size_y, int bin_size_x)
    : _bin_cnt_y(bin_cnt_y), _bin_cnt_x(bin_cnt_x), _bin_size_y(bin_size_y), _bin_size_x(bin_size_x), _thread_nums(1)
{
  init();
}

DCT::~DCT()
{
  for (int i = 0; i < _bin_cnt_y; i++) {
    delete (_bin_density[i]);
    delete (_electro_phi[i]);
    delete (_electroForce_x[i]);
    delete (_electroForce_y[i]);
  }
  delete (_bin_density);
  delete (_electro_phi);
  delete (_electroForce_x);
  delete (_electroForce_y);
}

void DCT::init()
{
  _bin_density = new float*[_bin_cnt_y];
  _electro_phi = new float*[_bin_cnt_y];
  _electroForce_x = new float*[_bin_cnt_y];
  _electroForce_y = new float*[_bin_cnt_y];
  _buf_sequence = new float*[_bin_cnt_y];

  for (int i = 0; i < _bin_cnt_y; i++) {
    _bin_density[i] = new float[_bin_cnt_x];
    _electro_phi[i] = new float[_bin_cnt_x];
    _electroForce_x[i] = new float[_bin_cnt_x];
    _electroForce_y[i] = new float[_bin_cnt_x];
    _buf_sequence[i] = new float[(_bin_cnt_x + 2)];

    for (int j = 0; j < _bin_cnt_x; j++) {
      _bin_density[i][j] = _electro_phi[i][j] = _electroForce_x[i][j] = _electroForce_y[i][j] = 0.0f;
    }
    for (int j = 0; j < _bin_cnt_x + 2; j++) {
      _buf_sequence[i][j] = 0.0f;
    }
  }

  _cs_table.resize(std::max(_bin_cnt_y / 2, _bin_cnt_x / 4) + _bin_cnt_x / 4, 0);
  _wx.resize(_bin_cnt_x, 0);
  _wx_square.resize(_bin_cnt_x, 0);
  _wy.resize(_bin_cnt_y, 0);
  _wy_square.resize(_bin_cnt_y, 0);
  _work_area.resize(round(sqrt(std::max(_bin_cnt_y, _bin_cnt_x / 2))) + 2, 0);

  _expk_x.resize(_bin_cnt_x);
  _expk_y.resize(_bin_cnt_y);

  for (int k = 0; k < _bin_cnt_x; k++) {
    float pi_k_by_2cnt = DCT_PI * k / (2 * _bin_cnt_x);
    _expk_x[k] = {std::cos(pi_k_by_2cnt), -std::sin(pi_k_by_2cnt)};

    _wx[k] = 2.0 * DCT_PI * static_cast<float>(k) / _bin_cnt_x;
    _wx_square[k] = _wx[k] * _wx[k];
  }

  for (int k = 0; k < _bin_cnt_y; k++) {
    float pi_k_by_2cnt = DCT_PI * k / (2 * _bin_cnt_y);
    _expk_y[k] = {std::cos(pi_k_by_2cnt), -std::sin(pi_k_by_2cnt)};

    float y_x_ratio = static_cast<float>(_bin_size_y) / _bin_size_x;
    _wy[k] = 2.0 * DCT_PI * static_cast<float>(k) / _bin_cnt_y * y_x_ratio;
    _wy_square[k] = _wy[k] * _wy[k];
  }

  _complex_2d_list.resize(_bin_cnt_y);
  int column_num = _bin_cnt_x / 2 + 1;
  for (int i = 0; i < _bin_cnt_y; i++) {
    _complex_2d_list[i].resize(column_num, {0, 0});
  }
}

void DCT::updateDensity(int y, int x, float density)
{
  _bin_density[y][x] = density;
}

std::pair<float, float> DCT::get_electro_force(int y, int x)
{
  return std::make_pair(_electroForce_y[y][x], _electroForce_x[y][x]);
}

float DCT::get_electro_phi(int y, int x)
{
  return _electro_phi[y][x];
}

void DCT::doDCT(bool is_calculate_phi)
{
  // obtain auv
  dct2UseFFT2Process(_bin_density);

  // obtain auv_by_wu2_plus_wv2_wu
#pragma omp parallel for num_threads(_thread_nums)
  for (int i = 0; i < _bin_cnt_y; i++) {
    float wu = _wy[i];
    float wu2 = _wy_square[i];

    for (int j = 0; j < _bin_cnt_x; j++) {
      float wv = _wx[j];
      float wv2 = _wx_square[j];

      float auv = _bin_density[i][j];
      float phi = 0.0f;
      float electro_x = 0.0f, electro_y = 0.0f;
      if (i == 0 && j == 0) {
        phi = electro_x = electro_y = 0.0f;
      } else {
        float auv_by_wu2_plus_wv2 = auv / (wu2 + wv2);
        if (is_calculate_phi) {
          phi = auv_by_wu2_plus_wv2;
        }

        electro_y = auv_by_wu2_plus_wv2 * wu * 0.5;
        electro_x = auv_by_wu2_plus_wv2 * wv * 0.5;
      }
      _electroForce_x[i][j] = electro_x;
      _electroForce_y[i][j] = electro_y;
      _electro_phi[i][j] = phi;
    }
  }

  idxstAndIdctProcess(_electroForce_y);
  idctAndIdxstProcess(_electroForce_x);

  if (is_calculate_phi) {
    idct2UseFFT2Process(_electro_phi);
  }
}

void DCT::dct2UseFFT2Process(float** sequence)
{
  resetBufSequence();

  // std::clock_t pre_clock_start, pre_clock_end;
  // pre_clock_start = clock();
  dct2Preprocess(sequence, _buf_sequence);
  // pre_clock_end = clock();
  // std::cout << "DCT2 preprocess used time: " << double(pre_clock_end - pre_clock_start) / CLOCKS_PER_SEC << " s" << std::endl;

  // std::clock_t clock_start, clock_end;
  // clock_start = clock();
  rdft2d(_bin_cnt_y, _bin_cnt_x, 1, _buf_sequence, NULL, (int*) &_work_area[0], (float*) &_cs_table[0]);
  rdft2dsort(_bin_cnt_y, _bin_cnt_x, 1, _buf_sequence);
  // clock_end = clock();
  // std::cout << "sequence size: " << _bin_cnt_y * _bin_cnt_x << std::endl;
  // std::cout << "DCT2 rfft2d used time: " << double(clock_end - clock_start) / CLOCKS_PER_SEC << " s" << std::endl;

  // std::clock_t post_clock_start, post_clock_end;
  // post_clock_start = clock();
  dct2Postprocess(_buf_sequence, sequence);
  // post_clock_end = clock();
  // std::cout << "postprocess iter: " << test_iter++ << std::endl;
  // std::cout << "DCT2 postprocess used time: " << double(post_clock_end - post_clock_start) / CLOCKS_PER_SEC << " s" << std::endl;
}

void DCT::dct2Preprocess(float** sequence, float** buf_sequence)
{
  int half_x_cnt = _bin_cnt_x / 2;
#pragma omp parallel for num_threads(_thread_nums)
  for (int h_id = 0; h_id < _bin_cnt_y; h_id++) {
    for (int w_id = 0; w_id < _bin_cnt_x; w_id++) {
      int tmp_index, index_1, index_2;
      int condition = (((h_id & 1) == 0) << 1) | ((w_id & 1) == 0);
      switch (condition) {
        case 0:
          tmp_index = obtainIndex(2 * _bin_cnt_y - (h_id + 1), _bin_cnt_x - (w_id + 1) / 2, half_x_cnt);
          break;
        case 1:
          tmp_index = obtainIndex(2 * _bin_cnt_y - (h_id + 1), w_id / 2, half_x_cnt);
          break;
        case 2:
          tmp_index = obtainIndex(h_id, _bin_cnt_x - (w_id + 1) / 2, half_x_cnt);
          break;
        case 3:
          tmp_index = obtainIndex(h_id, w_id / 2, half_x_cnt);
          break;
        default:
          break;
      }
      index_1 = tmp_index / _bin_cnt_x;
      index_2 = tmp_index % _bin_cnt_x;

      buf_sequence[index_1][index_2] = sequence[h_id][w_id];
    }
  }
}

void DCT::dct2Postprocess(float** buf_sequence, float** sequence)
{
  resetComplex2DList();
  convertBufSequenceToComplex2DList();

  int half_cnt_y = _bin_cnt_y / 2;
  int half_cnt_x = _bin_cnt_x / 2;
  float four_over_yx = (float) (4.0 / (_bin_cnt_y * _bin_cnt_x));
  float two_over_yx = (float) (2.0 / (_bin_cnt_y * _bin_cnt_x));

#pragma omp parallel for num_threads(_thread_nums)
  for (int h_id = 0; h_id < half_cnt_y; h_id++) {
    for (int w_id = 0; w_id < half_cnt_x; w_id++) {
      int condition = ((h_id != 0) << 1) | (w_id != 0);
      switch (condition) {
        case 0: {
          sequence[0][0] = _complex_2d_list[0][0].real() * four_over_yx;
          sequence[0][half_cnt_x] = obtainRealPartMultiply(_expk_x[half_cnt_x], _complex_2d_list[0][half_cnt_x]) * four_over_yx;
          sequence[half_cnt_y][0] = _expk_y[half_cnt_y].real() * _complex_2d_list[half_cnt_y][0].real() * four_over_yx;
          sequence[half_cnt_y][half_cnt_x] = _expk_y[half_cnt_y].real()
                                             * obtainRealPartMultiply(_expk_x[half_cnt_x], _complex_2d_list[half_cnt_y][half_cnt_x])
                                             * four_over_yx;
          break;
        }
        case 1: {
          std::complex<float> tmp_complex_1 = _complex_2d_list[0][w_id];
          sequence[0][w_id] = obtainRealPartMultiply(_expk_x[w_id], tmp_complex_1) * four_over_yx;
          sequence[0][_bin_cnt_x - w_id] = -obtainImaginaryPartMultiply(_expk_x[w_id], tmp_complex_1) * four_over_yx;

          std::complex<float> tmp_complex_2 = _complex_2d_list[half_cnt_y][w_id];
          sequence[half_cnt_y][w_id] = _expk_y[half_cnt_y].real() * obtainRealPartMultiply(_expk_x[w_id], tmp_complex_2) * four_over_yx;
          sequence[half_cnt_y][_bin_cnt_x - w_id]
              = -_expk_y[half_cnt_y].real() * obtainImaginaryPartMultiply(_expk_x[w_id], tmp_complex_2) * four_over_yx;
          break;
        }
        case 2: {
          std::complex<float> tmp_complex_1, tmp_complex_2, tmp_complex_up, tmp_complex_down;
          tmp_complex_1 = _complex_2d_list[h_id][0];
          tmp_complex_2 = _complex_2d_list[(_bin_cnt_y - h_id)][0];
          float tmp_complex_up_real = _expk_y[h_id].real() * (tmp_complex_1.real() + tmp_complex_2.real())
                                      + _expk_y[h_id].imag() * (tmp_complex_2.imag() - tmp_complex_1.imag());
          float tmp_complex_down_real = -_expk_y[h_id].imag() * (tmp_complex_1.real() + tmp_complex_2.real())
                                        + _expk_y[h_id].real() * (tmp_complex_2.imag() - tmp_complex_1.imag());
          sequence[h_id][0] = tmp_complex_up_real * two_over_yx;
          sequence[_bin_cnt_y - h_id][0] = tmp_complex_down_real * two_over_yx;

          tmp_complex_1 = obtainComplexAdd(_complex_2d_list[h_id][half_cnt_x], _complex_2d_list[_bin_cnt_y - h_id][half_cnt_x]);
          tmp_complex_2 = obtainComplexSubtract(_complex_2d_list[h_id][half_cnt_x], _complex_2d_list[_bin_cnt_y - h_id][half_cnt_x]);
          tmp_complex_up = {_expk_y[h_id].real() * tmp_complex_1.real() - _expk_y[h_id].imag() * tmp_complex_2.imag(),
                            _expk_y[h_id].real() * tmp_complex_1.imag() + _expk_y[h_id].imag() * tmp_complex_2.real()};
          tmp_complex_down = {-_expk_y[h_id].imag() * tmp_complex_1.real() - _expk_y[h_id].real() * tmp_complex_2.imag(),
                              -_expk_y[h_id].imag() * tmp_complex_1.imag() + _expk_y[h_id].real() * tmp_complex_2.real()};
          sequence[h_id][half_cnt_x] = obtainRealPartMultiply(_expk_x[half_cnt_x], tmp_complex_up) * two_over_yx;
          sequence[_bin_cnt_y - h_id][half_cnt_x] = obtainRealPartMultiply(_expk_x[half_cnt_x], tmp_complex_down) * two_over_yx;
          break;
        }
        case 3: {
          std::complex<float> tmp_complex_1, tmp_complex_2, tmp_complex_up, tmp_complex_down;
          tmp_complex_1 = obtainComplexAdd(_complex_2d_list[h_id][w_id], _complex_2d_list[_bin_cnt_y - h_id][w_id]);
          tmp_complex_2 = obtainComplexSubtract(_complex_2d_list[h_id][w_id], _complex_2d_list[_bin_cnt_y - h_id][w_id]);
          tmp_complex_up = {_expk_y[h_id].real() * tmp_complex_1.real() - _expk_y[h_id].imag() * tmp_complex_2.imag(),
                            _expk_y[h_id].real() * tmp_complex_1.imag() + _expk_y[h_id].imag() * tmp_complex_2.real()};
          tmp_complex_down = {-_expk_y[h_id].imag() * tmp_complex_1.real() - _expk_y[h_id].real() * tmp_complex_2.imag(),
                              -_expk_y[h_id].imag() * tmp_complex_1.imag() + _expk_y[h_id].real() * tmp_complex_2.real()};
          sequence[h_id][w_id] = obtainRealPartMultiply(_expk_x[w_id], tmp_complex_up) * two_over_yx;
          sequence[_bin_cnt_y - h_id][w_id] = obtainRealPartMultiply(_expk_x[w_id], tmp_complex_down) * two_over_yx;
          sequence[h_id][_bin_cnt_x - w_id] = -obtainImaginaryPartMultiply(_expk_x[w_id], tmp_complex_up) * two_over_yx;
          sequence[_bin_cnt_y - h_id][_bin_cnt_x - w_id] = -obtainImaginaryPartMultiply(_expk_x[w_id], tmp_complex_down) * two_over_yx;
          break;
        }
        default:
          assert(0);
          break;
      }
    }
  }
}

void DCT::idct2UseFFT2Process(float** sequence)
{
  resetBufSequence();
  idct2Preprocess(sequence, _buf_sequence);
  rdft2dsort(_bin_cnt_y, _bin_cnt_x, -1, _buf_sequence);
  rdft2d(_bin_cnt_y, _bin_cnt_x, -1, _buf_sequence, NULL, (int*) &_work_area[0], (float*) &_cs_table[0]);
  resetForInverseMatrix(_buf_sequence);
  idct2Postprocess(_buf_sequence, sequence);
}

void DCT::idct2Preprocess(float** sequence, float** buf_sequence)
{
  resetComplex2DList();
  int half_cnt_y = _bin_cnt_y / 2;
  int half_cnt_x = _bin_cnt_x / 2;

#pragma omp parallel for num_threads(_thread_nums)
  for (int h_id = 0; h_id < half_cnt_y; h_id++) {
    for (int w_id = 0; w_id < half_cnt_x; w_id++) {
      int condition = ((h_id != 0) << 1) | (w_id != 0);
      switch (condition) {
        case 0: {
          float tmp_1;
          std::complex<float> tmp_complex_up, tmp_complex_1, tmp_complex_2;

          _complex_2d_list[0][0] = {sequence[0][0], 0};

          tmp_1 = sequence[0][half_cnt_x];
          tmp_complex_up = {tmp_1, tmp_1};
          tmp_complex_1 = obtainComplexMultiply(_expk_x[half_cnt_x], tmp_complex_up);
          _complex_2d_list[0][half_cnt_x] = obtainComplexConjugate(tmp_complex_1);

          tmp_1 = sequence[half_cnt_y][0];
          tmp_complex_up = {tmp_1, tmp_1};
          tmp_complex_1 = obtainComplexMultiply(_expk_y[half_cnt_y], tmp_complex_up);
          _complex_2d_list[half_cnt_y][0] = obtainComplexConjugate(tmp_complex_1);

          tmp_1 = sequence[half_cnt_y][half_cnt_x];
          tmp_complex_up = {0, 2 * tmp_1};
          tmp_complex_1 = obtainComplexMultiply(_expk_y[half_cnt_y], _expk_x[half_cnt_x]);
          tmp_complex_2 = obtainComplexMultiply(tmp_complex_1, tmp_complex_up);
          _complex_2d_list[half_cnt_y][half_cnt_x] = obtainComplexConjugate(tmp_complex_2);
          break;
        }
        case 1: {
          std::complex<float> tmp_complex_up, tmp_complex_1, tmp_complex_2;
          tmp_complex_up = {sequence[0][w_id], sequence[0][_bin_cnt_x - w_id]};
          tmp_complex_1 = obtainComplexMultiply(_expk_x[w_id], tmp_complex_up);
          _complex_2d_list[0][w_id] = obtainComplexConjugate(tmp_complex_1);

          float tmp_1 = sequence[half_cnt_y][w_id];
          float tmp_2 = sequence[half_cnt_y][_bin_cnt_x - w_id];
          tmp_complex_up = {tmp_1 - tmp_2, tmp_1 + tmp_2};

          tmp_complex_1 = obtainComplexMultiply(_expk_y[half_cnt_y], _expk_x[w_id]);
          tmp_complex_2 = obtainComplexMultiply(tmp_complex_1, tmp_complex_up);
          _complex_2d_list[half_cnt_y][w_id] = obtainComplexConjugate(tmp_complex_2);
          break;
        }
        case 2: {
          float tmp_1, tmp_3;
          std::complex<float> tmp_complex_up, tmp_complex_down, tmp_complex_1, tmp_complex_2;

          tmp_1 = sequence[h_id][0];
          tmp_3 = sequence[_bin_cnt_y - h_id][0];
          tmp_complex_up = {tmp_1, tmp_3};
          tmp_complex_down = {tmp_3, tmp_1};

          tmp_complex_1 = obtainComplexMultiply(_expk_y[h_id], tmp_complex_up);
          _complex_2d_list[h_id][0] = obtainComplexConjugate(tmp_complex_1);
          tmp_complex_2 = obtainComplexMultiply(_expk_y[_bin_cnt_y - h_id], tmp_complex_down);
          _complex_2d_list[_bin_cnt_y - h_id][0] = obtainComplexConjugate(tmp_complex_2);

          tmp_1 = sequence[h_id][half_cnt_x];
          tmp_3 = sequence[_bin_cnt_y - h_id][half_cnt_x];
          tmp_complex_up = {tmp_1 - tmp_3, tmp_3 + tmp_1};
          tmp_complex_down = {tmp_3 - tmp_1, tmp_1 + tmp_3};

          tmp_complex_1 = obtainComplexMultiply(_expk_y[h_id], _expk_x[half_cnt_x]);
          tmp_complex_2 = obtainComplexMultiply(tmp_complex_1, tmp_complex_up);
          _complex_2d_list[h_id][half_cnt_x] = obtainComplexConjugate(tmp_complex_2);
          tmp_complex_1 = obtainComplexMultiply(_expk_y[_bin_cnt_y - h_id], _expk_x[half_cnt_x]);
          tmp_complex_2 = obtainComplexMultiply(tmp_complex_1, tmp_complex_down);
          _complex_2d_list[_bin_cnt_y - h_id][half_cnt_x] = obtainComplexConjugate(tmp_complex_2);
          break;
        }
        case 3: {
          float tmp_1 = sequence[h_id][w_id];
          float tmp_2 = sequence[h_id][_bin_cnt_x - w_id];
          float tmp_3 = sequence[_bin_cnt_y - h_id][w_id];
          float tmp_4 = sequence[_bin_cnt_y - h_id][_bin_cnt_x - w_id];

          std::complex<float> tmp_complex_up, tmp_complex_down, tmp_complex_1, tmp_complex_2;
          tmp_complex_up = {tmp_1 - tmp_4, tmp_3 + tmp_2};
          tmp_complex_down = {tmp_3 - tmp_2, tmp_1 + tmp_4};

          tmp_complex_1 = obtainComplexMultiply(_expk_y[h_id], _expk_x[w_id]);
          tmp_complex_2 = obtainComplexMultiply(tmp_complex_1, tmp_complex_up);
          _complex_2d_list[h_id][w_id] = obtainComplexConjugate(tmp_complex_2);

          tmp_complex_1 = obtainComplexMultiply(_expk_y[_bin_cnt_y - h_id], _expk_x[w_id]);
          tmp_complex_2 = obtainComplexMultiply(tmp_complex_1, tmp_complex_down);
          _complex_2d_list[_bin_cnt_y - h_id][w_id] = obtainComplexConjugate(tmp_complex_2);
          break;
        }

        default:
          assert(0);
          break;
      }
    }
  }
  convertComplex2DListToBufSequence();
}

void DCT::idct2Postprocess(float** buf_sequence, float** sequence)
{
  int bin_cnt_yx = _bin_cnt_y * _bin_cnt_x;
#pragma omp parallel for num_threads(_thread_nums)
  for (int h_id = 0; h_id < _bin_cnt_y; h_id++) {
    for (int w_id = 0; w_id < _bin_cnt_x; w_id++) {
      int condition = ((h_id < _bin_cnt_y / 2) << 1) | (w_id < _bin_cnt_x / 2);
      int index_1, index_2;
      switch (condition) {
        case 0: {
          index_1 = ((_bin_cnt_y - h_id) << 1) - 1;
          index_2 = ((_bin_cnt_x - w_id) << 1) - 1;
          break;
        }
        case 1: {
          index_1 = ((_bin_cnt_y - h_id) << 1) - 1;
          index_2 = (w_id << 1);
          break;
        }
        case 2: {
          index_1 = (h_id << 1);
          index_2 = ((_bin_cnt_x - w_id) << 1) - 1;
          break;
        }
        case 3: {
          index_1 = (h_id << 1);
          index_2 = (w_id << 1);
          break;
        }
        default:
          assert(0);
          break;
      }
      sequence[index_1][index_2] = buf_sequence[h_id][w_id] * bin_cnt_yx;
    }
  }
}

void DCT::idctAndIdxstProcess(float** sequence)
{
  resetBufSequence();

  // std::clock_t pre_clock_start, pre_clock_end;
  // pre_clock_start = clock();
  idctAndIdxstPreprocess(sequence, _buf_sequence);
  // pre_clock_end = clock();
  // std::cout << "IdctAndIdxst preprocess used time: " << double(pre_clock_end - pre_clock_start) / CLOCKS_PER_SEC << " s" << std::endl;

  // std::clock_t clock_start, clock_end;
  // clock_start = clock();
  rdft2dsort(_bin_cnt_y, _bin_cnt_x, -1, _buf_sequence);
  rdft2d(_bin_cnt_y, _bin_cnt_x, -1, _buf_sequence, NULL, (int*) &_work_area[0], (float*) &_cs_table[0]);
  resetForInverseMatrix(_buf_sequence);
  // clock_end = clock();
  // std::cout << "sequence size: " << _bin_cnt_y * _bin_cnt_x << std::endl;
  // std::cout << "IdctAndIdxst rfft2d used time: " << double(clock_end - clock_start) / CLOCKS_PER_SEC << " s" << std::endl;

  // std::clock_t post_clock_start, post_clock_end;
  // post_clock_start = clock();
  idctAndIdxstPostprocess(_buf_sequence, sequence);
  // post_clock_end = clock();
  // std::cout << "IdctAndIdxst postprocess used time: " << double(post_clock_end - post_clock_start) / CLOCKS_PER_SEC << " s" << std::endl;
}

void DCT::idctAndIdxstPreprocess(float** sequence, float** buf_sequence)
{
  resetComplex2DList();
  int half_cnt_y = _bin_cnt_y / 2;
  int half_cnt_x = _bin_cnt_x / 2;

#pragma omp parallel for num_threads(_thread_nums)
  for (int h_id = 0; h_id < half_cnt_y; h_id++) {
    for (int w_id = 0; w_id < half_cnt_x; w_id++) {
      int condition = ((h_id != 0) << 1) | (w_id != 0);
      switch (condition) {
        case 0: {
          float tmp_1;
          std::complex<float> tmp_complex_up, tmp_complex_1, tmp_complex_2;
          _complex_2d_list[0][0] = {0.0f, 0.0f};
          tmp_1 = sequence[0][half_cnt_x];
          tmp_complex_up = {tmp_1, tmp_1};
          tmp_complex_1 = obtainComplexMultiply(_expk_x[half_cnt_x], tmp_complex_up);
          _complex_2d_list[0][half_cnt_x] = obtainComplexConjugate(tmp_complex_1);

          _complex_2d_list[half_cnt_y][0] = {0.0f, 0.0f};
          tmp_1 = sequence[half_cnt_y][half_cnt_x];
          tmp_complex_up = {0.0f, 2 * tmp_1};

          tmp_complex_1 = obtainComplexMultiply(_expk_y[half_cnt_y], _expk_x[half_cnt_x]);
          tmp_complex_2 = obtainComplexMultiply(tmp_complex_1, tmp_complex_up);
          _complex_2d_list[half_cnt_y][half_cnt_x] = obtainComplexConjugate(tmp_complex_2);
          break;
        }
        case 1: {
          std::complex<float> tmp_complex_up, tmp_complex_1, tmp_complex_2;
          tmp_complex_up = {sequence[0][_bin_cnt_x - w_id], sequence[0][w_id]};
          tmp_complex_1 = obtainComplexMultiply(_expk_x[w_id], tmp_complex_up);
          _complex_2d_list[0][w_id] = obtainComplexConjugate(tmp_complex_1);

          float tmp_1 = sequence[half_cnt_y][_bin_cnt_x - w_id];
          float tmp_2 = sequence[half_cnt_y][w_id];
          tmp_complex_up = {tmp_1 - tmp_2, tmp_1 + tmp_2};

          tmp_complex_1 = obtainComplexMultiply(_expk_y[half_cnt_y], _expk_x[w_id]);
          tmp_complex_2 = obtainComplexMultiply(tmp_complex_1, tmp_complex_up);
          _complex_2d_list[half_cnt_y][w_id] = obtainComplexConjugate(tmp_complex_2);
          break;
        }
        case 2: {
          float tmp_1, tmp_3;
          std::complex<float> tmp_complex_up, tmp_complex_down, tmp_complex_1, tmp_complex_2;

          _complex_2d_list[h_id][0] = {0.0f, 0.0f};
          _complex_2d_list[_bin_cnt_y - h_id][0] = {0.0f, 0.0f};

          tmp_1 = sequence[h_id][half_cnt_x];
          tmp_3 = sequence[_bin_cnt_y - h_id][half_cnt_x];
          tmp_complex_up = {tmp_1 - tmp_3, tmp_3 + tmp_1};
          tmp_complex_down = {tmp_3 - tmp_1, tmp_1 + tmp_3};

          tmp_complex_1 = obtainComplexMultiply(_expk_y[h_id], _expk_x[half_cnt_x]);
          tmp_complex_2 = obtainComplexMultiply(tmp_complex_1, tmp_complex_up);
          _complex_2d_list[h_id][half_cnt_x] = obtainComplexConjugate(tmp_complex_2);

          tmp_complex_1 = obtainComplexMultiply(_expk_y[_bin_cnt_y - h_id], _expk_x[half_cnt_x]);
          tmp_complex_2 = obtainComplexMultiply(tmp_complex_1, tmp_complex_down);
          _complex_2d_list[_bin_cnt_y - h_id][half_cnt_x] = obtainComplexConjugate(tmp_complex_2);
          break;
        }
        case 3: {
          float tmp_1 = sequence[h_id][_bin_cnt_x - w_id];
          float tmp_2 = sequence[h_id][w_id];
          float tmp_3 = sequence[_bin_cnt_y - h_id][_bin_cnt_x - w_id];
          float tmp_4 = sequence[_bin_cnt_y - h_id][w_id];

          std::complex<float> tmp_complex_up, tmp_complex_down, tmp_complex_1, tmp_complex_2;
          tmp_complex_up = {tmp_1 - tmp_4, tmp_3 + tmp_2};
          tmp_complex_down = {tmp_3 - tmp_2, tmp_1 + tmp_4};

          tmp_complex_1 = obtainComplexMultiply(_expk_y[h_id], _expk_x[w_id]);
          tmp_complex_2 = obtainComplexMultiply(tmp_complex_1, tmp_complex_up);
          _complex_2d_list[h_id][w_id] = obtainComplexConjugate(tmp_complex_2);

          tmp_complex_1 = obtainComplexMultiply(_expk_y[_bin_cnt_y - h_id], _expk_x[w_id]);
          tmp_complex_2 = obtainComplexMultiply(tmp_complex_1, tmp_complex_down);
          _complex_2d_list[_bin_cnt_y - h_id][w_id] = obtainComplexConjugate(tmp_complex_2);
          break;
        }

        default:
          assert(0);
          break;
      }
    }
  }
  convertComplex2DListToBufSequence();
}

void DCT::idctAndIdxstPostprocess(float** buf_sequence, float** sequence)
{
  int bin_cnt_yx = _bin_cnt_y * _bin_cnt_x;
#pragma omp parallel for num_threads(_thread_nums)
  for (int h_id = 0; h_id < _bin_cnt_y; h_id++) {
    for (int w_id = 0; w_id < _bin_cnt_x; w_id++) {
      int condition = ((h_id < _bin_cnt_y / 2) << 1) | (w_id < _bin_cnt_x / 2);
      int index_1, index_2;
      switch (condition) {
        case 0: {
          index_1 = ((_bin_cnt_y - h_id) << 1) - 1;
          index_2 = ((_bin_cnt_x - w_id) << 1) - 1;
          sequence[index_1][index_2] = -buf_sequence[h_id][w_id] * bin_cnt_yx;
          break;
        }
        case 1: {
          index_1 = ((_bin_cnt_y - h_id) << 1) - 1;
          index_2 = (w_id << 1);
          sequence[index_1][index_2] = buf_sequence[h_id][w_id] * bin_cnt_yx;
          break;
        }
        case 2: {
          index_1 = (h_id << 1);
          index_2 = ((_bin_cnt_x - w_id) << 1) - 1;
          sequence[index_1][index_2] = -buf_sequence[h_id][w_id] * bin_cnt_yx;
          break;
        }
        case 3: {
          index_1 = (h_id << 1);
          index_2 = (w_id << 1);
          sequence[index_1][index_2] = buf_sequence[h_id][w_id] * bin_cnt_yx;
          break;
        }
        default:
          assert(0);
          break;
      }
    }
  }
}

void DCT::idxstAndIdctProcess(float** sequence)
{
  resetBufSequence();

  // std::clock_t pre_clock_start, pre_clock_end;
  // pre_clock_start = clock();
  idxstAndIdctPreprocess(sequence, _buf_sequence);
  // pre_clock_end = clock();
  // std::cout << "IdxstAndIdct preprocess used time: " << double(pre_clock_end - pre_clock_start) / CLOCKS_PER_SEC << " s" << std::endl;

  // std::clock_t clock_start, clock_end;
  // clock_start = clock();
  rdft2dsort(_bin_cnt_y, _bin_cnt_x, -1, _buf_sequence);
  rdft2d(_bin_cnt_y, _bin_cnt_x, -1, _buf_sequence, NULL, (int*) &_work_area[0], (float*) &_cs_table[0]);
  resetForInverseMatrix(_buf_sequence);
  // clock_end = clock();
  // std::cout << "sequence size: " << _bin_cnt_y * _bin_cnt_x << std::endl;
  // std::cout << "IdxstAndIdct rfft2d used time: " << double(clock_end - clock_start) / CLOCKS_PER_SEC << " s" << std::endl;

  // std::clock_t post_clock_start, post_clock_end;
  // post_clock_start = clock();
  idxstAndIdctPostprocess(_buf_sequence, sequence);
  // post_clock_end = clock();
  // std::cout << "IdxstAndIdct postprocess used time: " << double(post_clock_end - post_clock_start) / CLOCKS_PER_SEC << " s" << std::endl;
}

void DCT::idxstAndIdctPreprocess(float** sequence, float** buf_sequence)
{
  resetComplex2DList();
  int half_cnt_y = _bin_cnt_y / 2;
  int half_cnt_x = _bin_cnt_x / 2;

#pragma omp parallel for num_threads(_thread_nums)
  for (int h_id = 0; h_id < half_cnt_y; h_id++) {
    for (int w_id = 0; w_id < half_cnt_x; w_id++) {
      int condition = ((h_id != 0) << 1) | (w_id != 0);
      switch (condition) {
        case 0: {
          float tmp_1;
          std::complex<float> tmp_complex_up, tmp_complex_1, tmp_complex_2;

          _complex_2d_list[0][0] = {0.0f, 0.0f};
          _complex_2d_list[0][half_cnt_x] = {0.0f, 0.0f};

          tmp_1 = sequence[half_cnt_y][0];
          tmp_complex_up = {tmp_1, tmp_1};
          tmp_complex_1 = obtainComplexMultiply(_expk_y[half_cnt_y], tmp_complex_up);
          _complex_2d_list[half_cnt_y][0] = obtainComplexConjugate(tmp_complex_1);

          tmp_1 = sequence[half_cnt_y][half_cnt_x];
          tmp_complex_up = {0.0f, 2 * tmp_1};
          tmp_complex_1 = obtainComplexMultiply(_expk_y[half_cnt_y], _expk_x[half_cnt_x]);
          tmp_complex_2 = obtainComplexMultiply(tmp_complex_1, tmp_complex_up);
          _complex_2d_list[half_cnt_y][half_cnt_x] = obtainComplexConjugate(tmp_complex_2);
          break;
        }
        case 1: {
          _complex_2d_list[0][w_id] = {0.0f, 0.0f};

          std::complex<float> tmp_complex_up, tmp_complex_1, tmp_complex_2;
          float tmp_1 = sequence[half_cnt_y][w_id];
          float tmp_2 = sequence[half_cnt_y][_bin_cnt_x - w_id];
          tmp_complex_up = {tmp_1 - tmp_2, tmp_1 + tmp_2};

          tmp_complex_1 = obtainComplexMultiply(_expk_y[half_cnt_y], _expk_x[w_id]);
          tmp_complex_2 = obtainComplexMultiply(tmp_complex_1, tmp_complex_up);
          _complex_2d_list[half_cnt_y][w_id] = obtainComplexConjugate(tmp_complex_2);
          break;
        }
        case 2: {
          float tmp_1, tmp_3;
          std::complex<float> tmp_complex_up, tmp_complex_down, tmp_complex_1, tmp_complex_2;

          tmp_1 = sequence[_bin_cnt_y - h_id][0];
          tmp_3 = sequence[h_id][0];
          tmp_complex_up = {tmp_1, tmp_3};
          tmp_complex_down = {tmp_3, tmp_1};

          tmp_complex_1 = obtainComplexMultiply(_expk_y[h_id], tmp_complex_up);
          _complex_2d_list[h_id][0] = obtainComplexConjugate(tmp_complex_1);
          tmp_complex_2 = obtainComplexMultiply(_expk_y[_bin_cnt_y - h_id], tmp_complex_down);
          _complex_2d_list[_bin_cnt_y - h_id][0] = obtainComplexConjugate(tmp_complex_2);

          tmp_1 = sequence[_bin_cnt_y - h_id][half_cnt_x];
          tmp_3 = sequence[h_id][half_cnt_x];
          tmp_complex_up = {tmp_1 - tmp_3, tmp_3 + tmp_1};
          tmp_complex_down = {tmp_3 - tmp_1, tmp_1 + tmp_3};

          tmp_complex_1 = obtainComplexMultiply(_expk_y[h_id], _expk_x[half_cnt_x]);
          tmp_complex_2 = obtainComplexMultiply(tmp_complex_1, tmp_complex_up);
          _complex_2d_list[h_id][half_cnt_x] = obtainComplexConjugate(tmp_complex_2);

          tmp_complex_1 = obtainComplexMultiply(_expk_y[_bin_cnt_y - h_id], _expk_x[half_cnt_x]);
          tmp_complex_2 = obtainComplexMultiply(tmp_complex_1, tmp_complex_down);
          _complex_2d_list[_bin_cnt_y - h_id][half_cnt_x] = obtainComplexConjugate(tmp_complex_2);
          break;
        }
        case 3: {
          float tmp_1 = sequence[_bin_cnt_y - h_id][w_id];
          float tmp_2 = sequence[_bin_cnt_y - h_id][_bin_cnt_x - w_id];
          float tmp_3 = sequence[h_id][w_id];
          float tmp_4 = sequence[h_id][_bin_cnt_x - w_id];

          std::complex<float> tmp_complex_up, tmp_complex_down, tmp_complex_1, tmp_complex_2;
          tmp_complex_up = {tmp_1 - tmp_4, tmp_3 + tmp_2};
          tmp_complex_down = {tmp_3 - tmp_2, tmp_1 + tmp_4};

          tmp_complex_1 = obtainComplexMultiply(_expk_y[h_id], _expk_x[w_id]);
          tmp_complex_2 = obtainComplexMultiply(tmp_complex_1, tmp_complex_up);
          _complex_2d_list[h_id][w_id] = obtainComplexConjugate(tmp_complex_2);

          tmp_complex_1 = obtainComplexMultiply(_expk_y[_bin_cnt_y - h_id], _expk_x[w_id]);
          tmp_complex_2 = obtainComplexMultiply(tmp_complex_1, tmp_complex_down);
          _complex_2d_list[_bin_cnt_y - h_id][w_id] = obtainComplexConjugate(tmp_complex_2);
          break;
        }

        default:
          assert(0);
          break;
      }
    }
  }
  convertComplex2DListToBufSequence();
}

void DCT::idxstAndIdctPostprocess(float** buf_sequence, float** sequence)
{
  int bin_cnt_yx = _bin_cnt_y * _bin_cnt_x;
#pragma omp parallel for num_threads(_thread_nums)
  for (int h_id = 0; h_id < _bin_cnt_y; h_id++) {
    for (int w_id = 0; w_id < _bin_cnt_x; w_id++) {
      int condition = ((h_id < _bin_cnt_y / 2) << 1) | (w_id < _bin_cnt_x / 2);
      int index_1, index_2;
      switch (condition) {
        case 0: {
          index_1 = ((_bin_cnt_y - h_id) << 1) - 1;
          index_2 = ((_bin_cnt_x - w_id) << 1) - 1;
          sequence[index_1][index_2] = -buf_sequence[h_id][w_id] * bin_cnt_yx;
          break;
        }
        case 1: {
          index_1 = ((_bin_cnt_y - h_id) << 1) - 1;
          index_2 = (w_id << 1);
          sequence[index_1][index_2] = -buf_sequence[h_id][w_id] * bin_cnt_yx;
          break;
        }
        case 2: {
          index_1 = (h_id << 1);
          index_2 = ((_bin_cnt_x - w_id) << 1) - 1;
          sequence[index_1][index_2] = buf_sequence[h_id][w_id] * bin_cnt_yx;
          break;
        }
        case 3: {
          index_1 = (h_id << 1);
          index_2 = (w_id << 1);
          sequence[index_1][index_2] = buf_sequence[h_id][w_id] * bin_cnt_yx;
          break;
        }
        default:
          assert(0);
          break;
      }
    }
  }
}

void DCT::resetForInverseMatrix(float** sequence)
{
#pragma omp parallel for num_threads(_thread_nums)
  for (int j1 = 0; j1 < _bin_cnt_y; j1++) {
    for (int j2 = 0; j2 < _bin_cnt_x; j2++) {
      sequence[j1][j2] *= 2.0 / _bin_cnt_y / _bin_cnt_x;
    }
  }
}

int DCT::obtainIndex(int h_id, int w_id, int column_cnt)
{
  return (h_id * column_cnt + w_id);
}

float DCT::obtainRealPartMultiply(std::complex<float>& x, std::complex<float>& y)
{
  return x.real() * y.real() - x.imag() * y.imag();
}

float DCT::obtainImaginaryPartMultiply(std::complex<float>& x, std::complex<float>& y)
{
  return x.real() * y.imag() + x.imag() * y.real();
}

std::complex<float> DCT::obtainComplexAdd(std::complex<float>& x, std::complex<float>& y)
{
  std::complex<float> res = {x.real() + y.real(), x.imag() + y.imag()};
  return res;
}

std::complex<float> DCT::obtainComplexSubtract(std::complex<float>& x, std::complex<float>& y)
{
  std::complex<float> res = {x.real() - y.real(), x.imag() - y.imag()};
  return res;
}

std::complex<float> DCT::obtainComplexMultiply(std::complex<float>& x, std::complex<float>& y)
{
  std::complex<float> res;
  res.real(x.real() * y.real() - x.imag() * y.imag());
  res.imag(x.real() * y.imag() + x.imag() * y.real());
  return res;
}

std::complex<float> DCT::obtainComplexConjugate(std::complex<float>& x)
{
  std::complex<float> res = {x.real(), -x.imag()};
  return res;
}

void DCT::resetBufSequence()
{
#pragma omp parallel for num_threads(_thread_nums)
  for (int i = 0; i < _bin_cnt_y; i++) {
    for (int j = 0; j < _bin_cnt_x + 2; j++) {
      _buf_sequence[i][j] = 0;
    }
  }
}
void DCT::resetComplex2DList()
{
#pragma omp parallel for num_threads(_thread_nums)
  for (int i = 0; i < _bin_cnt_y; i++) {
    for (int j = 0; j < _bin_cnt_x / 2 + 1; j++) {
      _complex_2d_list[i][j] = {0, 0};
    }
  }
}

void DCT::convertBufSequenceToComplex2DList()
{
#pragma omp parallel for num_threads(_thread_nums)
  for (int i = 0; i < _bin_cnt_y; i++) {
    for (int j = 0; (j < _bin_cnt_x + 2) && (j + 1 < _bin_cnt_x + 2); j += 2) {
      int index = j / 2;
      _complex_2d_list[i][index] = {_buf_sequence[i][j], -_buf_sequence[i][j + 1]};
    }
  }
}

void DCT::convertComplex2DListToBufSequence()
{
  int half_cnt_x = _bin_cnt_x / 2;

#pragma omp parallel for num_threads(_thread_nums)
  for (int i = 0; i < _bin_cnt_y; i++) {
    for (int j = 0; j < half_cnt_x + 1; j++) {
      std::complex<float> cur_complex = _complex_2d_list[i][j];
      int index = 2 * j;
      _buf_sequence[i][index] = cur_complex.real();
      _buf_sequence[i][index + 1] = -cur_complex.imag();
    }
  }
}

}  // namespace ipl