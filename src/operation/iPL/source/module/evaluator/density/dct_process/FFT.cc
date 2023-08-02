#include "FFT.hh"

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "omp.h"

#define FFT_PI 3.141592653589793238462L

namespace ipl {

FFT::FFT() : _binCnt_x(0), _binCnt_y(0), _binSize_x(0), _binSize_y(0), _thread_nums(1)
{
}

FFT::FFT(int binCnt_x, int binCnt_y, int binSize_x, int binSize_y)
    : _binCnt_x(binCnt_x), _binCnt_y(binCnt_y), _binSize_x(binSize_x), _binSize_y(binSize_y)
{
  init();
}

FFT::~FFT()
{
  using std::vector;
  for (int i = 0; i < _binCnt_x; i++) {
    delete (_bin_density[i]);
    delete (_electro_phi[i]);
    delete (_electroForce_x[i]);
    delete (_electroForce_y[i]);
  }
  delete (_bin_density);
  delete (_electro_phi);
  delete (_electroForce_x);
  delete (_electroForce_y);

  _cs_table.clear();
  _wx.clear();
  _wx_square.clear();
  _wy.clear();
  _wy_square.clear();

  _work_area.clear();
}

void FFT::init()
{
  _bin_density = new float*[_binCnt_x];
  _electro_phi = new float*[_binCnt_x];
  _electroForce_x = new float*[_binCnt_x];
  _electroForce_y = new float*[_binCnt_x];

  for (int i = 0; i < _binCnt_x; i++) {
    _bin_density[i] = new float[_binCnt_y];
    _electro_phi[i] = new float[_binCnt_y];
    _electroForce_x[i] = new float[_binCnt_y];
    _electroForce_y[i] = new float[_binCnt_y];

    for (int j = 0; j < _binCnt_y; j++) {
      _bin_density[i][j] = _electro_phi[i][j] = _electroForce_x[i][j] = _electroForce_y[i][j] = 0.0f;
    }
  }

  _cs_table.resize(std::max(_binCnt_x, _binCnt_y) * 3 / 2, 0);

  _wx.resize(_binCnt_x, 0);
  _wx_square.resize(_binCnt_x, 0);
  _wy.resize(_binCnt_y, 0);
  _wy_square.resize(_binCnt_y, 0);

  _work_area.resize(round(sqrt(std::max(_binCnt_x, _binCnt_y))) + 2, 0);

  for (int i = 0; i < _binCnt_x; i++) {
    _wx[i] = FFT_PI * static_cast<float>(i) / static_cast<float>(_binCnt_x);
    _wx_square[i] = _wx[i] * _wx[i];
  }

  for (int i = 0; i < _binCnt_y; i++) {
    _wy[i] = FFT_PI * static_cast<float>(i) / static_cast<float>(_binCnt_y)
             * (static_cast<float>(_binSize_y) / static_cast<float>(_binSize_x));
    _wy_square[i] = _wy[i] * _wy[i];
  }
}

void FFT::updateDensity(int x, int y, float density)
{
  _bin_density[x][y] = density;
}

void FFT::doFFT(bool is_calculate_phi)
{
  ddct2d(_binCnt_x, _binCnt_y, -1, _bin_density, NULL, (int*) &_work_area[0], (float*) &_cs_table[0]);

#pragma omp parallel for num_threads(_thread_nums)
  for (int i = 0; i < _binCnt_x; i++) {
    _bin_density[i][0] *= 0.5;
  }

#pragma omp parallel for num_threads(_thread_nums)
  for (int i = 0; i < _binCnt_y; i++) {
    _bin_density[0][i] *= 0.5;
  }

#pragma omp parallel for num_threads(_thread_nums)
  for (int i = 0; i < _binCnt_x; i++) {
    for (int j = 0; j < _binCnt_y; j++) {
      _bin_density[i][j] *= 4.0 / _binCnt_x / _binCnt_y;
    }
  }

#pragma omp parallel for num_threads(_thread_nums)
  for (int i = 0; i < _binCnt_x; i++) {
    float wu = _wx[i];
    float wu2 = _wx_square[i];

    for (int j = 0; j < _binCnt_y; j++) {
      float wv = _wy[j];
      float wv2 = _wy_square[j];

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
        electro_x = auv_by_wu2_plus_wv2 * wu;
        electro_y = auv_by_wu2_plus_wv2 * wv;
      }
      _electroForce_x[i][j] = electro_x;
      _electroForce_y[i][j] = electro_y;
      _electro_phi[i][j] = phi;
    }
  }
  // Inverse DCT
  ddsct2d(_binCnt_x, _binCnt_y, 1, _electroForce_x, NULL, (int*) &_work_area[0], (float*) &_cs_table[0]);
  ddcst2d(_binCnt_x, _binCnt_y, 1, _electroForce_y, NULL, (int*) &_work_area[0], (float*) &_cs_table[0]);

  if (is_calculate_phi) {
    ddct2d(_binCnt_x, _binCnt_y, 1, _electro_phi, NULL, (int*) &_work_area[0], (float*) &_cs_table[0]);
  }
}

std::pair<float, float> FFT::get_electro_force(int x, int y)
{
  return std::make_pair(_electroForce_x[x][y], _electroForce_y[x][y]);
}

float FFT::get_electro_phi(int x, int y)
{
  return _electro_phi[x][y];
}

}  // namespace ipl