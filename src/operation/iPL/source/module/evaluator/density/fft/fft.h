/*
 * @Author: S.J Chen
 * @Date: 2022-03-09 19:53:30
 * @LastEditTime: 2022-03-09 19:54:29
 * @LastEditors: S.J Chen
 * @Description: This describe fft in placer
 * @FilePath: /iEDA/src/iPL/src/evaluator/density/fft/fft.h
 * Contact : https://github.com/sjchanson
 */

#ifndef FFT_H
#define FFT_H

#include <vector>

class FFT
{
 public:
  FFT();
  FFT(int binCnt_x, int binCnt_y, int binSize_x, int binSize_y);
  ~FFT();

  void init();

  void updateDensity(int x, int y, float density);

  void doFFT();

  // return func
  std::pair<float, float> get_electro_force(int x, int y);
  float                   get_electro_phi(int x, int y);

 private:
  // 2D array; width: binCntX_, height: binCntY_
  // No hope to use Vector at this moment...
  float** _bin_density;
  float** _electro_phi;
  float** _electroForce_x;
  float** _electroForce_y;

  // cos/sin table (prev: w_2d)
  // length:  max(binCntX, binCntY) * 3 / 2
  std::vector<float> _cs_table;

  // wx. length:  binCntX_
  std::vector<float> _wx;
  std::vector<float> _wx_square;

  // wy. length:  binCntY_
  std::vector<float> _wy;
  std::vector<float> _wy_square;

  // work area for bit reversal (prev: ip)
  // length: round(sqrt( max(binCntX_, binCntY_) )) + 2
  std::vector<int> _work_area;

  int _binCnt_x;
  int _binCnt_y;
  int _binSize_x;
  int _binSize_y;
};

//
// The following FFT library came from
// http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html
//
//
/// 1D FFT ////////////////////////////////////////////////////////////////
void cdft(int n, int isgn, float* a, int* ip, float* w);
void ddct(int n, int isgn, float* a, int* ip, float* w);
void ddst(int n, int isgn, float* a, int* ip, float* w);

/// 2D FFT ////////////////////////////////////////////////////////////////
void cdft2d(int, int, int, float**, float*, int*, float*);
void rdft2d(int, int, int, float**, float*, int*, float*);
void ddct2d(int, int, int, float**, float*, int*, float*);
void ddst2d(int, int, int, float**, float*, int*, float*);
void ddsct2d(int n1, int n2, int isgn, float** a, float* t, int* ip, float* w);
void ddcst2d(int n1, int n2, int isgn, float** a, float* t, int* ip, float* w);

/// 3D FFT ////////////////////////////////////////////////////////////////
void cdft3d(int, int, int, int, float***, float*, int*, float*);
void rdft3d(int, int, int, int, float***, float*, int*, float*);
void ddct3d(int, int, int, int, float***, float*, int*, float*);
void ddst3d(int, int, int, int, float***, float*, int*, float*);
void ddscct3d(int, int, int, int isgn, float***, float*, int*, float*);
void ddcsct3d(int, int, int, int isgn, float***, float*, int*, float*);
void ddccst3d(int, int, int, int isgn, float***, float*, int*, float*);

#endif  // FFT_H