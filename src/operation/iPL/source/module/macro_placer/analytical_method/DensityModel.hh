/**
 * @file DensityModel.hh
 * @author Fuxing Huang (fxxhuang@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef IPL_IMP_DENSITYMODEL_HH
#define IPL_IMP_DENSITYMODEL_HH
#include <omp.h>

#include <Eigen/Dense>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "utility/Geometry.hh"
using std::vector;
using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
namespace ipl {
template <typename T>
using PointUnSet = std::unordered_set<Coordinate<T>, CoordinateHash<T>>;
template <typename T>
using PointUnMap = std::unordered_map<int, Coordinate<T>>;
class DCT;
class DensityModel
{
 public:
  DensityModel();
  ~DensityModel();
  void setConstant(const Vec& width, const Vec& height, double core_w, double core_h);
  void evaluate(const Mat& variable, Mat& grad, double& cost) const;
  void updateDensityMap(const Vec& x, const Vec& y, const Vec& r);
  std::pair<PointUnSet<int>, PointUnMap<int>> rectangleDraing(double cx, double cy, double w, double h, double r = 0);
  void lineDraing(const Coordinate<int>& a, const Coordinate<int>& b, PointUnSet<int>& set);
  void polygonClipArea();
  void updateBinDensity(int x, int y, double density);
  void getEDensityGradient();

 private:
  DCT* _dct;
  Vec _width;
  Vec _height;
  int _num_var;
  int _core_w;
  int _core_h;
  int _num_bins_x;
  int _num_bins_y;
  int _bin_size_x;
  int _bin_size_y;
  float _area_bin;
};

}  // namespace ipl

#endif