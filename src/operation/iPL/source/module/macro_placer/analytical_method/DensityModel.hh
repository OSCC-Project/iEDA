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

#include <Eigen/Dense>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "utility/Geometry.hh"

using std::vector;
using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
namespace std {
template <typename T>
struct hash<ipl::Coordinate<T>>
{
  std::size_t operator()(const ipl::Coordinate<T>& c) const noexcept
  {
    std::size_t h1 = std::hash<T>{}(c.get_x());
    std::size_t h2 = std::hash<T>{}(c.get_y());
    return h1 ^ (h2 << 1);
  }
};
}  // namespace std
namespace ipl {
template <typename T>
using PointUnSet = std::unordered_set<Coordinate<T>>;
// template <typename T>
// using PointUnSet = std::set<Coordinate<T>>;
template <typename T>
using PointUnMap = std::unordered_map<int, Coordinate<T>>;
class DCT;
class DensityModel
{
 public:
  DensityModel();
  ~DensityModel();
  void setConstant(const Vec& width, const Vec& height, double core_w, double core_h);
  void setThreads(int n) { _num_threads = n; }
  void evaluate(const Mat& variable, Mat& grad, double& cost) const;
  void updateDensityMap(const Vec& x, const Vec& y, const Vec& r) const;
  std::pair<PointUnSet<int>, PointUnMap<int>> rectangleDraing(double cx, double cy, double w, double h, double r = 0) const;
  std::pair<PointUnSet<int>, PointUnMap<int>> ploygonDraing(const Ploygon<double>& ploygon) const;
  void lineDraing(const Coordinate<int>& a, const Coordinate<int>& b, PointUnSet<int>& set) const;
  double getOverflow() const;
  void geteDensityGradient(Mat& grad) const;

 private:
  DCT* _dct;
  Vec _width;
  Vec _height;
  double _sum_macro_area;
  int _num_var;
  int _core_w;
  int _core_h;
  int _num_bins_x;
  int _num_bins_y;
  int _bin_size_x;
  int _bin_size_y;
  int _num_threads;
  float _area_bin;
  mutable vector<PointUnMap<int>> _ploygonlist;
  mutable vector<float> _utilization;
};

}  // namespace ipl

#endif