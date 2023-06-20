#include "DensityModel.hh"

#include <cmath>
#include <vector>

#include "dct_process/DCT.hh"
namespace ipl {

vector<Coordinate<double>> getEndpoint(double cx, double cy, double w, double h, double r)
{
  vector<Coordinate<double>> endpoint(4);
  double x_off = 0.5 * w;
  double y_off = 0.5 * h;
  double cos = std::cos(r);
  double sin = std::sin(r);
  endpoint[0] = {cx - x_off * cos + y_off * sin, cy - x_off * sin - y_off * cos};
  endpoint[1] = {cx + x_off * cos + y_off * sin, cy + x_off * sin - y_off * cos};
  endpoint[2] = {cx + x_off * cos - y_off * sin, cy + x_off * sin + y_off * cos};
  endpoint[3] = {cx - x_off * cos - y_off * sin, cy - x_off * sin + y_off * cos};
  return endpoint;
}

DensityModel::DensityModel()
{
}

DensityModel::~DensityModel()
{
  delete _dct;
}

void DensityModel::setConstant(const Vec& width, const Vec& height, double core_w, double core_h)
{
  _width = width;
  _height = height;
  _num_var = width.rows();
  double ar = std::floor(core_h / core_w);
  _num_bins_x = std::pow(2, static_cast<int>(std::ceil(std::log2(std::sqrt(9 * _num_var / ar)))));
  _num_bins_y = std::pow(2, static_cast<int>(std::ceil(std::log2(std::sqrt(9 * _num_var * ar)))));
  _bin_size_x = std::round(core_w / _num_bins_x);
  _bin_size_y = std::round(core_h / _num_bins_y);
  _dct = new DCT(_num_bins_x, _num_bins_y, _bin_size_x, _bin_size_y);
  _area_bin = static_cast<float>(_bin_size_x * _bin_size_y);
}

void DensityModel::evaluate(const Mat& variable, Mat& grad, double& cost) const
{
}

void DensityModel::updateDensityMap(const Vec& x, const Vec& y, const Vec& r)
{
  float** density_map = _dct->get_density_2d_ptr();
  for (int i = 0; i < _num_bins_x; i++) {
    std::fill_n(density_map[i], _num_bins_y, 0);
  }
  for (int i = 0; i < _num_var; i++) {
    const auto& intersect_bins = rectangleDraing(x(i), y(i), _width(i), _height(i), r(i));
    const auto& set = intersect_bins.first;
    const auto& map = intersect_bins.second;
    for (const auto& col : map) {
      int x = col.first;
      for (int y = col.second.get_x(); y < col.second.get_y(); y++) {
        if (set.contains({x, y})) {
          continue;
        }
        density_map[x][y] += _area_bin;
      }
    }
  }
}
std::pair<PointUnSet<int>, PointUnMap<int>> DensityModel::rectangleDraing(double cx, double cy, double w, double h, double r)
{
  PointUnSet<int> set;
  const auto& endpoint = getEndpoint(cx, cy, w, h, r);
  for (size_t i = 0; i < 4; i++) {
    const auto& a = endpoint[i % 4];
    const auto& b = endpoint[(i + 1) % 4];
    int x1 = std::floor(a.get_x() / _bin_size_x);
    int y1 = std::floor(a.get_y() / _bin_size_y);
    int x2 = std::floor(b.get_x() / _bin_size_x);
    int y2 = std::floor(b.get_y() / _bin_size_y);
    lineDraing({x1, y1}, {x2, y2}, set);
  }
  PointUnMap<int> map;
  for (const auto& point : set) {
    if (map.contains(point.get_x())) {
      map[point.get_x()] = {point.get_y(), point.get_y()};
    } else {
      const auto& col = map[point.get_x()];
      map[point.get_x()] = {std::min(col.get_x(), point.get_y()), std::max(col.get_y(), point.get_y())};
    }
  }
  return {set, map};
}

void DensityModel::lineDraing(const Coordinate<int>& a, const Coordinate<int>& b, PointUnSet<int>& set)
{
  int lx = a.get_x();
  int ly = a.get_y();
  int ux = b.get_x();
  int uy = b.get_y();
  bool steep = abs(uy - ly) > abs(ux - lx);
  if (steep) {
    std::swap(lx, ly);
    std::swap(ux, uy);
  }
  if (lx > ux) {
    std::swap(lx, ux);
    std::swap(ly, uy);
  }
  int deltax = ux - lx;
  int deltay = abs(uy - ly);
  int error = deltax / 2;
  int ystep;
  int y = ly;
  if (ly < uy) {
    ystep = 1;
  } else {
    ystep = -1;
  }
  for (int x = lx; x <= ux; x++) {
    if (steep) {
      set.emplace(y, x);
    } else {
      set.emplace(x, y);
    }
    error -= deltay;
    if (error < 0) {
      y += ystep;
      error += deltax;
    }
  }
}

void DensityModel::updateBinDensity(int x, int y, double density)
{
  _dct->updateDensity(x, y, static_cast<float>(density));
}
}  // namespace ipl