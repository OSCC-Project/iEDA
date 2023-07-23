#include "DensityModel.hh"

#include <omp.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <ranges>
#include <tuple>
#include <vector>

#include "dct_process/DCT.hh"

namespace ipl {
// NOLINTBEGIN

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
vector<Coordinate<double>> getEndpoint(double lx, double ly, double w, double h)
{
  vector<Coordinate<double>> endpoint(4);
  endpoint[0] = {lx, ly};
  endpoint[1] = {lx + w, ly};
  endpoint[2] = {lx + w, ly + h};
  endpoint[3] = {lx, ly + h};
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
  _sum_macro_area = _width.dot(_height);
  double ar = core_h / core_w;
  _num_bins_x = std::pow(2, static_cast<int>(std::ceil(std::log2(std::sqrt(10 * _num_var / ar)))));
  _num_bins_y = std::pow(2, static_cast<int>(std::ceil(std::log2(std::sqrt(10 * _num_var * ar)))));
  // _num_bins_x = 512;
  // _num_bins_y = 512;
  _bin_size_x = std::round(core_w / _num_bins_x);
  _bin_size_y = std::round(core_h / _num_bins_y);
  _dct = new DCT(_num_bins_x, _num_bins_y, _bin_size_x, _bin_size_y);
  // _dct->set_thread_nums(_num_threads);
  // _dct->set_thread_nums(8);
  _area_bin = static_cast<float>(_bin_size_x * _bin_size_y);
  _ploygonlist.resize(_num_var);
  _utilization.resize(_num_var);
}

void DensityModel::evaluate(const Mat& variable, Mat& grad, double& cost) const
{
  // std::cout << "num bins: " << _num_bins_x << std::endl;
  auto start = std::chrono::steady_clock::now();
  updateDensityMap(variable.col(0), variable.col(1), variable.col(2));
  auto end = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  // std::cout << "bindensity times: " << elapsed.count() << " ms" << std::endl;
  cost = getOverflow();
  start = std::chrono::steady_clock::now();
  _dct->doDCT(false);
  end = std::chrono::steady_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  // std::cout << "fft times: " << elapsed.count() << " ms\n" << std::endl;
  geteDensityGradient(grad);
}

void DensityModel::updateDensityMap(const Vec& x, const Vec& y, const Vec& r) const
{
  float** density_map = _dct->get_density_2d_ptr();
#pragma omp parallel for num_threads(_num_threads)
  for (int i = 0; i < _num_bins_x; i++) {
    std::fill_n(density_map[i], _num_bins_y, 0.0f);
  }
#pragma omp parallel for num_threads(_num_threads)
  for (int i = 0; i < _num_var; i++) {
    Ploygon<double> ploygon(getEndpoint(x(i), y(i), _width(i), _height(i), r(i)));
    const auto& [set, map] = ploygonDraing(ploygon);
    int count = 0;

    for (const auto& [x, col] : map)
      count += (col.get_y() - col.get_x() + 1);
    float overlap = _width(i) * _height(i) / count / _area_bin;
    _utilization[i] = overlap;
    for (const auto& [x, col] : map) {
      for (int y = col.get_x(); y <= col.get_y(); y++) {
#pragma omp atomic
        density_map[x][y] += overlap;
      }
    }
    _ploygonlist[i] = std::move(map);
  }
}
std::pair<PointUnSet<int>, PointUnMap<int>> DensityModel::rectangleDraing(double cx, double cy, double w, double h, double r) const
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
    x1 = std::clamp(x1, 0, _num_bins_x - 1);
    x2 = std::clamp(x2, 0, _num_bins_x - 1);
    y1 = std::clamp(y1, 0, _num_bins_y - 1);
    y2 = std::clamp(y2, 0, _num_bins_y - 1);
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

std::pair<PointUnSet<int>, PointUnMap<int>> DensityModel::ploygonDraing(const Ploygon<double>& ploygon) const
{
  PointUnSet<int> set;
  const auto& endpoint = ploygon.getCoordinates();
  for (size_t i = 0; i < 4; i++) {
    const auto& a = endpoint[i % 4];
    const auto& b = endpoint[(i + 1) % 4];
    int x1 = std::floor(a.get_x() / _bin_size_x);
    int y1 = std::floor(a.get_y() / _bin_size_y);
    int x2 = std::floor(b.get_x() / _bin_size_x);
    int y2 = std::floor(b.get_y() / _bin_size_y);
    x1 = std::clamp(x1, 0, _num_bins_x - 1);
    x2 = std::clamp(x2, 0, _num_bins_x - 1);
    y1 = std::clamp(y1, 0, _num_bins_y - 1);
    y2 = std::clamp(y2, 0, _num_bins_y - 1);
    lineDraing({x1, y1}, {x2, y2}, set);
  }
  PointUnMap<int> map;
  for (const auto& point : set) {
    if (!map.contains(point.get_x())) {
      map[point.get_x()] = {point.get_y(), point.get_y()};
    } else {
      const auto& col = map[point.get_x()];
      map[point.get_x()] = {std::min(col.get_x(), point.get_y()), std::max(col.get_y(), point.get_y())};
    }
  }
  return {std::move(set), std::move(map)};
}

void DensityModel::lineDraing(const Coordinate<int>& a, const Coordinate<int>& b, PointUnSet<int>& set) const
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
    // if (x < 0 || x >= _num_bins_x) {
    //   std::cout << x;
    // }
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

double DensityModel::getOverflow() const
{
  float** bin_density = _dct->get_density_2d_ptr();
  float sum = 0;
#pragma omp parallel for num_threads(_num_threads) reduction(+ : sum)
  for (int i = 0; i < _num_bins_x; i++) {
    for (int j = 0; j < _num_bins_y; j++) {
      sum += std::max(0.0f, bin_density[i][j] - 1.0f);
    }
  }
  return sum * _bin_size_x * _bin_size_y / _sum_macro_area;
}

void DensityModel::geteDensityGradient(Mat& grad) const
{
  float** xi_x = _dct->get_electro_x_2d_ptr();
  float** xi_y = _dct->get_electro_y_2d_ptr();
#pragma omp parallel for num_threads(_num_threads)
  for (int i = 0; i < _num_var; i++) {
    float util = _utilization[i];
    grad(i, 0) = 0;
    grad(i, 1) = 0;
    for (const auto& [x, col] : _ploygonlist[i]) {
      for (int y = col.get_x(); y <= col.get_y(); y++) {
        grad(i, 0) -= xi_y[x][y] * util;
        grad(i, 1) -= xi_x[x][y] * util;
      }
    }
  }
}

// NOLINTEND
}  // namespace ipl