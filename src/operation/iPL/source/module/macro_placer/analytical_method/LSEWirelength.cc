#include "LSEWirelength.hh"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
namespace ipl {
void LSEWirelength::setConstant(const vector<Triplet<double>>& moveable_x_offset, const vector<Triplet<double>>& moveable_y_offset,
                                const vector<Triplet<double>>& fixed_x_location, const vector<Triplet<double>>& fixed_y_location)
{
  _orig_x_offset.setFromTriplets(moveable_x_offset.begin(), moveable_x_offset.end());
  _orig_y_offset.setFromTriplets(moveable_y_offset.begin(), moveable_y_offset.end());
  int num_fix = 0;
  for (const auto& trip : fixed_x_location) {
    num_fix = std::max(trip.row(), num_fix);
  }
  _fix_x = SpMat(num_fix + 1, _num_edges);
  _fix_y = SpMat(num_fix + 1, _num_edges);

  _fix_x.setFromTriplets(fixed_x_location.begin(), fixed_x_location.end());
  _fix_y.setFromTriplets(fixed_y_location.begin(), fixed_y_location.end());

  _exp_pos_x = _orig_x_offset;
  _exp_neg_x = _orig_x_offset;
  _exp_pos_y = _orig_x_offset;
  _exp_neg_y = _orig_x_offset;
  _x_offset = _orig_x_offset;
  _y_offset = _orig_x_offset;
}

void LSEWirelength::updatePinLocation(const Vec& x, const Vec& y, const Vec& r, const double& gamma) const
{
  auto exp_div = [gamma](double val) { return std::exp(val / gamma); };
  const auto& rcos = r.array().cos();
  const auto& rsin = r.array().sin();
  auto get_pin_off = [&](double& x_off, double& y_off, int id) {
    double x_temp = x_off;
    double y_temp = y_off;
    x_off = x_temp * rcos(id) - y_temp * rsin(id);
    y_off = x_temp * rsin(id) + y_temp * rcos(id);
  };

#pragma omp parallel for num_threads(48)
  for (int k = 0; k < _exp_pos_x.outerSize(); ++k) {
    double x_min = std::numeric_limits<double>::max();
    double x_max = std::numeric_limits<double>::lowest();
    double y_min = std::numeric_limits<double>::max();
    double y_max = std::numeric_limits<double>::lowest();

    auto it_fx = SpMat::InnerIterator(_fix_x, k);
    auto it_fy = SpMat::InnerIterator(_fix_y, k);

    while (it_fx) {
      assert(it_fx.row() == it_fy.row());
      x_min = std::min(it_fx.value(), x_min);
      x_max = std::max(it_fx.value(), x_max);
      y_min = std::min(it_fy.value(), y_min);
      y_max = std::max(it_fy.value(), y_max);

      ++it_fx;
      ++it_fy;
    }

    auto it_oxo = SpMat::InnerIterator(_orig_x_offset, k);
    auto it_oyo = SpMat::InnerIterator(_orig_y_offset, k);
    auto it_xo = SpMat::InnerIterator(_x_offset, k);
    auto it_yo = SpMat::InnerIterator(_y_offset, k);

    while (it_oxo) {
      int row = it_oxo.row();

      assert(row == it_oyo.row());
      assert(row == it_xo.row());
      assert(row == it_yo.row());

      double x_off = it_oxo.value();
      double y_off = it_oyo.value();
      get_pin_off(x_off, y_off, row);
      it_xo.valueRef() = x_off;
      it_yo.valueRef() = y_off;

      x_min = std::min(x(row) + x_off, x_min);
      x_max = std::max(x(row) + x_off, x_max);
      y_min = std::min(y(row) + y_off, y_min);
      y_max = std::max(y(row) + y_off, y_max);

      ++it_oxo;
      ++it_oyo;
      ++it_xo;
      ++it_yo;
    }

    _hpwl(k) = x_max - x_min + y_max - y_min;

    it_xo = SpMat::InnerIterator(_x_offset, k);
    it_yo = SpMat::InnerIterator(_y_offset, k);
    auto it_px = SpMat::InnerIterator(_exp_pos_x, k);
    auto it_nx = SpMat::InnerIterator(_exp_neg_x, k);
    auto it_py = SpMat::InnerIterator(_exp_pos_y, k);
    auto it_ny = SpMat::InnerIterator(_exp_neg_y, k);

    double sum_exp_pos_x_k = 0.0;
    double sum_exp_neg_x_k = 0.0;
    double sum_exp_pos_y_k = 0.0;
    double sum_exp_neg_y_k = 0.0;

    it_fx = SpMat::InnerIterator(_fix_x, k);
    it_fy = SpMat::InnerIterator(_fix_y, k);

    while (it_fx) {
      assert(it_fx.row() == it_fy.row());
      sum_exp_pos_x_k += exp_div(it_fx.value() - x_max);
      sum_exp_neg_x_k += exp_div(x_min - it_fx.value());
      sum_exp_pos_y_k += exp_div(it_fy.value() - y_max);
      sum_exp_neg_y_k += exp_div(y_min - it_fy.value());
      ++it_fx;
      ++it_fy;
    }

    while (it_xo) {
      int row = it_xo.row();

      assert(row == it_yo.row());
      assert(row == it_px.row());
      assert(row == it_py.row());

      double exp_px = exp_div(x(row) + it_xo.value() - x_max);
      double exp_nx = exp_div(x_min - x(row) - it_xo.value());
      double exp_py = exp_div(y(row) + it_yo.value() - y_max);
      double exp_ny = exp_div(y_min - y(row) - it_yo.value());
      it_px.valueRef() = exp_px;
      it_nx.valueRef() = exp_nx;
      it_py.valueRef() = exp_py;
      it_ny.valueRef() = exp_ny;

      sum_exp_pos_x_k += exp_px;
      sum_exp_neg_x_k += exp_nx;
      sum_exp_pos_y_k += exp_py;
      sum_exp_neg_y_k += exp_ny;

      ++it_xo;
      ++it_yo;
      ++it_px;
      ++it_nx;
      ++it_py;
      ++it_ny;
    }
    _sum_exp_pos_x(k) = 1 / sum_exp_pos_x_k;
    _sum_exp_neg_x(k) = 1 / sum_exp_neg_x_k;
    _sum_exp_pos_y(k) = 1 / sum_exp_pos_y_k;
    _sum_exp_neg_y(k) = 1 / sum_exp_neg_y_k;
  }
}

void LSEWirelength::evaluate(const Mat& variable, Mat& grad, double& cost, const double& gamma) const
{
  updatePinLocation(variable.col(0), variable.col(1), variable.col(2), gamma);
  cost = _hpwl.sum();
  grad.col(0) = _exp_pos_x * _sum_exp_pos_x - _exp_neg_x * _sum_exp_neg_x;  // exp(x/gamma)/Σ exp(x/gamma) - exp(-x/gamma)/Σ exp(-x/gamma).
  grad.col(1) = _exp_pos_y * _sum_exp_pos_y - _exp_neg_y * _sum_exp_neg_y;  // exp(y/gamma)/Σ exp(y/gamma) - exp(-y/gamma)/Σ exp(-y/gamma).
  grad.col(2) = grad.col(0).asDiagonal() * -_y_offset * Vec::Ones(_y_offset.cols())    // grad_x * y_displacement
                + grad.col(1).asDiagonal() * _x_offset * Vec::Ones(_x_offset.cols());  //  + grad_y * x_displacement.
  // double norm = 0.5 * (grad.col(0).lpNorm<1>() + grad.col(1).lpNorm<1>()) / grad.col(2).lpNorm<1>();
  // grad.col(2) = norm * grad.col(2);
}
void LSEWirelength::init()
{
  _orig_x_offset = SpMat(_num_vertexs, _num_edges);
  _orig_y_offset = SpMat(_num_vertexs, _num_edges);
  _sum_exp_pos_x = Vec::Zero(_num_edges);
  _sum_exp_neg_x = Vec::Zero(_num_edges);
  _sum_exp_pos_y = Vec::Zero(_num_edges);
  _sum_exp_neg_y = Vec::Zero(_num_edges);
  _hpwl = Vec::Zero(_num_edges);
}
}  // namespace ipl
