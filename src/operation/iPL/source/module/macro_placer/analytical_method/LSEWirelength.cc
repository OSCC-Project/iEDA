#include "LSEWirelength.hh"

#include <algorithm>
#include <cassert>
namespace ipl {
void LSEWirelength::setConstant(Triplets<pair<double, double>>&& moveable_pin_offsets,
                                const Triplets<pair<double, double>>& fixed_pin_location)
{
  auto compare = [](const Eigen::Triplet<std::pair<double, double>>& a, const Eigen::Triplet<std::pair<double, double>>& b) {
    return (a.col() < b.col()) || ((a.col() == b.col()) && (a.row() < b.row()));
  };
  _offset = Triplets<pair<double, double>>(moveable_pin_offsets);
  std::sort(_offset.begin(), _offset.end(), compare);

  Triplets<double> _moveable_conn;
  Triplets<double> _fixed_conn_x;
  Triplets<double> _fixed_conn_y;

  int v_num = 0;
  int e_num = 0;
  for (const auto& trip : moveable_pin_offsets) {
    int row = trip.row();
    int col = trip.col();
    _moveable_conn.emplace_back(row, col);
    v_num = std::max(v_num, row);
    e_num = std::max(e_num, col);
  }

  _exp_pos_x = SpMat(v_num, e_num);

  for (const auto& trip : fixed_pin_location) {
    int row = trip.row();
    int col = trip.col();
    _fixed_conn_x.emplace_back(row, col, trip.value().first);
    _fixed_conn_y.emplace_back(row, col, trip.value().second);
    v_num = std::max(v_num, row);
    e_num = std::max(e_num, col);
  }
  _fix_x = SpMat(v_num, e_num);
  _fix_y = SpMat(v_num, e_num);

  _exp_pos_x.setFromTriplets(_moveable_conn.begin(), _moveable_conn.end());
  _fix_x.setFromTriplets(_fixed_conn_x.begin(), _fixed_conn_x.end());
  _fix_y.setFromTriplets(_fixed_conn_y.begin(), _fixed_conn_y.end());

  _exp_pos_y = _exp_pos_x;
  _x_offset = _exp_pos_x;
  _y_offset = _exp_pos_x;
  _sum_exp_pos_x = Vec::Zero(v_num);
  _sum_exp_neg_x = Vec::Zero(v_num);
  _sum_exp_pos_y = Vec::Zero(v_num);
  _sum_exp_neg_y = Vec::Zero(v_num);
}

void LSEWirelength::updatePinLocation(const Vec& x, const Vec& y, const Vec& r, const double& gamma) const
{
  auto exp_pos_div = [gamma](double val) { return std::exp(val / gamma); };
  auto exp_neg_div = [gamma](double val) { return std::exp(val / -gamma); };
  const auto& rcos = r.array().cos();
  const auto& rsin = r.array().sin();
  auto get_pin_off = [&](double& x_off, double& y_off, int id) {
    double x_temp = x_off;
    double y_temp = y_off;
    x_off = x_temp * rcos(id) - y_temp * rsin(id);
    y_off = x_temp * rsin(id) + y_temp * rcos(id);
  };
  const auto& sum_exp_io_pos_x = _fix_x.unaryViewExpr(exp_pos_div);
  const auto& sum_exp_io_neg_x = _fix_x.unaryViewExpr(exp_neg_div);
  const auto& sum_exp_io_pos_y = _fix_y.unaryViewExpr(exp_pos_div);
  const auto& sum_exp_io_neg_y = _fix_y.unaryViewExpr(exp_neg_div);
#pragma omp parallel for num_threads(8)
  for (int k = 0; k < _exp_pos_x.outerSize(); ++k) {
    Eigen::SparseMatrix<double>::InnerIterator it_xo(_x_offset, k);
    Eigen::SparseMatrix<double>::InnerIterator it_yo(_y_offset, k);
    Eigen::SparseMatrix<double>::InnerIterator it_px(_exp_pos_x, k);
    Eigen::SparseMatrix<double>::InnerIterator it_py(_exp_pos_y, k);

    double sum_exp_pos_x_k = 0.0;
    double sum_exp_neg_x_k = 0.0;
    double sum_exp_pos_y_k = 0.0;
    double sum_exp_neg_y_k = 0.0;

    while (it_xo) {
      assert(it_xo.row() == it_yo.row());
      assert(it_xo.row() == it_px.row());
      assert(it_xo.row() == it_py.row());
      double x_off = 0;
      double y_off = 10;
      int id = it_xo.row();
      get_pin_off(x_off, y_off, id);
      it_xo.valueRef() = x_off;
      it_yo.valueRef() = y_off;
      x_off = exp_pos_div(x_off + x(id));
      y_off = exp_pos_div(y_off + y(id));
      it_px.valueRef() = x_off;
      it_py.valueRef() = y_off;

      sum_exp_pos_x_k += x_off;
      sum_exp_neg_x_k += 1 / x_off;
      sum_exp_pos_y_k += y_off;
      sum_exp_neg_y_k += 1 / y_off;

      ++it_xo;
      ++it_yo;
      ++it_px;
      ++it_py;
    }
    _sum_exp_pos_x(k) = 1 / (sum_exp_io_pos_x.col(k).sum() + sum_exp_pos_x_k);
    _sum_exp_neg_x(k) = 1 / (sum_exp_io_neg_x.col(k).sum() + sum_exp_neg_x_k);
    _sum_exp_pos_y(k) = 1 / (sum_exp_io_pos_y.col(k).sum() + sum_exp_pos_y_k);
    _sum_exp_neg_y(k) = 1 / (sum_exp_io_neg_y.col(k).sum() + sum_exp_neg_y_k);
  }
}

void LSEWirelength::evaluate(const Vec& x, const Vec& y, const Vec& r, const double& gamma) const
{
  updatePinLocation(x, y, r, gamma);
  auto inverse = [](double val) { return 1 / val; };
  const auto& exp_neg_x = _exp_pos_x.unaryViewExpr(inverse);
  const auto& exp_neg_y = _exp_pos_y.unaryViewExpr(inverse);
  Mat grad = Mat::Zero(x.rows(), 3);
  grad.col(0) = _exp_pos_x * _sum_exp_pos_x - exp_neg_x * _sum_exp_neg_x;
  grad.col(1) = _exp_pos_y * _sum_exp_pos_y - exp_neg_y * _sum_exp_neg_y;
  grad.col(2) = grad.col(0).asDiagonal() * -_y_offset * Vec::Ones(_y_offset.cols())
                + grad.col(1).asDiagonal() * _x_offset * Vec::Ones(_x_offset.cols());
}
}  // namespace ipl
