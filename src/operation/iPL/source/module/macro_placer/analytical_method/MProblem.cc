#include "MProblem.hh"

#include <Eigen/Sparse>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <string>

#include "DensityModel.hh"
#include "LSEWirelength.hh"
#include "utility/Image.hh"
// #include "MPDB.hh"
namespace ipl {
void MProblem::setRandom(int num_macros, int num_nets, int netdgree, double core_w, double core_h, double utilization)
{
  _num_types = 3;
  _num_macros = num_macros;
  _num_nets = num_nets;
  _core_width = core_w;
  _core_height = core_h;
  _width = Vec(_num_macros);
  _height = Vec(_num_macros);

  vector<Triplet<double>> moveable_x_offset;
  vector<Triplet<double>> moveable_y_offset;
  vector<Triplet<double>> fixed_x_location;
  vector<Triplet<double>> fixed_y_location;

  double area = core_w * core_h * utilization / num_macros;

  std::random_device r;
  std::default_random_engine e1(r());
  std::uniform_int_distribution<int> rand_bool(0, 1);
  std::uniform_int_distribution<int> rand_range(-5, 5);
  std::uniform_real_distribution<double> rand_ar(1.0 / 3.0, 3.0);
  std::uniform_int_distribution<int> rand_connect(0, _num_macros - 1);

  for (int i = 0; i < _num_macros; i++) {
    double h = std::sqrt(area * rand_ar(e1));
    double w = area / h;
    _width(i) = w;
    _height(i) = h;
  }

  auto rand = [&](double& x, double& y) {
    x = rand_bool(e1) ? 0.5 : -0.5;
    y = (double) rand_range(e1) / 10;
    (rand_bool(e1)) ? std::swap(x, y) : void();
  };
  double x = 0;
  double y = 0;
  double cx = core_w / 2;
  double cy = core_h / 2;
  for (int j = 0; j < _num_nets; j++) {
    for (int i = 0; i < netdgree; i++) {
      int v = rand_connect(e1);
      rand(x, y);
      if (v < _num_nets) {
        moveable_x_offset.emplace_back(v, j, x * _width(i));
        moveable_y_offset.emplace_back(v, j, y * _height(i));
      } else {
        fixed_x_location.emplace_back(v - _num_nets, i, cx * (1 + x));
        fixed_y_location.emplace_back(v - _num_nets, i, cy * (1 + y));
      }
    }
  }
  for (int i = 0; i < _num_nets; i++) {
    rand(x, y);
    fixed_x_location.emplace_back(0, i, cx * (1 + x));
    fixed_y_location.emplace_back(0, i, cy * (1 + y));
    rand(x, y);
    fixed_x_location.emplace_back(1, i, cx * (1 + x));
    fixed_y_location.emplace_back(1, i, cy * (1 + y));
  }
  _wl = std::make_shared<LSEWirelength>(LSEWirelength(_num_macros, _num_nets));
  _wl->setConstant(moveable_x_offset, moveable_y_offset, fixed_x_location, fixed_y_location);
  initDensityModel();
}

void MProblem::set_db(MPDB* db)
{
  if (db == nullptr) {
    return;
  }
  _db = db;
  _num_types = 3;
  _num_macros = _db->get_total_macro_list().size();
  _num_nets = _db->get_new_net_list().size();
  _core_width = _db->get_layout()->get_core_shape()->get_width();
  _core_height = _db->get_layout()->get_core_shape()->get_height();
  const auto& macro_list = _db->get_total_macro_list();
  for (uint32_t i = 0; i < macro_list.size(); i++) {
    _inst2id[macro_list[i]] = i;
  }

  initWirelengthModel();
  initDensityModel();
}

void MProblem::initWirelengthModel()
{
  vector<Triplet<double>> moveable_x_offset;
  vector<Triplet<double>> moveable_y_offset;
  vector<Triplet<double>> fixed_x_location;
  vector<Triplet<double>> fixed_y_location;
  uint32_t col = 0;
  for (const auto& net : _db->get_new_net_list()) {
    uint32_t fix_row = 0;
    for (const auto& pin : net->get_pin_list()) {
      if (!pin->is_io_pin()) {
        uint32_t row = _inst2id.at(pin->get_instance());
        moveable_x_offset.emplace_back(row, col, pin->get_orig_xoff());
        moveable_y_offset.emplace_back(row, col, pin->get_orig_yoff());
      } else {
        fixed_x_location.emplace_back(fix_row, col, pin->get_x());
        fixed_y_location.emplace_back(fix_row, col, pin->get_x());
        fix_row++;
      }
    }
    col++;
  }
  _wl = std::make_shared<LSEWirelength>(LSEWirelength(_num_macros, _num_nets));
  _wl->setConstant(moveable_x_offset, moveable_y_offset, fixed_x_location, fixed_y_location);
}
void MProblem::initDensityModel()
{
  _density = std::make_shared<DensityModel>();
  _density->setConstant(_width, _height, _core_width, _core_height);
}

void MProblem::evalWirelength(const Mat& variable, Mat& gradient, double& cost, const double& gamma) const
{
  _wl->evaluate(variable, gradient, cost, 100);
}

void MProblem::evalDensity(const Mat& variable, Mat& gradient, double& cost) const
{
  _density->evaluate(variable, gradient, cost);
}

double MProblem::getPenaltyFactor() const
{
  return 0.0;
}

void MProblem::drawImage(const Mat& variable, int index) const
{
  Image img(_core_width, _core_height, _num_macros);
  img.drawRect(_core_width / 2, _core_height / 2, _core_width, _core_height);
  for (int i = 0; i < _num_macros; i++) {
    img.drawRect(variable(i, 0), variable(i, 1), _width(i), _height(i), variable(i, 2));
  }
  img.save("/home/huangfuxing/Prog_cpp/iEDA/bin/img" + std::to_string(index) + ".jpg");
}

void MProblem::setThreads(size_t n)
{
  _wl->setThreads(n);
  _density->setThreads(n);
  Problem::setThreads(n);
}

void MProblem::evaluate(const Mat& variable, Mat& gradient, double& cost, int iter) const
{
  const Vec& x = variable.col(0);
  const Vec& y = variable.col(1);
  const Vec& r = variable.col(2);
  double hpwl = 0;
  Mat wl_g = Mat::Zero(variableMatrixRows(), variableMatrixcols());
  Mat d_g = Mat::Zero(variableMatrixRows(), variableMatrixcols());
  evalDensity(variable, d_g, hpwl);
  cost = hpwl;
  evalWirelength(variable, wl_g, hpwl, 100);
  if (iter == 0)
    _lambda = wl_g.lpNorm<1>() / d_g.lpNorm<1>();
  d_g = _lambda * d_g;
  gradient = wl_g + d_g;
  if (iter % 1 == 0)
    drawImage(variable, iter);
}

Vec MProblem::getSolutionDistance(const Mat& lhs, const Mat& rhs) const
{
  constexpr double k_2pi = 2 * M_PI;
  auto angle_dis = [](double a, double b) {
    double angle_a = std::fmod(a, k_2pi);
    double angle_b = std::fmod(b, k_2pi);
    double dis = std::fabs(angle_a - angle_b);
    dis = std::min(dis, k_2pi - dis);
    return dis;
  };
  Vec dis(lhs.cols());
  auto lhs_xy = lhs.leftCols(2);
  auto rhs_xy = rhs.leftCols(2);
  double dis_xy = (lhs_xy - rhs_xy).norm();
  dis(0) = dis_xy;
  dis(1) = dis_xy;
  dis(2) = lhs.col(2).binaryExpr(rhs.col(2), angle_dis).norm();
  return dis;
}

Vec MProblem::getGradientDistance(const Mat& lhs, const Mat& rhs) const
{
  Vec dis(3);
  double dis_xy = (lhs.leftCols(2) - rhs.leftCols(2)).norm();
  dis(0) = dis_xy;
  dis(1) = dis_xy;
  dis(2) = (lhs.col(2) - rhs.col(2)).norm();
  return dis;
}

void MProblem::getVariableBounds(const Mat& variable, Mat& low, Mat& upper) const
{
  const auto& w_2 = _width / 2;
  const auto& y_2 = _height / 2;
  const auto& cos = variable.col(2).array().cos();
  const auto& sin = variable.col(2).array().sin();
  for (int i = 0; i < variableMatrixRows(); i++) {
    // double cx = variable(i, 0);
    // double cy = variable(i, 1);
    double x_off = std::max(std::abs(w_2(i) * cos(i) - y_2(i) * sin(i)), std::abs(w_2(i) * cos(i) + y_2(i) * sin(i)));
    double y_off = std::max(std::abs(w_2(i) * sin(i) - y_2(i) * cos(i)), std::abs(w_2(i) * sin(i) + y_2(i) * cos(i)));
    assert(x_off > 0 || y_off > 0);
    low(i, 0) = 0.0 + x_off + 1;
    upper(i, 0) = _core_width - x_off - 1;
    low(i, 1) = 0.0 + y_off + 1;
    upper(i, 1) = _core_height - y_off - 1;
    low(i, 2) = std::numeric_limits<double>::lowest();
    upper(i, 2) = std::numeric_limits<double>::max();
  }
}

// double SolutionDistance(const Vec& a, const Vec& b, int col) const
// {
//   constexpr double k_2pi = 2 * M_PI;
//   auto angle_dis = [](double a, double b) {
//     double angle_a = std::fmod(a, k_2pi);
//     double angle_b = std::fmod(b, k_2pi);
//     double dis = std::fabs(angle_a - angle_b);
//     dis = std::min(dis, k_2pi - dis);
//     return dis;
//   };
//   return (col != 2 ? (a - b).norm() : a.binaryExpr(b, angle_dis).norm());
//   // return (col != 2 ? (a - b).norm() : a.dot(b) / a.norm() / b.norm());
// }

}  // namespace ipl
