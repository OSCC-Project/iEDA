#include "MProblem.hh"

#include <Eigen/Sparse>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <string>

#include "LSEWirelength.hh"
#include "utility/Image.hh"
// #include "MPDB.hh"
namespace ipl {
void MProblem::setRandom(int num_macros, int num_nets, int netdgree, double core_w, double core_h, double utilization)
{
  _num_types = 3;
  _num_macros = num_macros;
  _num_nets = num_macros;
  _core_width = core_w;
  _core_height = core_h;
  _width = Vec(_num_macros);
  _height = Vec(_num_macros);

  vector<Triplet<double>> moveable_x_offset;
  vector<Triplet<double>> moveable_y_offset;
  vector<Triplet<double>> fixed_x_location;
  vector<Triplet<double>> fixed_y_location;

  double sq = std::sqrt(utilization);
  double w = 2 * sq * core_w / num_macros;
  double h = 2 * sq * core_h / num_macros;

  _width.setConstant(w);
  _height.setConstant(h);

  std::random_device r;
  std::default_random_engine e1(r());
  std::uniform_int_distribution<int> rand_bool(0, 1);
  std::uniform_int_distribution<int> rand_range(-5, 5);

  auto rand = [&](double& x, double& y) {
    x = rand_bool(e1) ? 0.5 : -0.5;
    y = (double) rand_range(e1) / 10;
    (rand_bool(e1)) ? std::swap(x, y) : void();
  };
  double x = 0;
  double y = 0;
  for (int j = 0; j < _num_nets; j++) {
    for (int i = 0; i < netdgree; i++) {
      rand(x, y);
      moveable_x_offset.emplace_back((j + i) % _num_macros, j, x * _width(i));
      moveable_y_offset.emplace_back((j + i) % _num_macros, j, y * _height(i));
    }
  }
  double cx = core_w / 2;
  double cy = core_h / 2;
  for (int i = 0; i < _num_nets; i++) {
    rand(x, y);
    fixed_x_location.emplace_back(0, i, cx * (1 + x));
    fixed_y_location.emplace_back(0, i, cy * (1 + y));
    rand(x, y);
    fixed_x_location.emplace_back(1, i, cx * (1 + x));
    fixed_y_location.emplace_back(1, i, cy * (1 + y));
  }
  wl = std::make_shared<LSEWirelength>(LSEWirelength(_num_macros, _num_nets));
  wl->setConstant(moveable_x_offset, moveable_y_offset, fixed_x_location, fixed_y_location);
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
  wl = std::make_shared<LSEWirelength>(LSEWirelength(_num_macros, _num_nets));
  wl->setConstant(moveable_x_offset, moveable_y_offset, fixed_x_location, fixed_y_location);
}
void MProblem::initDensityModel()
{
}

void MProblem::evalWirelength(const Mat& variable, Mat& gradient, double& cost, const double& gamma) const
{
  wl->evaluate(variable, gradient, cost, 100);
}

double MProblem::getPenaltyFactor() const
{
  return 0.0;
}

void MProblem::drawImage(const Mat& variable, int index) const
{
  Image img(_core_width, _core_height, _num_macros);
  for (int i = 0; i < _num_macros; i++) {
    img.drawRect(variable(i, 0), variable(i, 1), _width(i), _height(i), variable(i, 2));
  }
  img.save("/home/huangfuxing/Prog_cpp/iEDA/bin/t" + std::to_string(index) + ".jpg");
}

void MProblem::evaluate(const Mat& variable, Mat& gradient, double& cost, int iter) const
{
  const Vec& x = variable.col(0);
  const Vec& y = variable.col(1);
  const Vec& r = variable.col(2);
  double hpwl = 0;
  evalWirelength(variable, gradient, hpwl, 100);
  cost = hpwl;
  assert(!std::isnan(cost));
  if (iter % 1 == 0)
    drawImage(variable, iter);
}

double MProblem::getSolutionDistance(const Vec& a, const Vec& b, int col) const
{
  constexpr double k_2pi = 2 * M_PI;
  auto angle_dis = [](double a, double b) {
    double angle_a = std::fmod(a, k_2pi);
    double angle_b = std::fmod(b, k_2pi);
    double dis = std::fabs(angle_a - angle_b);
    dis = std::min(dis, k_2pi - dis);
    return dis;
  };
  return (col != 2 ? (a - b).norm() : a.binaryExpr(b, angle_dis).norm());
  // return (col != 2 ? (a - b).norm() : a.dot(b) / a.norm() / b.norm());
}

}  // namespace ipl
