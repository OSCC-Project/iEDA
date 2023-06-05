#include "MProblem.hh"

#include <Eigen/Sparse>
#include <algorithm>
#include <cassert>
#include <cmath>

#include "MPDB.hh"
namespace ipl::imp {

void MProblem::set_db(MPDB* db)
{
  if (db == nullptr) {
    return;
  }
  _db = db;
  _var_cols = 3;
  _var_rows = _db->get_total_macro_list().size();
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
  const auto& nets = _db->get_new_net_list();

  std::vector<Eigen::Triplet<double>> macro_triplets;
  std::vector<Eigen::Triplet<double>> io_triplets_x;
  std::vector<Eigen::Triplet<double>> io_triplets_y;

  Eigen::Index net_num = 0;
  Eigen::Index max_io_num = 0;

  for (const auto& net : nets) {
    Eigen::Index io_num = 0;
    for (const auto& pin : net->get_pin_list()) {
      if (pin->is_io_pin()) {
        io_triplets_x.emplace_back(io_num, net_num, pin->get_x());
        io_triplets_y.emplace_back(io_num, net_num, pin->get_y());
        max_io_num = std::max(max_io_num, io_num++);
      } else {
        assert(pin->get_instance() != nullptr);
        macro_triplets.emplace_back(_inst2id[pin->get_instance()], net_num, 1.0);
      }
    }
    net_num++;
  }

  _io_conn_x = SpMat(max_io_num, nets.size());
  _io_conn_y = SpMat(max_io_num, nets.size());
  _connectivity = SpMat(_var_rows, nets.size());

  _io_conn_x.setFromTriplets(io_triplets_x.begin(), io_triplets_x.end());
  _io_conn_y.setFromTriplets(io_triplets_y.begin(), io_triplets_y.end());
  _connectivity.setFromTriplets(macro_triplets.begin(), macro_triplets.end());

  _io_conn_x.makeCompressed();
  _io_conn_y.makeCompressed();
  _connectivity.makeCompressed();
}
void MProblem::initDensityModel()
{
}

Mat MProblem::getWirelengthGradient(const Vec& x, const Vec& y, const Vec& r, double gamma) const
{
  const auto& macros = _db->get_total_macro_list();
  const auto& nets = _db->get_new_net_list();

  auto exp_pos_div = [gamma](double val) { return std::exp(val / gamma); };
  auto exp_neg_div = [gamma](double val) { return std::exp(val / -gamma); };
  auto inverse = [](double val) { return (val == 0) ? 0 : 1 / val; };

  // const auto& pos_x_exp = x.unaryExpr(exp_pos_div);      // exp(x_i/gamma)
  // const auto& pos_y_exp = y.unaryExpr(exp_pos_div);      // exp(y_i/gamma)
  // const auto& neg_x_exp = pos_x_exp.unaryExpr(inverse);  // exp(-x_i/gamma)
  // const auto& neg_y_exp = pos_y_exp.unaryExpr(inverse);  // exp(-y_i/gamma)

  Eigen::RowVectorXd ones_1 = Eigen::RowVectorXd::Ones(_io_conn_x.rows());

  const auto& io_exp_pos_x = ones_1 * _io_conn_x.unaryExpr(exp_pos_div);
  const auto& io_exp_neg_x = ones_1 * _io_conn_x.unaryExpr(exp_neg_div);
  const auto& io_exp_pos_y = ones_1 * _io_conn_y.unaryExpr(exp_pos_div);
  const auto& io_exp_neg_y = ones_1 * _io_conn_y.unaryExpr(exp_neg_div);

  auto get_pin_off = [](double& x_off, double& y_off, double r) {
    double x_temp = x_off;
    double y_temp = y_off;
    double cos = std::cos(r);
    double sin = std::sin(r);
    x_off = x_temp * cos - y_temp * sin;
    y_off = x_temp * sin + y_temp * cos;
  };
  // The hypergraph matrix of nets where the each coeff is the coordinate of the corresponding pin.
  SpMat connectivity_x = _connectivity;
  SpMat connectivity_y = _connectivity;

  Eigen::Index net_num = 0;
  for (const auto& net : nets) {
    for (const auto& pin : net->get_pin_list()) {
      if (pin->is_io_pin())
        continue;
      double x_off = (double) pin->get_orig_xoff();
      double y_off = (double) pin->get_orig_yoff();
      uint32_t id = _inst2id.at(pin->get_instance());
      get_pin_off(x_off, y_off, r(id));
      connectivity_x.coeffRef(id, net_num) = x(id) + x_off;
      connectivity_y.coeffRef(id, net_num) = y(id) + y_off;
    }
    net_num++;
  }
  const auto& conn_pos_x = connectivity_x.unaryExpr(exp_pos_div).eval();  // exp((x+x_off)/gamma)
  const auto& conn_pos_y = connectivity_y.unaryExpr(exp_pos_div).eval();  // exp(-(x+x_off)/gamma)
  const auto& conn_neg_x = conn_pos_x.unaryExpr(inverse).eval();          // exp((y+y_off)/gamma)
  const auto& conn_neg_y = conn_pos_y.unaryExpr(inverse).eval();          // exp(-(y+y_off)/gamma)
  Eigen::RowVectorXd ones_2 = Eigen::RowVectorXd::Ones(_connectivity.rows());

  const auto& sum_exp_pos_x = (ones_2 * conn_pos_x + io_exp_pos_x).unaryExpr(inverse).transpose();  // ∑ exp((x+x_off)/gamma)
  const auto& sum_exp_neg_x = (ones_2 * conn_neg_x + io_exp_neg_x).unaryExpr(inverse).transpose();  // ∑ exp(-(x+x_off)/gamma)
  const auto& sum_exp_pos_y = (ones_2 * conn_pos_y + io_exp_pos_y).unaryExpr(inverse).transpose();  // ∑ exp((y+y_off)/gamma)
  const auto& sum_exp_neg_y = (ones_2 * conn_neg_y + io_exp_neg_y).unaryExpr(inverse).transpose();  // ∑ exp(-(y+y_off)/gamma)

  Vec grad_x = conn_pos_x * sum_exp_pos_x - conn_neg_x * sum_exp_neg_x;
  Vec grad_y = conn_pos_y * sum_exp_pos_y - conn_neg_y * sum_exp_neg_y;
  // Vec grad_r = grad_x

  // const auto& sum_exp_pos_x = pos_x_exp.transpose() * conn_pos_x + io_exp_pos_x;  // ∑ exp((x+x_off)/gamma)
  // const auto& sum_exp_neg_x = neg_x_exp.transpose() * conn_neg_x + io_exp_neg_x;  // ∑ exp(-(x+x_off)/gamma)
  // const auto& sum_exp_pos_y = pos_y_exp.transpose() * conn_pos_y + io_exp_pos_y;  // ∑ exp((y+y_off)/gamma)
  // const auto& sum_exp_neg_y = neg_y_exp.transpose() * conn_neg_y + io_exp_neg_y;  // ∑ exp(-(y+y_off)/gamma)

  //  connectivity_x = (connectivity_x / gamma).unaryExpr(dexp);
  //  connectivity_y = (connectivity_y / gamma).unaryExpr(dexp);

  // sum_exp_pos_x += ones_1 * connectivity_x;
  // sum_exp_neg_x += ones_1 * connectivity_x.unaryExpr(inverse);
  // sum_exp_pos_y += ones_1 * connectivity_y;
  // sum_exp_neg_y += ones_1 * connectivity_y.unaryExpr(inverse);

  // sum_exp_pos_x = ;

  // Vec grad_x
  //     = connectivity_x * sum_exp_pos_x.unaryExpr(inverse) - connectivity_x.unaryExpr(inverse) - sum_exp_neg_x.unaryExpr(inverse);
  // Vec grad_y
  //     = connectivity_y * sum_exp_pos_y.unaryExpr(inverse) - connectivity_y.unaryExpr(inverse) - sum_exp_neg_y.unaryExpr(inverse);

  return Mat();
}

Mat MProblem::getDensityGradient(const Vec& x, const Vec& y, const Vec& r) const
{
  return Mat();
}

double MProblem::evalHpwl(const Vec& x, const Vec& y, const Vec& r) const
{
  return 0.0;
}

double MProblem::evalOverflow(const Vec& x, const Vec& y, const Vec& r) const
{
  return 0.0;
}

double MProblem::getPenaltyFactor() const
{
  return 0.0;
}

void MProblem::evaluate(const Mat& variable, Mat& gradient, double& cost, int iter) const
{
  const Vec& x = variable.col(0);
  const Vec& y = variable.col(1);
  const Vec& r = variable.col(2);
  const Mat& wirelength_grad = getWirelengthGradient(x, y, r, 100);
  const Mat& density_grad = getDensityGradient(x, y, r);
  double penalty = getPenaltyFactor();
  gradient = wirelength_grad + penalty * density_grad;
  cost = evalHpwl(x, y, r);
}

}  // namespace ipl::imp
