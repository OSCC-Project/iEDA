#include "MProblem.hh"

#include <Eigen/Sparse>
#include <algorithm>

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
  _connectivity = SparseMatrix<double>(_var_rows, nets.size());
  //   _sum_exp_x.resize(nets.size());
  _io_pin_pos.resize(nets.size());

  std::vector<Eigen::Triplet<double>> triplets;
  Eigen::Index net_num = 0;
  for (const auto& net : nets) {
    for (const auto& pin : net->get_pin_list()) {
      if (pin->is_io_pin()) {
        _io_pin_pos[net_num].emplace_back(pin->get_x(), pin->get_y());
      } else {
        triplets.emplace_back(net_num, _inst2id[pin->get_instance()], 1.0);
      }
      net_num++;
    }
  }
  _connectivity.setFromTriplets(triplets.begin(), triplets.end());
}
void MProblem::initDensityModel()
{
}

MatrixXd MProblem::getWirelengthGradient(const VectorXd& x, const VectorXd& y, const VectorXd& r, double gamma) const
{
  const auto& macros = _db->get_total_macro_list();
  const auto& nets = _db->get_new_net_list();
  const VectorXd& exp_x = (x / gamma).array().exp();
  const VectorXd& exp_y = (y / gamma).array().exp();

  return MatrixXd();
}

MatrixXd MProblem::getDensityGradient(const VectorXd& x, const VectorXd& y, const VectorXd& r) const
{
  return MatrixXd();
}

double MProblem::evalHpwl(const VectorXd& x, const VectorXd& y, const VectorXd& r) const
{
  return 0.0;
}

double MProblem::getPenaltyFactor() const
{
  return 0.0;
}

void MProblem::evaluate(const MatrixXd& variable, MatrixXd& gradient, double& cost, int iter) const
{
  const VectorXd& x = variable.col(0);
  const VectorXd& y = variable.col(1);
  const VectorXd& r = variable.col(2);
  const MatrixXd& wirelength_grad = getWirelengthGradient(x, y, r, 100);
  const MatrixXd& density_grad = getDensityGradient(x, y, r);
  double penalty = getPenaltyFactor();
  gradient = wirelength_grad + penalty * density_grad;
  cost = evalHpwl(x, y, r);
}

}  // namespace ipl::imp
