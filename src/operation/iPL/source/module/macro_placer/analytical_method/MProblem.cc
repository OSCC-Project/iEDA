#include "MProblem.hh"

#include "MPDB.hh"
namespace ipl::imp {

void MProblem::set_db(MPDB* db)
{
  if (db == nullptr) {
    return;
  }
  _db = db;
  _var_cols = 3;
  _var_rows = _db->get_place_macro_list().size();
  _core_width = _db->get_layout()->get_core_shape()->get_width();
  _core_height = _db->get_layout()->get_core_shape()->get_height();
}

MatrixXd MProblem::getWirelengthGradient(const VectorXd& x, const VectorXd& y, const VectorXd& angle, double gamma) const
{
  return MatrixXd();
}

MatrixXd MProblem::getDensityGradient(const VectorXd& x, const VectorXd& y, const VectorXd& angle) const
{
  return MatrixXd();
}

double MProblem::getPenaltyFactor() const
{
  return 0.0;
}

void MProblem::evaluate(const MatrixXd& variable, MatrixXd& gradient, float& cost, int iter) const
{
  const VectorXd& x = variable.col(0);
  const VectorXd& y = variable.col(1);
  const VectorXd& angle = variable.col(2);
  const MatrixXd& wirelength_grad = getWirelengthGradient(x, y, angle, 100);
  const MatrixXd& density_grad = getDensityGradient(x, y, angle);
  double penalty = getPenaltyFactor();
  gradient = wirelength_grad + penalty * density_grad;
}

}  // namespace ipl::imp
