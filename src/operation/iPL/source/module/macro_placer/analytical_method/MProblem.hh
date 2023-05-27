/**
 * @file MProplem.hh
 * @author Fuxing Huang (fxxhuang@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-05-16
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef IPL_MP_MPROBLEM
#define IPL_MP_MPROBLEM
#include <vector>

// #include "MPDB.hh"
#include "Problem.hh"

namespace ipl::imp {
class MPDB;
class MProblem final : public Problem
{
 public:
  explicit MProblem(MPDB* db) { set_db(db); }
  ~MProblem() {}
  void evaluate(const MatrixXd& variable, MatrixXd& gradient, float& cost, int iter) const override;
  virtual double getLowerBound(int row, int col) const override { return _bound[_var_rows * col + row].first; }
  virtual double getUpperBound(int row, int col) const override { return _bound[_var_rows * col + row].second; };
  virtual int variableMatrixRows() const override { return _var_rows; };
  virtual int variableMatrixcols() const override { return _var_cols; };
  void set_db(MPDB* db);

 private:
  MatrixXd getWirelengthGradient(const VectorXd& x, const VectorXd& y, const VectorXd& angle, double gamma) const;
  MatrixXd getDensityGradient(const VectorXd& x, const VectorXd& y, const VectorXd& angle) const;
  double evalHpwl(const VectorXd& x, const VectorXd& y, const VectorXd& angle) const;
  double getPenaltyFactor() const;
  void updateLowerBound(int row, int col, double lower) { _bound[_var_rows * col + row].first = lower; }
  void updateUpperBound(int row, int col, double upper) { _bound[_var_rows * col + row].second = upper; }

 private:
  MPDB* _db = nullptr;
  int _var_rows = 0;
  int _var_cols = 0;
  double _core_width = 0;
  double _core_height = 0;
  std::vector<std::pair<double, double>> _bound;
};

}  // namespace ipl::imp

#endif  // IPL_MP_MPROBLEM