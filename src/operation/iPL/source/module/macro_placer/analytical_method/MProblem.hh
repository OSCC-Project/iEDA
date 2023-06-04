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

#include <Eigen/Sparse>
#include <memory>
#include <unordered_map>
#include <vector>

#include "Problem.hh"

using SpMat = Eigen::SparseMatrix<double>;
using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
using Eigen::SparseMatrix;
using std::pair;
using std::unique_ptr;
using std::unordered_map;
using std::vector;
namespace ipl::imp {
class MPDB;
class FPInst;
class LSEWireLength;
class MProblem final : public Problem
{
 public:
  explicit MProblem(MPDB* db) { set_db(db); }
  ~MProblem() {}
  virtual void evaluate(const Mat& variable, Mat& gradient, double& cost, int iter) const override;
  virtual double getLowerBound(int row, int col) const override { return _bound[_var_rows * col + row].first; }
  virtual double getUpperBound(int row, int col) const override { return _bound[_var_rows * col + row].second; };
  virtual int variableMatrixRows() const override { return _var_rows; };
  virtual int variableMatrixcols() const override { return _var_cols; };
  void set_db(MPDB* db);

 private:
  void initWirelengthModel();
  void initDensityModel();
  Mat getWirelengthGradient(const Vec& x, const Vec& y, const Vec& r, double gamma) const;
  Mat getDensityGradient(const Vec& x, const Vec& y, const Vec& r) const;
  double evalHpwl(const Vec& x, const Vec& y, const Vec& r) const;
  double evalOverflow(const Vec& x, const Vec& y, const Vec& r) const;
  double getPenaltyFactor() const;
  void updateLowerBound(int row, int col, double lower) { _bound[_var_rows * col + row].first = lower; }
  void updateUpperBound(int row, int col, double upper) { _bound[_var_rows * col + row].second = upper; }

 private:
  MPDB* _db = {nullptr};
  int _var_rows = {};
  int _var_cols = {};
  double _core_width = {};
  double _core_height = {};

  unordered_map<FPInst*, uint32_t> _inst2id = {};
  vector<pair<double, double>> _bound = {};
  SpMat _connectivity;
  SpMat _io_conn_x;
  SpMat _io_conn_y;
  Vec _sum_exp_x = {};
  Vec _sum_exp_y = {};
};

}  // namespace ipl::imp

#endif  // IPL_MP_MPROBLEM