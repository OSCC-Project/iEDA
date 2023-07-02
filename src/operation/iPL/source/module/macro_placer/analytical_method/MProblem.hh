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

#include "MPDB.hh"
#include "Problem.hh"

using SpMat = Eigen::SparseMatrix<double>;
using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
using Eigen::SparseMatrix;
using std::pair;
using std::unique_ptr;
using std::unordered_map;
using std::vector;
namespace ipl {
using imp::FPInst;
using imp::MPDB;
class LSEWirelength;
class DensityModel;
class MProblem final : public Problem
{
 public:
  explicit MProblem(MPDB* db) { set_db(db); }
  MProblem() {}
  ~MProblem() {}
  virtual void setThreads(size_t n) override;
  virtual void evaluate(const Mat& variable, Mat& gradient, double& cost, int iter) const override;
  virtual Vec getSolutionDistance(const Mat& lhs, const Mat& rhs) const override;
  virtual Vec getGradientDistance(const Mat& lhs, const Mat& rhs) const override;
  virtual void getVariableBounds(const Mat& variable, Mat& low, Mat& upper) const override;
  virtual int variableMatrixRows() const override { return _num_macros; };
  virtual int variableMatrixcols() const override { return _num_types; };
  void setRandom(int num_macros, int num_nets, int netdgree, double core_w, double core_h, double utilization = 0.8);
  // void setRandomNetlist(int num_macros, int num_nets, int netdgree);

 private:
  void set_db(MPDB* db);
  void initWirelengthModel();
  void initDensityModel();
  void evalWirelength(const Mat& variable, Mat& gradient, double& cost, const double& gamma) const;
  void evalDensity(const Mat& variable, Mat& gradient, double& cost) const;
  double getPenaltyFactor() const;
  void updateLowerBound(int row, int col, double lower) { _bound[_num_macros * col + row].first = lower; }
  void updateUpperBound(int row, int col, double upper) { _bound[_num_macros * col + row].second = upper; }
  void drawImage(const Mat& variable, int index) const;

 private:
  MPDB* _db{};
  Vec _width;
  Vec _height;
  std::shared_ptr<LSEWirelength> _wl{};
  std::shared_ptr<DensityModel> _density{};
  int _num_macros{};
  int _num_nets{};
  int _num_types{};
  double _core_width{};
  double _core_height{};

  unordered_map<FPInst*, uint32_t> _inst2id{};
  vector<pair<double, double>> _bound{};
};

}  // namespace ipl

#endif  // IPL_MP_MPROBLEM