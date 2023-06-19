/**
 * @file LSEWirelength.hh
 * @author Fuxing Huang (fxxhuang@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-06-11
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef IPL_MP_LSE_WIRELENGTH
#define IPL_MP_LSE_WIRELENGTH
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
using SpMat = Eigen::SparseMatrix<double>;
using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
using paird = std::pair<double, double>;
using Eigen::Triplet;
using std::vector;
namespace ipl {
class LSEWirelength
{
 public:
  LSEWirelength() = delete;
  LSEWirelength(int num_v, int num_e) : _num_vertexs(num_v), _num_edges(num_e) { init(); }
  ~LSEWirelength() {}
  void setConstant(const vector<Triplet<double>>& moveable_x_offset, const vector<Triplet<double>>& moveable_y_offset,
                   const vector<Triplet<double>>& fixed_x_location, const vector<Triplet<double>>& fixed_y_location);

  void updatePinLocation(const Vec& x, const Vec& y, const Vec& r, const double& gamma) const;

  void evaluate(const Mat& variable, Mat& grad, double& cost, const double& gamma) const;

 private:
  void init();

 private:
  int _num_vertexs;
  int _num_edges;
  mutable SpMat _exp_pos_x;    // exp(x/gamma) O(|pins|)
  mutable SpMat _exp_neg_x;    // exp(-x/gamma) O(|pins|)
  mutable SpMat _exp_pos_y;    // exp(y/gamma) O(|pins|)
  mutable SpMat _exp_neg_y;    // exp(-y/gamma) O(|pins|)
  mutable SpMat _x_offset;     // x_offset after rotation O(|pins|)
  mutable SpMat _y_offset;     // y_offset after rotation O(|pins|)
  mutable Vec _sum_exp_pos_x;  // Σ exp(x/gamma) O(|E|)
  mutable Vec _sum_exp_neg_x;  // Σ exp(-x/gamma) O(|E|)
  mutable Vec _sum_exp_pos_y;  // Σ exp(y/gamma) O(|E|)
  mutable Vec _sum_exp_neg_y;  // Σ exp(-y/gamma) O(|E|)
  mutable Vec _hpwl;

  SpMat _fix_x;  // fix pin x O(|IO|)
  SpMat _fix_y;  // fix pin y O(|IO|)

  SpMat _orig_x_offset;  // original x_offset O(|pins|)
  SpMat _orig_y_offset;  // original y_offset O(|pins|)
};
}  // namespace ipl

#endif  // IPL_MP_LSE_WIRELENGTH