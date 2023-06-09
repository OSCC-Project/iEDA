#ifndef IPL_MP_LSE_WIRELENGTH
#define IPL_MP_LSE_WIRELENGTH
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
using SpMat = Eigen::SparseMatrix<double>;
using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
using std::pair;
using std::vector;
template <typename T>
using Triplets = std::vector<Eigen::Triplet<T>>;
namespace ipl {
class MPDB;
class LSEWirelength
{
 public:
  LSEWirelength() {}
  ~LSEWirelength() {}
  void setConstant(Triplets<pair<double, double>>&& moveable_pin_offsets, const Triplets<pair<double, double>>& fixed_pin_location);

  void updatePinLocation(const Vec& x, const Vec& y, const Vec& r, const double& gamma) const;

  void evaluate(const Vec& x, const Vec& y, const Vec& r, const double& gamma) const;

 private:
  mutable SpMat _exp_pos_x;  // exp(x/gamma)
  mutable SpMat _exp_pos_y;  // exp(y/gamma)
  mutable SpMat _fix_x;
  mutable SpMat _fix_y;
  mutable SpMat _x_offset;
  mutable SpMat _y_offset;
  mutable Vec _sum_exp_pos_x;  // Σ exp(x/gamma)
  mutable Vec _sum_exp_neg_x;  // Σ exp(-x/gamma)
  mutable Vec _sum_exp_pos_y;  // Σ exp(y/gamma)
  mutable Vec _sum_exp_neg_y;  // Σ exp(-y/gamma)
  Triplets<pair<double, double>> _offset;
};
}  // namespace ipl

#endif  // IPL_MP_LSE_WIRELENGTH