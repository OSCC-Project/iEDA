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
#include "Problem.hh"
namespace ipl {
class MProplem final : Problem
{
 public:
  MProplem(/* args */);
  ~MProplem();
  virtual void evaluate() {}
};

MProplem::MProplem(/* args */)
{
}

MProplem::~MProplem()
{
}

}  // namespace ipl

#endif  // IPL_MP_MPROBLEM