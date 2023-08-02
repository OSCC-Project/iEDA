#include "Problem.hh"

Vec ipl::Problem::getSolutionDistance(const Mat& lhs, const Mat& rhs) const
{
  return Vec((lhs - rhs).colwise().norm());
}

Vec ipl::Problem::getGradientDistance(const Mat& lhs, const Mat& rhs) const
{
  return Vec((lhs - rhs).colwise().norm());
}
