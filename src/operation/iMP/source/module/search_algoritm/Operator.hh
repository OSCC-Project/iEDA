#pragma once
#include <cassert>
#include <functional>
#include <random>
#include <tuple>
#include <unordered_map>
namespace imp {

template <typename Code, typename T, typename RandGenerator = std::mt19937>
struct Perturb
{
  using PerturbFunc = std::function<T(Code&, RandGenerator&)>;
  T operator()(Code& code) { return perturb_funcs[dist(gen)](code, gen); }

  Perturb(int seed, const std::initializer_list<double>& wl, const std::initializer_list<PerturbFunc>& fl)
      : gen(seed), dist(wl), perturb_funcs(fl)
  {
    assert(wl.size() == fl.size());
  }
  RandGenerator gen;
  std::discrete_distribution<size_t> dist;
  std::vector<PerturbFunc> perturb_funcs;
};

template <typename Code, typename T, typename RandGenerator = std::mt19937>
auto make_perturb(int seed, const std::initializer_list<double>& wl,
                  const std::initializer_list<std::function<T(Code&, RandGenerator&)>>& fl)
{
  return Perturb<Code, T, RandGenerator>(seed, wl, fl);
}

template <typename Code, typename T>
struct Restore
{
  void operator()(Code& code, T key) {}

  Restore(const std::initializer_list<T>& keyl, const std::initializer_list<std::function<void(Code&, T)>> fl) {}
  std::unordered_map<T, std::function<void(Code&, T)>> restore_funcs;
};

}  // namespace imp
