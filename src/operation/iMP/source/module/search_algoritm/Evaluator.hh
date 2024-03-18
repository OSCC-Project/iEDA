#pragma once
#include <functional>
#include <numeric>
#include <vector>
namespace imp {

template <template <typename> typename Wirelength, typename T>
struct EvalWirelength
{
  template <typename Product>
  float operator()(const Product& product)
  {
    return this->operator()(product.x, product.y);
  }
  float operator()(const std::vector<T>& lx, const std::vector<T>& ly)
  {
    return (float) total_wirelength(lx, ly) / total_netweight / max_wirelength;
  }
  EvalWirelength(float outline_width, float outline_height, const Wirelength<T>& wl, const std::vector<T>& net_weights)
      : total_wirelength(wl)
  {
    max_wirelength = outline_width + outline_height;
    total_netweight = std::accumulate(net_weights.begin(), net_weights.end(), 0);
  }
  float total_netweight;
  float max_wirelength;
  Wirelength<T> total_wirelength;
};

template <template <typename> typename Wirelength, typename T>
struct EvalWirelength2
{
  template <typename Product>
  float operator()(const Product& product)
  {
    return total_wirelength(product.x, product.y, product.dx, product.dy) / total_netweight / max_wirelength;
  }
  EvalWirelength2(float outline_width, float outline_height, const Wirelength<T>& wl, const std::vector<float>& net_weights)
      : total_wirelength(wl)
  {
    max_wirelength = outline_width + outline_height;
    total_netweight = std::accumulate(net_weights.begin(), net_weights.end(), 0);
  }
  float total_netweight;
  float max_wirelength;
  Wirelength<T> total_wirelength;
};

template <typename T>
struct EvalOutline
{
  template <typename Product>
  float operator()(const Product& product)
  {
    return this->operator()(product.width, product.height);
  }
  float operator()(T width, T height) { return std::max(1.f, width / outline_width) * std::max(1.f, height / outline_height) - 1.f; }
  EvalOutline(T outlineWidth, T outlineHeight) : outline_width(outlineWidth), outline_height(outlineHeight) {}
  float outline_width;
  float outline_height;
};

template <typename T>
struct EvalOutOfBound
{
  template <typename Product>
  float operator()(const Product& product)
  {
    double out_of_bound_cost = 0;
    const auto& dx = product.dx;
    const auto& dy = product.dy;
    const auto& lx = product.x;
    const auto& ly = product.y;
    const auto outline_ux = outline_lx + outline_height;
    const auto outline_uy = outline_ly + outline_height;
    for (size_t i = 0; i < product.dx.size(); ++i) {
      if (lx[i] + dx[i] > outline_ux || ly[i] + dy[i] > outline_uy) {
        out_of_bound_cost += (double(dx[i]) * dy[i]);
      }
    }
    out_of_bound_cost = out_of_bound_cost / outline_width / outline_height;
    // std::cout << "SAOUTOFBOUND:" << out_of_bound_cost << "\n";
    return out_of_bound_cost;
  }

  EvalOutOfBound(T outlineWidth, T outlineHeight, T outlineLX, T outlineLY)
      : outline_width(outlineWidth), outline_height(outlineHeight), outline_lx(outlineLX), outline_ly(outlineLY)
  {
  }
  T outline_width;
  T outline_height;
  T outline_lx;
  T outline_ly;
};

template <typename Code, typename Product>
struct Evaluator
{
  using Decoder = std::function<void(const Code&, Product&)>;
  using CostFunc = std::function<float(const Product&)>;
  double operator()(const Code& code)
  {
    decode(code, product);
    double cost = 0.0;
    for (size_t i = 0; i < cost_functions.size(); i++) {
      auto cost_i = cost_weights[i] * cost_functions[i](product) / cost_norm[i];
      cost += cost_i;
    }
    return cost;
  }
  Evaluator(const Decoder& decode_t, const std::initializer_list<double>& wl, const std::initializer_list<CostFunc>& fl)
      : decode(decode_t), cost_weights(wl), cost_functions(fl)
  {
    cost_norm.resize(cost_functions.size(), 1);
  }
  template <typename Perturb>
  void initalize(const Product& inital_product, Code inital_code, Perturb perturb, size_t num_perturb)
  {
    product = inital_product;
    std::vector<double> total_cost(cost_functions.size(), 0.f);
    for (size_t i = 0; i < num_perturb; i++) {
      perturb(inital_code);
      decode(inital_code, product);
      for (size_t j = 0; j < cost_functions.size(); j++) {
        total_cost[j] += cost_functions[j](product);
      }
    }
    for (size_t i = 0; i < cost_functions.size(); i++) {
      cost_norm[i] = std::max(total_cost[i] / num_perturb, 1.0);
    }
  }
  Decoder decode;
  std::vector<double> cost_weights;
  std::vector<CostFunc> cost_functions;

  Product product;
  std::vector<double> cost_norm;
};

}  // namespace imp
