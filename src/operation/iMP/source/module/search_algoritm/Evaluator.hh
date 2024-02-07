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

template <typename Code, typename Product>
struct Evaluator
{
  using Decoder = std::function<void(const Code&, Product&)>;
  using CostFunc = std::function<float(const Product&)>;
  float operator()(const Code& code)
  {
    decode(code, product);
    float cost = 0.f;
    for (size_t i = 0; i < cost_functions.size(); i++) {
      cost += cost_weights[i] * cost_functions[i](product) / cost_norm[i];
    }
    return cost;
  }
  Evaluator(const Decoder& decode_t, const std::initializer_list<float>& wl, const std::initializer_list<CostFunc>& fl)
      : decode(decode_t), cost_weights(wl), cost_functions(fl)
  {
    cost_norm.resize(cost_functions.size(), 1);
  }
  template <typename Perturb>
  void initalize(const Product& inital_product, Code inital_code, Perturb perturb, size_t num_perturb)
  {
    product = inital_product;
    std::vector<float> total_cost(cost_functions.size(), 0.f);
    for (size_t i = 0; i < num_perturb; i++) {
      perturb(inital_code);
      decode(inital_code, product);
      for (size_t j = 0; j < cost_functions.size(); j++) {
        total_cost[j] += cost_functions[j](product);
      }
    }
    for (size_t i = 0; i < cost_functions.size(); i++) {
      cost_norm[i] = std::max(total_cost[i] / num_perturb, 1.f);
    }
  }
  Decoder decode;
  std::vector<float> cost_weights;
  std::vector<CostFunc> cost_functions;

  Product product;
  std::vector<float> cost_norm;
};


}  // namespace imp
