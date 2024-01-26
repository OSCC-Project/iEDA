#include <chrono>
#include <iostream>
#include <random>

#include "Evaluator.hh"
#include "Hpwl.hh"
#include "Operator.hh"
#include "SeqPair.hh"
#include "SimulateAnneal.hh"

struct RotateStatus
{
  bool is_rotate = false;
};

struct Coordinate
{
  int32_t width{0};
  int32_t height{0};
  std::vector<int32_t> x{};
  std::vector<int32_t> y{};
};

using namespace imp;
int main()
{
  auto bench_begin = std::chrono::high_resolution_clock::now();

  size_t N = 300;
  size_t E = 2000;
  size_t min_degree = 2;
  int32_t mean_blk_area = 100;
  float min_ar = 1.f / 3.f, max_ar = 3.f;  // ar = h / w
  float outline_ar = 1;
  float utilization = 0.8;

  int seed = std::random_device()();  // set random seed

  std::mt19937 gen1(seed);
  std::mt19937 gen2(seed);
  std::mt19937 gen3(seed);
  std::mt19937 gen4(seed);
  std::mt19937 gen5(seed);
  std::mt19937 gen6(seed);
  std::poisson_distribution<size_t> rand_degree(min_degree);
  std::uniform_int_distribution<size_t> dis(0, N - 1);
  std::uniform_real_distribution<float> blk_ar(min_ar, max_ar);
  std::normal_distribution<float> blk_area(mean_blk_area, mean_blk_area / 10);

  std::vector<RotateStatus> rotate(N, {false});
  std::vector<bool> ignore(N, false);
  std::vector<int32_t> blk_widths(N);
  std::vector<int32_t> blk_heights(N);
  std::vector<int32_t> inital_lx(N, 0);
  std::vector<int32_t> inital_ly(N, 0);
  int32_t total_area = 0;

  // Random generate shape of each blk
  for (size_t i = 0; i < N; i++) {
    float ar = blk_ar(gen1);
    float area = blk_area(gen2);
    blk_heights[i] = std::round(std::sqrt(ar * area));
    blk_widths[i] = std::round(area / (float) blk_heights[i]);
    total_area += blk_widths[i] * blk_heights[i];
  }
  ignore[0] = true;  // Ignore the first blk;

  // Define shape of outline
  int32_t outline_area = std::round(total_area / utilization);
  int32_t outline_height = std::round(std::sqrt(outline_area * outline_ar));
  int32_t outline_width = std::round(outline_area / (float) outline_height);
  int32_t outline_lx = 10;
  int32_t outline_ly = 10;

  // Random genertate net
  std::vector<size_t> eptr(E + 1);
  size_t cur = 0;
  std::generate(eptr.begin(), eptr.end(), [&]() { return cur += rand_degree(gen3); });
  std::vector<size_t> eind(eptr.back());
  std::generate(eind.begin(), eind.end(), [&]() { return dis(gen4); });
  std::vector<int32_t> net_weight(E, 1);

  // Decoder products must contain .x, .y .width .height properties.
  Coordinate coor{.x = inital_lx, .y = inital_ly};


  auto bench_end = std::chrono::high_resolution_clock::now();
  std::cout << "Random Bench time: " << std::chrono::duration<double>(bench_end - bench_begin).count() << "s\n";

  SeqPair<RotateStatus> sp(rotate, gen6);

  using DimFunc = FastPackSP<RotateStatus, Coordinate>::DimFunc;
  using IgnoreFunc = FastPackSP<RotateStatus, Coordinate>::IgnoreFunc;

  DimFunc get_blk_width = [&](size_t i, const RotateStatus& b) { return b.is_rotate ? blk_heights[i] : blk_widths[i]; };
  DimFunc get_blk_height = [&](size_t i, const RotateStatus& b) { return b.is_rotate ? blk_widths[i] : blk_heights[i]; };
  IgnoreFunc is_ignore = [&](size_t i, const RotateStatus&) { return ignore[i]; };

  // packing function to get coordinate
  FastPackSP<RotateStatus, Coordinate> pack(outline_lx, outline_ly, get_blk_width, get_blk_height, is_ignore);

  using Decoder = Evaluator<SeqPair<RotateStatus>, Coordinate>::Decoder;
  using CostFunc = Evaluator<SeqPair<RotateStatus>, Coordinate>::CostFunc;

  Decoder decoder = pack;

  Hpwl<int32_t> hpwl(eptr, eind, {}, {}, {}, 1);
  CostFunc wl = EvalWirelength(outline_width, outline_height, hpwl, net_weight);
  CostFunc ol = EvalOutline(outline_width, outline_height);

  // Self-difine eval fucntion
  CostFunc area
      = [&](const Coordinate& coor) { return (float) coor.width * (float) coor.height / (float) outline_width / (float) outline_height; };

  // combination eval fucntiion using self-define weight
  Evaluator<SeqPair<RotateStatus>, Coordinate> eval(decoder, {0.1, 1, 1}, {area, wl, ol});

  using PerturbFunc = Perturb<SeqPair<RotateStatus>, void>::PerturbFunc;

  PerturbFunc ps_op = PosSwap();
  PerturbFunc ns_op = NegSwap();
  PerturbFunc ds_op = DoubleSwap();
  PerturbFunc pi_op = PosInsert();
  PerturbFunc ni_op = NegInsert();
  PerturbFunc rotate_op = [](SeqPair<RotateStatus>& sp, std::mt19937& gen) {
    std::uniform_int_distribution<size_t> get_ridx(0, sp.size - 1);
    size_t id = get_ridx(gen);
    sp.properties[id].is_rotate = !sp.properties[id].is_rotate;
  };

  Perturb<SeqPair<RotateStatus>, void> perturb(seed, {0.3, 0.3, 0.3, 0.0}, {ps_op, ns_op, ds_op, rotate_op});

  // intalize the norm cost and inital product
  eval.initalize(coor, sp, perturb, N * 15 / 10);

  SimulateAnneal solve{.seed = seed, .num_perturb = N * 15 / 10, .cool_rate = 0.92, .inital_temperature = 1000};

  auto sa_start = std::chrono::high_resolution_clock::now();

  sp = solve(sp, eval, perturb, [](auto&& x) { std::cout << x << std::endl; });

  auto sa_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> sa_elapsed = std::chrono::duration<double>(sa_end - sa_start);
  std::cout << "SA time: " << sa_elapsed.count() << "s\n";
  return 0;
}