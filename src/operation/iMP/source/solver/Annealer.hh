/**
 * @file Annealer.hh
 * @author Fuxing Huang (fxxhuang@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-07-11
 *
 * @copyright Copyright (c) 2023 PCNL
 *
 */
#ifndef IMP_ANNEALER_H
#define IMP_ANNEALER_H
#include <cstdint>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "Logger.hpp"
namespace imp {
struct SAOption
{
  int max_iters = 500;
  int num_operates = 60;
  double cool_rate = 0.92;
  double init_pro = 0.95;
  double start_temperature = 1000;
};

class SASolution
{
 public:
  SASolution() = default;
  virtual void operate() = 0;
  virtual void rollback() = 0;
  virtual void update() = 0;
  virtual double evaluate() = 0;
};
class SimulateAnneal
{
 public:
  static bool solve(SASolution& solution, const SAOption& opt);

 private:
  SimulateAnneal() = delete;
  ~SimulateAnneal() = delete;
};

template <typename SolutionType>
class SATask
{
 public:
  SATask(const SolutionType& solution, const std::function<double(SolutionType&)> evaluate, const std::function<void(SolutionType&)> action,
         int max_iters, int num_actions, double cool_rate, double temperature)
      : _solution(solution),
        _evaluate(evaluate),
        _action(action),
        _max_iters(max_iters),
        _num_actions(num_actions),
        _cool_rate(cool_rate),
        _temperature(temperature)
  {
  }
  // getter
  SolutionType& get_solution() { return _solution; }
  std::function<double(SolutionType&)>& get_evaluate() { return _evaluate; }
  std::function<void(SolutionType&)>& get_action() { return _action; }
  int get_max_iters() const { return _max_iters; }
  int get_num_actions() const { return _num_actions; }
  double get_cool_rate() const { return _cool_rate; }
  double get_temperature() const { return _temperature; }

  // setter
  void set_solution(const SolutionType& solution) { _solution = solution; }
  void set_evaluate(const std::function<double(SolutionType&)>& evaluate) { _evaluate = evaluate; }
  void set_action(const std::function<void(SolutionType&)>& action) { _action = action; }
  void set_max_iters(int max_iters) { _max_iters = max_iters; }
  void set_num_actions(int num_actions) { _num_actions = num_actions; }
  void set_cool_rate(double cool_rate) { _cool_rate = cool_rate; }
  void set_temperature(double temperature) { _temperature = temperature; }

 private:
  SolutionType _solution;
  std::function<double(SolutionType&)> _evaluate;
  std::function<void(SolutionType&)> _action;
  int _max_iters;
  int _num_actions;
  double _cool_rate;
  double _temperature;
};

template <typename SolutionType>
std::vector<SolutionType> SASolve(SATask<SolutionType>& sa_task)
{
  return SASolve<SolutionType>(sa_task.get_solution(), sa_task.get_evaluate(), sa_task.get_action(), sa_task.get_max_iters(),
                               sa_task.get_num_actions(), sa_task.get_cool_rate(), sa_task.get_temperature());
}

template <typename SolutionType>
std::vector<SolutionType> SASolve(SolutionType& solution, std::function<double(SolutionType&)> evaluate,
                                  std::function<void(SolutionType&)> action, int max_iters, int num_actions, double cool_rate,
                                  double temperature = 10000)
{
  std::vector<SolutionType> historys;
  double inital_temp = temperature;
  double cur_cost = evaluate(solution);
  double last_cost = 0.;
  double temp_cost{0.}, delta_cost{0.}, random{0.};

  std::random_device r;
  std::mt19937 e1(r());
  std::uniform_real_distribution<double> real_rand(0., 1.);
  // fast sa
  SolutionType solution_t = solution;
  for (int iter = 0; iter < max_iters && temperature >= 0.01; ++iter) {
    last_cost = cur_cost;
    for (int times = 0; times < num_actions; ++times) {
      action(solution_t);
      temp_cost = evaluate(solution_t);
      delta_cost = temp_cost - cur_cost;
      random = real_rand(e1);
      if (exp(-delta_cost * inital_temp / temperature) > random) {
        solution = solution_t;
        cur_cost = temp_cost;
      } else {
        solution_t = solution;
      }
    }
    INFO("iter: ", iter, " temperature: ", temperature, " cost: ", cur_cost, " dis: ", cur_cost - last_cost);
    temperature *= cool_rate;
    if ((cur_cost - last_cost) != 0)
      historys.push_back(solution);
  }

  return historys;
}

}  // namespace imp

#endif