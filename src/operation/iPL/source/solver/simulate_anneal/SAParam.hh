#pragma once

#include <stdint.h>

namespace ipl {
class SAParam
{
 public:
  SAParam(){};
  ~SAParam(){};

  void set_max_num_step(uint32_t step) { _max_num_step = step; }
  void set_perturb_per_step(uint32_t step) { _perturb_per_step = step; }
  void set_cool_rate(float rate) { _cool_rate = rate; }
  void set_init_pro(float pro) { _init_pro = pro; }
  void set_init_temperature(float temperatrure) { _init_temperature = temperatrure; }

  uint32_t get_max_num_step() { return _max_num_step; }
  uint32_t get_perturb_per_step() { return _perturb_per_step; }
  float get_cool_rate() { return _cool_rate; }
  float get_init_pro() { return _init_pro; }
  float get_init_temperature() { return _init_temperature; }

 protected:
  uint32_t _max_num_step = 100;
  uint32_t _perturb_per_step = 60;
  float _cool_rate = 0.92;
  float _init_pro = 0.95;
  float _init_temperature = 1000;  // default 30000
};
}  // namespace ipl
