#pragma once
#include <string>
#include <vector>

#include "SAParam.hh"

using std::string;
using std::vector;

namespace ipl::imp {

enum class PartitionType
{
  KM,
  Metis
};

enum class SolutionTYPE
{
  BST,
  SP
};

class Setting : public SAParam
{
 public:
  Setting(){};
  ~Setting(){};

  // set
  void set_core_density(float density) { _core_density = density; }
  void set_new_macro_density(float density) { _new_macro_density = density; }
  void set_macro_halo_x(int halo) { _macro_halo_x = halo; }
  void set_macro_halo_y(int halo) { _macro_halo_y = halo; }
  void set_partition_type(PartitionType type) { _partition_type = type; }
  void set_parts(int parts) { _parts = parts; }
  void set_ncon(int ncon) { _ncon = ncon; }
  void set_ufactor(int ufactor) { _ufactor = ufactor; }
  void set_weight_area(float weight) { _weight_area = weight; }
  void set_weight_e_area(float weight) { _weight_e_area = weight; }
  void set_weight_wl(float weight) { _weight_wl = weight; }
  void set_weight_boundary(float weight) { _weight_boundary = weight; }
  void set_weight_notch(float weight) { _weight_notch = weight; }
  void set_weight_guidance(float weight) { _weight_guidance = weight; }
  void set_swap_pro(float pro) { _swap_pro = pro; }
  void set_move_pro(float pro) { _move_pro = pro; }
  void set_output_path(std::string path) { _output_path = path; }
  void set_solution_type(SolutionTYPE type) { _type = type; }

  // get
  float get_core_density() const { return _core_density; }
  float get_new_macro_density() const { return _new_macro_density; }
  uint32_t get_macro_halo_x() const { return _macro_halo_x; }
  uint32_t get_macro_halo_y() const { return _macro_halo_y; }
  PartitionType get_partition_type() const { return _partition_type; }
  int get_parts() const { return _parts; }
  int get_ncon() const { return _ncon; }
  int get_ufactor() const { return _ufactor; }
  float get_weight_area() const { return _weight_area; }
  float get_weight_e_area() const { return _weight_e_area; }
  float get_weight_boundary() const { return _weight_boundary; }
  float get_weight_notch() const { return _weight_notch; }
  float get_weight_wl() const { return _weight_wl; }
  float get_weight_guidance() const { return _weight_guidance; }
  float get_swap_pro() const { return _swap_pro; }
  float get_move_pro() const { return _move_pro; }
  std::string get_output_path() const { return _output_path; }
  SolutionTYPE get_solution_type() const { return _type; }

 private:
  // macroplacer
  float _core_density;
  float _new_macro_density;
  uint32_t _macro_halo_x;
  uint32_t _macro_halo_y;

  // partition
  PartitionType _partition_type = PartitionType::Metis;
  int _parts = 16;  // the number of cluster
  int _ncon = 5;    // The number of balancing constraints
  int _ufactor = 400;

  // simulate anneal
  float _weight_area = 1;
  float _weight_e_area = 5;
  float _weight_wl = 3;  // wire length
  float _weight_boundary = 1;
  float _weight_notch = 1;
  float _weight_guidance = 3;  // guidance
  // B* tree
  float _swap_pro = 0.5;  // the probability of swap
  float _move_pro = 0.5;  // the probability of move
  std::string _output_path = ".";

  SolutionTYPE _type = SolutionTYPE::BST;
};

}  // namespace ipl::imp