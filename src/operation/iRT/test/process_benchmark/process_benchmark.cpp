#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

class SingleCore
{
 public:
  SingleCore(std::string name, double area)
  {
    _name = name;
    _area = area;
  }
  ~SingleCore() = default;
  // getter
  std::string& get_name() { return _name; }
  double get_area() const { return _area; }
  double get_cell_num() const { return _cell_num; }
  double get_ff_num() const { return _ff_num; }
  double get_net_num() const { return _net_num; }
  // setter
  void set_name(const std::string& name) { _name = name; }
  void set_area(const double area) { _area = area; }
  void set_cell_num(const double cell_num) { _cell_num = cell_num; }
  void set_ff_num(const double ff_num) { _ff_num = ff_num; }
  void set_net_num(const double net_num) { _net_num = net_num; }

  // function

 private:
  std::string _name;
  double _area;
  double _cell_num;
  double _ff_num;
  double _net_num;
};

class MultiCore
{
 public:
  MultiCore(const double target_single_core_num, const double target_utilization_rate)
  {
    _target_single_core_num = target_single_core_num;
    _target_utilization_rate = target_utilization_rate;
  }
  ~MultiCore() = default;
  // getter
  double get_target_single_core_num() const { return _target_single_core_num; }
  double get_target_utilization_rate() const { return _target_utilization_rate; }
  std::string& get_name() { return _name; }
  std::vector<std::string>& get_single_core_name_list() { return _single_core_name_list; }
  double get_area() const { return _area; }
  double get_cell_num() const { return _cell_num; }
  double get_ff_num() const { return _ff_num; }
  double get_net_num() const { return _net_num; }
  double get_die_size() const { return _die_size; }
  // setter
  void set_target_single_core_num(const double target_single_core_num) { _target_single_core_num = target_single_core_num; }
  void set_target_utilization_rate(const double target_utilization_rate) { _target_utilization_rate = target_utilization_rate; }
  void set_name(const std::string& name) { _name = name; }
  void set_single_core_name_list(const std::vector<std::string>& single_core_name_list) { _single_core_name_list = single_core_name_list; }
  void set_area(const double area) { _area = area; }
  void set_cell_num(const double cell_num) { _cell_num = cell_num; }
  void set_ff_num(const double ff_num) { _ff_num = ff_num; }
  void set_net_num(const double net_num) { _net_num = net_num; }
  void set_die_size(const double die_size) { _die_size = die_size; }
  // function

 private:
  double _target_single_core_num;
  double _target_utilization_rate;
  std::string _name;
  std::vector<std::string> _single_core_name_list;
  double _area;
  double _cell_num;
  double _ff_num;
  double _net_num;
  double _die_size;
};

std::vector<SingleCore> getSingleCoreList()
{
  std::vector<SingleCore> single_core_list;
  single_core_list.emplace_back("u0_soc_top/u0_ysyx_210413", 0.18045468);
  single_core_list.emplace_back("u0_soc_top/u0_ysyx_210232", 0.185286547);
  single_core_list.emplace_back("u0_soc_top/u0_ysyx_210456", 0.199817111);
  single_core_list.emplace_back("u0_soc_top/u0_ysyx_210313", 0.204167539);
  single_core_list.emplace_back("ysyx_210528", 0.20559975);
  single_core_list.emplace_back("u0_soc_top/u2_ysyx_210428", 0.208610758);
  single_core_list.emplace_back("u0_soc_top/u1_ysyx_210191", 0.21065082);
  single_core_list.emplace_back("u0_soc_top/u6_ysyx_210669", 0.210808161);
  single_core_list.emplace_back("ysyx_210285", 0.215463859);
  single_core_list.emplace_back("ysyx_210727", 0.220440963);
  single_core_list.emplace_back("u0_soc_top/u3_ysyx_210247", 0.22073144);
  single_core_list.emplace_back("ysyx_210611", 0.221107);
  single_core_list.emplace_back("ysyx_210191", 0.221697);
  single_core_list.emplace_back("u0_soc_top/u4_ysyx_210295", 0.224764496);
  single_core_list.emplace_back("ysyx_210428", 0.225892);
  single_core_list.emplace_back("u0_soc_top/u0_ysyx_210340", 0.226262603);
  single_core_list.emplace_back("ysyx_210247", 0.229190233);
  single_core_list.emplace_back("ysyx_210417", 0.241648);
  single_core_list.emplace_back("ysyx_210133", 0.242081);
  single_core_list.emplace_back("u0_soc_top/u5_ysyx_210438", 0.243545973);
  single_core_list.emplace_back("ysyx_210243", 0.25043);
  single_core_list.emplace_back("u0_soc_top/u0_ysyx_210101", 0.255743309);
  single_core_list.emplace_back("ysyx_210302", 0.26443);
  single_core_list.emplace_back("ysyx_210292", 0.268804);
  single_core_list.emplace_back("ysyx_210195", 0.273613);
  single_core_list.emplace_back("ysyx_210457", 0.42334);
  single_core_list.emplace_back("ysyx_210134", 0.46017);
  return single_core_list;
}

void buildSingleCoreList(std::vector<SingleCore>& single_core_list)
{
  for (SingleCore& single_core : single_core_list) {
    single_core.set_cell_num(std::round(single_core.get_area() / 0.2 * 30000));
    single_core.set_ff_num(std::round(single_core.get_cell_num() / 8));
    single_core.set_net_num(single_core.get_cell_num());
  }
}

std::vector<MultiCore> getMultiCoreList()
{
  std::vector<MultiCore> multi_core_list;
  multi_core_list.emplace_back(1, 0.5);
  multi_core_list.emplace_back(1, 0.7);
  multi_core_list.emplace_back(2, 0.5);
  multi_core_list.emplace_back(2, 0.7);
  multi_core_list.emplace_back(4, 0.5);
  multi_core_list.emplace_back(4, 0.7);
  multi_core_list.emplace_back(8, 0.5);
  multi_core_list.emplace_back(8, 0.7);
  multi_core_list.emplace_back(12, 0.5);
  multi_core_list.emplace_back(12, 0.7);
  multi_core_list.emplace_back(20, 0.5);
  multi_core_list.emplace_back(20, 0.7);
  multi_core_list.emplace_back(27, 0.5);
  multi_core_list.emplace_back(27, 0.7);
  return multi_core_list;
}

void buildMultiCoreList(std::vector<MultiCore>& multi_core_list, std::vector<SingleCore>& single_core_list)
{
  std::vector<int32_t> random_idx_list;
  for (size_t i = 0; i < single_core_list.size(); i++) {
    random_idx_list.push_back(i);
  }
  for (size_t i = 0; i < multi_core_list.size(); i++) {
    MultiCore& multi_core = multi_core_list[i];
    multi_core.set_name(std::string("ysyx_") + std::to_string(i));

    std::vector<std::string> single_core_name_list;
    double area = 0;
    double cell_num = 0;
    double ff_num = 0;
    double net_num = 0;
    {
      srand(unsigned(time(NULL)));
      random_shuffle(random_idx_list.begin(), random_idx_list.end());

      size_t target_num = static_cast<size_t>(multi_core.get_target_single_core_num());
      size_t select_num = std::min(target_num, random_idx_list.size());

      for (size_t i = 0; i < select_num; i++) {
        SingleCore& single_core = single_core_list[random_idx_list[i]];
        single_core_name_list.push_back(single_core.get_name());
        area += single_core.get_area();
        cell_num += single_core.get_cell_num();
        ff_num += single_core.get_ff_num();
        net_num += single_core.get_net_num();
      }
    }
    multi_core.set_single_core_name_list(single_core_name_list);
    multi_core.set_area(area);
    multi_core.set_cell_num(cell_num);
    multi_core.set_ff_num(ff_num);
    multi_core.set_net_num(net_num);
    multi_core.set_die_size(std::sqrt(cell_num / multi_core.get_target_utilization_rate()));
  }
}

void writeCSV(const std::string& filename, const std::vector<std::vector<std::string>>& csv_data)
{
  std::ofstream file(filename);

  if (file.is_open()) {
    for (const auto& row : csv_data) {
      for (const auto& cell : row) {
        file << cell << ",";
      }
      file << "\n";
    }
    file.close();
    std::cout << "CSV file " << filename << " written successfully." << std::endl;
  } else {
    std::cout << "Unable to open file " << filename << std::endl;
  }
}

void printMultiCoreList(std::vector<MultiCore>& multi_core_list)
{
  std::vector<std::vector<std::string>> csv_data;

  std::vector<std::string> head_csv;
  head_csv.push_back("单核数");
  head_csv.push_back("利用率");
  head_csv.push_back("多核名");
  head_csv.push_back("单核列表");
  head_csv.push_back("面积");
  head_csv.push_back("估算cell个数");
  head_csv.push_back("估算FF个数");
  head_csv.push_back("估算net个数");
  head_csv.push_back("正方形die大小");
  csv_data.push_back(head_csv);

  for (MultiCore& multi_core : multi_core_list) {
    std::vector<std::string> value_csv;

    value_csv.push_back(std::to_string(static_cast<int32_t>(multi_core.get_target_single_core_num())));
    value_csv.push_back(std::to_string(multi_core.get_target_utilization_rate()));
    value_csv.push_back(multi_core.get_name());

    std::string single_core_name_list_string = "( ";
    for (std::string single_core_name : multi_core.get_single_core_name_list()) {
      single_core_name_list_string += single_core_name;
      single_core_name_list_string += " ";
    }
    single_core_name_list_string += ")";
    value_csv.push_back(single_core_name_list_string);

    value_csv.push_back(std::to_string(multi_core.get_area()));
    value_csv.push_back(std::to_string(static_cast<int32_t>(multi_core.get_cell_num())));
    value_csv.push_back(std::to_string(static_cast<int32_t>(multi_core.get_ff_num())));
    value_csv.push_back(std::to_string(static_cast<int32_t>(multi_core.get_net_num())));
    value_csv.push_back(std::to_string(static_cast<int32_t>(multi_core.get_die_size())));

    csv_data.push_back(value_csv);
  }

  writeCSV("/home/zengzhisheng/iEDA/aaa.csv", csv_data);
}

int32_t main()
{
  std::vector<SingleCore> single_core_list = getSingleCoreList();
  buildSingleCoreList(single_core_list);
  std::vector<MultiCore> multi_core_list = getMultiCoreList();
  buildMultiCoreList(multi_core_list, single_core_list);
  printMultiCoreList(multi_core_list);
  return 0;
}