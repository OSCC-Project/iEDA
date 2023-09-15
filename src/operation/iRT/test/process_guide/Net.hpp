class Net
{
 public:
  Net() = default;
  ~Net() = default;
  // getter
  std::string& get_net_name() { return _net_name; }
  std::vector<Guide>& get_guide_list() { return _guide_list; }
  // setter
  void set_net_name(const std::string& net_name) { _net_name = net_name; }
  void set_guide_list(const std::vector<Guide>& guide_list) { _guide_list = guide_list; }
  // function

 private:
  std::string _net_name;
  std::vector<Guide> _guide_list;
};