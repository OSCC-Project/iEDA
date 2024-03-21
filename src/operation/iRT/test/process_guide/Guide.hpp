class Guide
{
 public:
  Guide() = default;
  ~Guide() = default;
  // getter
  int get_ll_x() const { return _ll_x; }
  int get_ll_y() const { return _ll_y; }
  int get_ur_x() const { return _ur_x; }
  int get_ur_y() const { return _ur_y; }
  std::string& get_layer_name() { return _layer_name; }
  // setter
  void set_ll_x(const int ll_x) { _ll_x = ll_x; }
  void set_ll_y(const int ll_y) { _ll_y = ll_y; }
  void set_ur_x(const int ur_x) { _ur_x = ur_x; }
  void set_ur_y(const int ur_y) { _ur_y = ur_y; }
  void set_layer_name(const std::string& layer_name) { _layer_name = layer_name; }
  // function

 private:
  int _ll_x;
  int _ll_y;
  int _ur_x;
  int _ur_y;
  std::string _layer_name;
};