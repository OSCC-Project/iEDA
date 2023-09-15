class Guide
{
 public:
  Guide() = default;
  ~Guide() = default;
  // getter
  int get_lb_x() const { return _lb_x; }
  int get_lb_y() const { return _lb_y; }
  int get_rt_x() const { return _rt_x; }
  int get_rt_y() const { return _rt_y; }
  std::string& get_layer_name() { return _layer_name; }
  // setter
  void set_lb_x(const int lb_x) { _lb_x = lb_x; }
  void set_lb_y(const int lb_y) { _lb_y = lb_y; }
  void set_rt_x(const int rt_x) { _rt_x = rt_x; }
  void set_rt_y(const int rt_y) { _rt_y = rt_y; }
  void set_layer_name(const std::string& layer_name) { _layer_name = layer_name; }
  // function

 private:
  int _lb_x;
  int _lb_y;
  int _rt_x;
  int _rt_y;
  std::string _layer_name;
};