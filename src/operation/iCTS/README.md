# iCTS: Clock Tree Synthesis


## 1. Overview

Currently, iCTS supports clock tree construction under given constraints. Currently, iCTS takes DME and Slew-Aware as the basic framework and integrates multiple types of constraints for clock tree design. Due to the inability to obtain upstream Slew information in the bottom-up process of DME,
The downstream chained propagation will bring a large error. iCTS introduces the timing backward propagation method and performs timing propagation at the wire level during the buffer insertion stage to correct the insertion delay of the wire load unit.
iCTS uses a more accurate timing model for the estimation and propagation of timing information and constructs the clock tree with the goal of minimizing the design cost.

### Constraints

The constraints that iCTS can support are shown in the following table, where the constraint names correspond to the Config parameter names

| Constraint Name | Constraint Level |
| :-------------: | :--------------: |
|   skew_bound   |      Hard        |
|  max_buf_tran  |      Hard        |
|  max_sink_tran |      Hard        |
|     max_cap    |      Soft        |
|   max_fanout   |      Hard        |
|   max_length   |      Soft        |

### Timing Models

iCTS currently uses the PERI interconnect model as the slew calculation method for interconnects and the Elmore model as the delay calculation method for interconnects. For the insertion delay of buffers, the lookup table method (Lut) is used, which includes extensions of some methods, as shown in the following table.

| Model |                         Scenario                          |                        Extension                        |
| :---: | :------------------------------------------------------: | :------------------------------------------------------: |
|  PERI |          Calculation and propagation of interconnect slew          |          Formula calculation, timing backward propagation correction          |
| Elmore |              Calculation of interconnect delay              |                   Formula calculation                   |
| Unit RC | Conversion of the load capacitance and resistance of interconnects based on the unit capacitance and resistance |                   Formula calculation                   |
|   Lut  |     Calculation of buffer insertion delay and unit slew propagation     | Bilinear interpolation method, multiple linear fitting model, machine learning model |

### Goals

Under the premise of meeting the design constraints as much as possible, for scenarios with multiple buffering schemes, iCTS will adopt the scheme with the minimum design cost for clock tree design. The current measurement methods are as follows:

* Prefer the buffering scheme with a smaller size that meets the timing constraints.
* For the case of continuous buffer insertion to balance the clock skew, in order to simplify the complexity of timing calculation and propagation, consider using the same buffer and perform timing balancing at a uniform wire length interval. At this stage, the buffering scheme that minimizes the increase in cell area is preferred.

## 2. Multi-Clock Balance

iCTS can support multi-clock design. First, a clock tree is constructed for each clock one by one to form the basic clock tree result. The clock skew of the top-level node is analyzed through the timing evaluation of iSTA, and finally, timing balancing is performed according to the design requirements.

## 3. Support Extensions

### Multiple Buffer Types

iCTS supports clock tree design based on multiple buffer types (currently excluding inverters) and selects buffers using the strategy in [Goals](#Goals) during the buffer insertion process. This feature can be set in the `buffer_type` parameter in Config.

### Clustering to Reduce Scale

iCTS supports using the clustering method (K-Means) to reduce the running time of large-scale nets. For nets with more than 3000 registers, iCTS will automatically execute the K-Means algorithm and divide them into 50 clusters for local clock tree construction. After completing the local construction, the top-level clock tree is merged. Both the local construction and the top-level clock tree construction use the same clock tree algorithm.

During the K-Means clustering process, we set the initial number of iterations. In each iteration, we record the total register load capacitance of each cluster and consider the clustering result with a smaller variance of capacitance between clusters for clock tree construction.

### Machine Learning Models

iCTS supports the invocation of Python models and methods based on Cython. Currently, the encapsulated Linear, XGBoost, CatBoost models, and basic plotting methods of Matplotlib are available. You can specify the compilation option `SYS_PYTHON3_VERSION` to specify the system Python version,
And by turning on the `PY_MODEL` option, C++ and Python interaction can be achieved. In this mode, the timing-related Lut process will use machine learning models. 