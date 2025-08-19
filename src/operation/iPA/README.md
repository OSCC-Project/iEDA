# iPA: Power Analysis

## Introduction to the ipower

### Software structure diagram

<img src="../../../docs/resources/iPower.png" width="100%" height="35%" alt="iPower-structure" />

### Supported features

* Support read vcd file.
* Support output power analysis report, including different power type (internal power, switch power and leakage power) and different power group(clock_network, combinational, sequential,etc.)

## Example of how to use the iPower tool

### Write the tcl file(run_ipw.tcl)

* example tcl file：../source/data/example1/run_ipw.tcl

According to iSTA README.md, read verilog, .lib file, link design, read sdc, spef file.

#### Read vcd file

```
read_vcd test.vcd -top_name top-i
```

#### Get the power report

```
report_power
```

## Run the tcl file with iPower

```bash
 ./iPower run_ipwr.tcl
```
