# SALT

SALT (**S**teiner sh**A**llow-**L**ight **T**ree) is for generating VLSI routing topology.
It trades off between path length (shallowness) and wirelength (lightness).
More details are in [ICCAD'17](https://doi.org/10.1109/ICCAD.2017.8203828) paper.

Shallow | Light | SALT (shallow-light)
--------- | --------- | ---------
![rsa](/toys/RSA_toy1.tree.png) | ![flute](/toys/FLUTE_toy1.tree.png) | ![salt](/toys/SALT_toy1.tree.png)

## Quick Start

The simplest way to build and run SALT is as follows.
~~~
$ git clone https://github.com/chengengjie/salt
$ cd salt
$ ./scripts/build.py -o release
$ cd run
$ ./minimal_salt ../toys/toy1.net 1.0
~~~

To have a light installation of SALT in your own project, you only need to download folder `src/salt`.
See the [example](src/examples/minimal_main.cpp) for the usage. 

## Building SALT

**Step 1:** Download the source codes. For example,
~~~
$ git clone https://github.com/chengengjie/salt
~~~

**Step 2:** Go to the project root and build by
~~~
$ cd salt
$ ./scripts/build.py -o release
~~~

Note that this will generate two folders under the root, `build` and `run` (`build` contains intermediate files for build/compilation, while `run` contains binaries and auxiliary files).
More details are in `scripts/build.py`.

### Dependencies

* g++ (version >= 5.4.0) or other working c++ compliers
* CMake (version >= 3.5.1)
* Boost (version >= 1.58)
* Python (version 3, optional, for utility scripts)

## Runing SALT

### Toy

Go to the `run` directory and run binary `minimal_salt` with a toy net:
~~~
$ cd run
$ ./minimal_salt ../toys/toy1.net <epsilon>
~~~
The epsilon is the parameter controlling the trade-off between shallowness and lightness.
The output will be stored in file `SALT_toy1.tree`.
You can visualize it by
~~~
$ ../scripts/draw.py SALT_toy1.tree
~~~
Besides, to compare with some other methods (e.g., FLUTE, KRY, BRBC, PD, etc) as well as some other variants of SALT (e.g., without post processing), you may run binary `eval_single_salt`:
~~~
$ cd run
$ ./eval_single_salt -net ../toys/toy1.net -eps <epsilon>
~~~

### Batch Test

First, a file of input nets is needed.
The nets extracted from [ICCAD'15 Contest Problem B](https://doi.org/10.1109/ICCAD.2015.7372672) can be downloaded via [Dropbox](https://www.dropbox.com/sh/gcq1dh84ko9rjpz/AAAVT0pLZG_FMiOi0ORiKddva?dl=0).
For an input file, run binary `eval_batch_salt`:
~~~
$ cd run
$ ./eval_batch_salt <nets_file> <eval_file_suffix>
~~~
It constructs routing trees by several methods and epsilon values for each input net.
The evaluation statistics will be written into several files.
Each file summarizes the results for a specific range of # pins and a specific method, under various epsilon values and metrics (e.g., lightness, shallowness, delay, runtime, etc).

### Unit Test

Run the `build.py` with flag `-u` at the project root:
~~~
$ ./scripts/build.py -u
~~~

## Modules

* `scripts`: utility python scripts
* `src`: c++ source codes
    * `examples`: example application codes
    * `other_methods`: implemetation of other routing topology generation methods
    * `salt`: implementation of SALT
    * `unittest`: unit test
* `toys`: toy benchmarks

## File Formats

### Net

~~~
Net <net_id> <net_name> <pin_num> [-cap]
0 x0 y0 [cap0]
1 x1 y1 [cap1]
...
~~~
An example is [here](toys/toy1.net).

### Tree

~~~
Tree <net_id> <net_name> <pin_num> [-cap]
0 x0 y0 -1 [cap0]
1 x1 y1 parent_idx1 [cap1]
2 x2 y2 parent_idx2 [cap2]
...
k xk yk parent_idxk
...
~~~
An example is [here](toys/SALT_toy1.tree).
Note that tree nodes with indexes smaller than pin_num are pins, others are Steiner.
Also, Steiner nodes have no capacitance.