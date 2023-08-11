Coding style guidelines for UG
------------------------------

1. Source file directories and source file names
   Source files are located in organized directories in the “src” directory as follows.
    - ug: Source files of top abstracted base classes of UG
    - ug_*: There are two cases:
      1) Source files of abstracted base classes for a parallel framework developed on top of UG, for example, ug_bb, ug_cmaplap,
      2) Source files for interface classes for a specific branch-and-bound based “base solvers”,
         since UG was originally developed for parallel branch-and-bound, for example, ug_scip, ug_xpress.
   Source files in the directory ug_* are named as follows by using ug_bb as an example:
    - File names start with a lowercase letter with the second part of the directory name separated by ‘_’ and
      the name followed by "Para" plus some meaningful name connected by an uppercase letter. For example, bbParaSolverPool.h and bbParaSolverPool.cpp.
    - File names of the communicator objects are bbParaComm plus some meaningful name for the communicator. For example, bbParaCommMpi.h and bbParaCommMpi.cpp.
    - File names for transferred objects, which are transferred between the LoadCorrdinator object and the Solver object, are ended with “Th” if it is for shared memory version,
      or, are ended with “Mpi” if it is for distributed memory. For example, bbParaNodeTh.h, bbParaNodeTh.cpp, bbParaNodeMpi.h, bbParaNodeMpi.cpp.

2. Source file structure
   A source file consists of, in order:
   1) License and copyright information
   2) include statement
   3) program codes for classes and functions
   Exactly one blank line separating each section that is present.
   Ordering of class contents is as follows:
   1) private members
   2) protected members
   3) public members

3. Formating
  1) Spacing
  - Indentation is 3 spaces. No tabs anywhere in the code.
  - Every opening parenthesis requires an additional indentation of 3 spaces.
  - Spaces around all operators.
  - Spaces around the arguments inside an if/for/while-statement, as well as inside macros.
  - No spaces between control structure keywords like "if", "for", "while", "switch" and the corresponding brackets.
  - No spaces between a function name and the parenthesis in both the definition and function calls.
  - Braces are on a new line and not indented.
  - In function declarations, every parameter is on a new line.
  - Always only one declaration in a line.
  - Blank lines are inserted where it improves readability.
  - Multiple blank lines are used to structure the code where single blank lines are insufficient, e.g., between differrent sections of the code.
  2) Naming
  - Class names start with an uppercase letter (Heading starts with sencond part of directory name)
  - Variable names start with a lowercase letter
  - Use assert() to show preconditions for the parameters, invariants, and postconditions.
  - For each structure there is a typedef with the name in all upper case.
  - Defines should be named all upper case.
  3) Documentation
  - Document functions, parameters, and variables in a doxygen conformed way.

