/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the class library                   */
/*       SoPlex --- the Sequential object-oriented simPlex.                  */
/*                                                                           */
/*  Copyright 1996-2022 Zuse Institute Berlin                                */
/*                                                                           */
/*  Licensed under the Apache License, Version 2.0 (the "License");          */
/*  you may not use this file except in compliance with the License.         */
/*  You may obtain a copy of the License at                                  */
/*                                                                           */
/*      http://www.apache.org/licenses/LICENSE-2.0                           */
/*                                                                           */
/*  Unless required by applicable law or agreed to in writing, software      */
/*  distributed under the License is distributed on an "AS IS" BASIS,        */
/*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. */
/*  See the License for the specific language governing permissions and      */
/*  limitations under the License.                                           */
/*                                                                           */
/*  You should have received a copy of the Apache-2.0 license                */
/*  along with SoPlex; see the file LICENSE. If not email to soplex@zib.de.  */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file  soplexmain.cpp
 * @brief Command line interface of SoPlex LP solver
 */

#include <assert.h>
#include <math.h>
#include <string.h>

#include <iostream>
#include <iomanip>
#include <fstream>

#include "soplex.h"
#include "soplex/validation.h"

using namespace soplex;

// function prototype
int main(int argc, char* argv[]);

// prints usage and command line options
static
void printUsage(const char* const argv[], int idx)
{
   const char* usage =
      "general options:\n"
      "  --readbas=<basfile>    read starting basis from file\n"
      "  --writebas=<basfile>   write terminal basis to file\n"
      "  --writefile=<lpfile>   write LP to file in LP or MPS format depending on extension\n"
      "  --writedual=<lpfile>   write the dual LP to a file in LP or MPS formal depending on extension\n"
      "  --<type>:<name>=<val>  change parameter value using syntax of settings file entries\n"
      "  --loadset=<setfile>    load parameters from settings file (overruled by command line parameters)\n"
      "  --saveset=<setfile>    save parameters to settings file\n"
      "  --diffset=<setfile>    save modified parameters to settings file\n"
      "  --extsol=<value>       external solution for soplex to use for validation\n"
      "\n"
      "limits and tolerances:\n"
      "  -t<s>                  set time limit to <s> seconds\n"
      "  -i<n>                  set iteration limit to <n>\n"
      "  -f<eps>                set primal feasibility tolerance to <eps>\n"
      "  -o<eps>                set dual feasibility (optimality) tolerance to <eps>\n"
      "  -l<eps>                set validation tolerance to <eps>\n"
      "\n"
      "algorithmic settings (* indicates default):\n"
      "  --readmode=<value>     choose reading mode for <lpfile> (0* - floating-point, 1 - rational)\n"
      "  --solvemode=<value>    choose solving mode (0 - floating-point solve, 1* - auto, 2 - force iterative refinement)\n"
      "  --arithmetic=<value>   choose base arithmetic type (0 - double, 1 - quadprecision, 2 - higher multiprecision)\n"
#ifdef SOPLEX_WITH_MPFR
      "  --precision=<value>    choose precision for multiprecision solve (only active when arithmetic=2 minimal value = 50)\n"
#endif
#ifdef SOPLEX_WITH_CPPMPF
      "  --precision=<value>    choose precision for multiprecision solve (only active when arithmetic=2, possible values 50,100,200, compile with mpfr for arbitrary precision)\n"
#endif
      "  -s<value>              choose simplifier/presolver (0 - off, 1* - internal, 2*- PaPILO)\n"
      "  -g<value>              choose scaling (0 - off, 1 - uni-equilibrium, 2* - bi-equilibrium, 3 - geometric, 4 - iterated geometric, 5 - least squares, 6 - geometric-equilibrium)\n"
      "  -p<value>              choose pricing (0* - auto, 1 - dantzig, 2 - parmult, 3 - devex, 4 - quicksteep, 5 - steep)\n"
      "  -r<value>              choose ratio tester (0 - textbook, 1 - harris, 2 - fast, 3* - boundflipping)\n"
      "\n"
      "display options:\n"
      "  -v<level>              set verbosity to <level> (0 - error, 3 - normal, 5 - high)\n"
      "  -x                     print primal solution\n"
      "  -y                     print dual multipliers\n"
      "  -X                     print primal solution in rational numbers\n"
      "  -Y                     print dual multipliers in rational numbers\n"
      "  -q                     display detailed statistics\n"
      "  -c                     perform final check of optimal solution in original problem\n"
      "\n";

   if(idx <= 0)
      std::cerr << "missing input file\n\n";
   else
      std::cerr << "invalid option \"" << argv[idx] << "\"\n\n";

   std::cerr << "usage: " << argv[0] << " " << "[options] <lpfile>\n"
#ifdef SOPLEX_WITH_ZLIB
             << "  <lpfile>               linear program as .mps[.gz] or .lp[.gz] file\n\n"
#else
             << "  <lpfile>               linear program as .mps or .lp file\n\n"
#endif
             << usage;
}

// cleans up C strings
static
void freeStrings(char*& s1, char*& s2, char*& s3, char*& s4, char*& s5)
{
   if(s1 != 0)
   {
      delete [] s1;
      s1 = 0;
   }

   if(s2 != 0)
   {
      delete [] s2;
      s2 = 0;
   }

   if(s3 != 0)
   {
      delete [] s3;
      s3 = 0;
   }

   if(s4 != 0)
   {
      delete [] s4;
      s4 = 0;
   }

   if(s5 != 0)
   {
      delete [] s5;
      s5 = 0;
   }
}

/// performs external feasibility check with real type
///@todo implement external check; currently we use the internal methods for convenience

template <class R>
static
void checkSolutionReal(SoPlexBase<R>& soplex)
{
   if(soplex.hasPrimal())
   {
      R boundviol;
      R rowviol;
      R sumviol;

      if(soplex.getBoundViolation(boundviol, sumviol) && soplex.getRowViolation(rowviol, sumviol))
      {
         MSG_INFO1(soplex.spxout,
                   R maxviol = boundviol > rowviol ? boundviol : rowviol;
                   bool feasible = (maxviol <= soplex.realParam(SoPlexBase<R>::FEASTOL));
                   soplex.spxout << "Primal solution " << (feasible ? "feasible" : "infeasible")
                   << " in original problem (max. violation = " << std::scientific << maxviol
                   << std::setprecision(8) << std::fixed << ").\n");
      }
      else
      {
         MSG_INFO1(soplex.spxout, soplex.spxout << "Could not check primal solution.\n");
      }
   }
   else
   {
      MSG_INFO1(soplex.spxout, soplex.spxout << "No primal solution available.\n");
   }

   if(soplex.hasDual())
   {
      R redcostviol;
      R dualviol;
      R sumviol;

      if(soplex.getRedCostViolation(redcostviol, sumviol) && soplex.getDualViolation(dualviol, sumviol))
      {
         MSG_INFO1(soplex.spxout,
                   R maxviol = redcostviol > dualviol ? redcostviol : dualviol;
                   bool feasible = (maxviol <= soplex.realParam(SoPlexBase<R>::OPTTOL));
                   soplex.spxout << "Dual solution " << (feasible ? "feasible" : "infeasible")
                   << " in original problem (max. violation = " << std::scientific << maxviol
                   << std::setprecision(8) << std::fixed << ").\n"
                  );
      }
      else
      {
         MSG_INFO1(soplex.spxout, soplex.spxout << "Could not check dual solution.\n");
      }
   }
   else
   {
      MSG_INFO1(soplex.spxout, soplex.spxout << "No dual solution available.\n");
   }
}

/// performs external feasibility check with rational type
///@todo implement external check; currently we use the internal methods for convenience
template <class R>
static void checkSolutionRational(SoPlexBase<R>& soplex)
{
   if(soplex.hasPrimal())
   {
      Rational boundviol;
      Rational rowviol;
      Rational sumviol;

      if(soplex.getBoundViolationRational(boundviol, sumviol)
            && soplex.getRowViolationRational(rowviol, sumviol))
      {
         MSG_INFO1(soplex.spxout,
                   Rational maxviol = boundviol > rowviol ? boundviol : rowviol;
                   bool feasible = (maxviol <= soplex.realParam(SoPlexBase<R>::FEASTOL));
                   soplex.spxout << "Primal solution " << (feasible ? "feasible" : "infeasible") <<
                   " in original problem (max. violation = " << maxviol << ").\n"
                  );
      }
      else
      {
         MSG_INFO1(soplex.spxout, soplex.spxout << "Could not check primal solution.\n");
      }
   }
   else
   {
      MSG_INFO1(soplex.spxout, soplex.spxout << "No primal solution available.\n");
   }

   if(soplex.hasDual())
   {
      Rational redcostviol;
      Rational dualviol;
      Rational sumviol;

      if(soplex.getRedCostViolationRational(redcostviol, sumviol)
            && soplex.getDualViolationRational(dualviol, sumviol))
      {
         MSG_INFO1(soplex.spxout,
                   Rational maxviol = redcostviol > dualviol ? redcostviol : dualviol;
                   bool feasible = (maxviol <= soplex.realParam(SoPlexBase<R>::OPTTOL));
                   soplex.spxout << "Dual solution " << (feasible ? "feasible" : "infeasible") <<
                   " in original problem (max. violation = " << maxviol << ").\n"
                  );
      }
      else
      {
         MSG_INFO1(soplex.spxout, soplex.spxout << "Could not check dual solution.\n");
      }
   }
   else
   {
      MSG_INFO1(soplex.spxout, soplex.spxout << "No dual solution available.\n");
   }
}

/// performs external feasibility check according to check mode
template <class R>
void checkSolution(SoPlexBase<R>& soplex)
{
   if(soplex.intParam(SoPlexBase<R>::CHECKMODE) == SoPlexBase<R>::CHECKMODE_RATIONAL
         || (soplex.intParam(SoPlexBase<R>::CHECKMODE) == SoPlexBase<R>::CHECKMODE_AUTO
             && soplex.intParam(SoPlexBase<R>::READMODE) == SoPlexBase<R>::READMODE_RATIONAL))
   {
      checkSolutionRational(soplex);
   }
   else
   {
      checkSolutionReal(soplex);
   }

   MSG_INFO1(soplex.spxout, soplex.spxout << "\n");
}

template <class R>
static
void printPrimalSolution(SoPlexBase<R>& soplex, NameSet& colnames, NameSet& rownames,
                         bool real = true, bool rational = false)
{
   int printprec;
   int printwidth;
   printprec = (int) - log10(Real(Param::epsilon()));
   printwidth = printprec + 10;

   if(real)
   {
      VectorBase<R> primal(soplex.numCols());

      if(soplex.getPrimalRay(primal))
      {
         MSG_INFO1(soplex.spxout, soplex.spxout << "\nPrimal ray (name, value):\n";)

         for(int i = 0; i < soplex.numCols(); ++i)
         {
            if(isNotZero(primal[i]))
            {
               MSG_INFO1(soplex.spxout, soplex.spxout << colnames[i] << "\t"
                         << std::setw(printwidth) << std::setprecision(printprec)
                         << primal[i] << std::endl;)
            }
         }

         MSG_INFO1(soplex.spxout, soplex.spxout << "All other entries are zero (within "
                   << std::setprecision(1) << std::scientific << Param::epsilon()
                   << std::setprecision(8) << std::fixed
                   << ")." << std::endl;)
      }
      else if(soplex.isPrimalFeasible() && soplex.getPrimal(primal))
      {
         int nNonzeros = 0;
         MSG_INFO1(soplex.spxout, soplex.spxout << "\nPrimal solution (name, value):\n";)

         for(int i = 0; i < soplex.numCols(); ++i)
         {
            if(isNotZero(primal[i]))
            {
               MSG_INFO1(soplex.spxout, soplex.spxout << colnames[i] << "\t"
                         << std::setw(printwidth) << std::setprecision(printprec)
                         << primal[i] << std::endl;)
               ++nNonzeros;
            }
         }

         MSG_INFO1(soplex.spxout, soplex.spxout << "All other variables are zero (within "
                   << std::setprecision(1) << std::scientific << Param::epsilon()
                   << std::setprecision(8) << std::fixed
                   << "). Solution has " << nNonzeros << " nonzero entries." << std::endl;)
      }
      else
         MSG_INFO1(soplex.spxout, soplex.spxout << "No primal information available.\n")
      }

   if(rational)
   {
      VectorRational primal(soplex.numCols());

      if(soplex.getPrimalRayRational(primal))
      {
         MSG_INFO1(soplex.spxout, soplex.spxout << "\nPrimal ray (name, value):\n";)

         for(int i = 0; i < soplex.numCols(); ++i)
         {
            if(primal[i] != (Rational) 0)
            {
               MSG_INFO1(soplex.spxout, soplex.spxout << colnames[i] << "\t"
                         << std::setw(printwidth) << std::setprecision(printprec)
                         << primal[i] << std::endl;)
            }
         }

         MSG_INFO1(soplex.spxout, soplex.spxout << "All other entries are zero." << std::endl;)
      }

      if(soplex.isPrimalFeasible() && soplex.getPrimalRational(primal))
      {
         int nNonzeros = 0;
         MSG_INFO1(soplex.spxout, soplex.spxout << "\nPrimal solution (name, value):\n";)

         for(int i = 0; i < soplex.numColsRational(); ++i)
         {
            if(primal[i] != (Rational) 0)
            {
               MSG_INFO1(soplex.spxout, soplex.spxout << colnames[i] << "\t" << primal[i] << std::endl;)
               ++nNonzeros;
            }
         }

         MSG_INFO1(soplex.spxout, soplex.spxout << "All other variables are zero. Solution has "
                   << nNonzeros << " nonzero entries." << std::endl;)
      }
      else
         MSG_INFO1(soplex.spxout, soplex.spxout << "No primal (rational) solution available.\n")

      }
}

template <class R>
static
void printDualSolution(SoPlexBase<R>& soplex, NameSet& colnames, NameSet& rownames,
                       bool real = true, bool rational = false)
{
   int printprec;
   int printwidth;
   printprec = (int) - log10(Real(Param::epsilon()));
   printwidth = printprec + 10;

   if(real)
   {
      VectorBase<R> dual(soplex.numRows());

      if(soplex.getDualFarkas(dual))
      {
         MSG_INFO1(soplex.spxout, soplex.spxout << "\nDual ray (name, value):\n";)

         for(int i = 0; i < soplex.numRows(); ++i)
         {
            if(isNotZero(dual[i]))
            {
               MSG_INFO1(soplex.spxout, soplex.spxout << rownames[i] << "\t"
                         << std::setw(printwidth) << std::setprecision(printprec)
                         << dual[i] << std::endl;)
            }
         }

         MSG_INFO1(soplex.spxout, soplex.spxout << "All other entries are zero (within "
                   << std::setprecision(1) << std::scientific << Param::epsilon()
                   << std::setprecision(8) << std::fixed << ")." << std::endl;)
      }
      else if(soplex.isDualFeasible() && soplex.getDual(dual))
      {
         MSG_INFO1(soplex.spxout, soplex.spxout << "\nDual solution (name, value):\n";)

         for(int i = 0; i < soplex.numRows(); ++i)
         {
            if(isNotZero(dual[i]))
            {
               MSG_INFO1(soplex.spxout, soplex.spxout << rownames[i] << "\t"
                         << std::setw(printwidth) << std::setprecision(printprec)
                         << dual[i] << std::endl;)
            }
         }

         MSG_INFO1(soplex.spxout, soplex.spxout << "All other dual values are zero (within "
                   << std::setprecision(1) << std::scientific << Param::epsilon()
                   << std::setprecision(8) << std::fixed << ")." << std::endl;)

         VectorBase<R> redcost(soplex.numCols());

         if(soplex.getRedCost(redcost))
         {
            MSG_INFO1(soplex.spxout, soplex.spxout << "\nReduced costs (name, value):\n";)

            for(int i = 0; i < soplex.numCols(); ++i)
            {
               if(isNotZero(redcost[i]))
               {
                  MSG_INFO1(soplex.spxout, soplex.spxout << colnames[i] << "\t"
                            << std::setw(printwidth) << std::setprecision(printprec)
                            << redcost[i] << std::endl;)
               }
            }

            MSG_INFO1(soplex.spxout, soplex.spxout << "All other reduced costs are zero (within "
                      << std::setprecision(1) << std::scientific << Param::epsilon()
                      << std::setprecision(8) << std::fixed << ")." << std::endl;)
         }
      }
      else
         MSG_INFO1(soplex.spxout, soplex.spxout << "No dual information available.\n")
      }

   if(rational)
   {
      VectorRational dual(soplex.numRows());

      if(soplex.getDualFarkasRational(dual))
      {
         MSG_INFO1(soplex.spxout, soplex.spxout << "\nDual ray (name, value):\n";)

         for(int i = 0; i < soplex.numRows(); ++i)
         {
            if(dual[i] != (Rational) 0)
            {
               MSG_INFO1(soplex.spxout, soplex.spxout << rownames[i] << "\t"
                         << std::setw(printwidth)
                         << std::setprecision(printprec)
                         << dual[i] << std::endl;)
            }
         }

         MSG_INFO1(soplex.spxout, soplex.spxout << "All other entries are zero." << std::endl;)
      }

      if(soplex.isDualFeasible() && soplex.getDualRational(dual))
      {
         MSG_INFO1(soplex.spxout, soplex.spxout << "\nDual solution (name, value):\n";)

         for(int i = 0; i < soplex.numRowsRational(); ++i)
         {
            if(dual[i] != (Rational) 0)
               MSG_INFO1(soplex.spxout, soplex.spxout << rownames[i] << "\t" << dual[i] << std::endl;)
            }

         MSG_INFO1(soplex.spxout, soplex.spxout << "All other dual values are zero." << std::endl;)

         VectorRational redcost(soplex.numCols());

         if(soplex.getRedCostRational(redcost))
         {
            MSG_INFO1(soplex.spxout, soplex.spxout << "\nReduced costs (name, value):\n";)

            for(int i = 0; i < soplex.numCols(); ++i)
            {
               if(redcost[i] != (Rational) 0)
                  MSG_INFO1(soplex.spxout, soplex.spxout << colnames[i] << "\t" << redcost[i] << std::endl;)
               }

            MSG_INFO1(soplex.spxout, soplex.spxout << "All other reduced costs are zero." << std::endl;)
         }
      }
      else
         MSG_INFO1(soplex.spxout, soplex.spxout << "No dual (rational) solution available.\n")
      }
}

// Runs SoPlex with the parsed boost variables map
template <class R>
int runSoPlex(int argc, char* argv[])
{
   SoPlexBase<R>* soplex = nullptr;

   Timer* readingTime = nullptr;
   Validation<R>* validation = nullptr;
   int optidx;

   const char* lpfilename = nullptr;
   char* readbasname = nullptr;
   char* writebasname = nullptr;
   char* writefilename = nullptr;
   char* writedualfilename = nullptr;
   char* loadsetname = nullptr;
   char* savesetname = nullptr;
   char* diffsetname = nullptr;
   bool printPrimal = false;
   bool printPrimalRational = false;
   bool printDual = false;
   bool printDualRational = false;
   bool displayStatistics = false;
   bool checkSol = false;

   int returnValue = 0;

   try
   {
      NameSet rownames;
      NameSet colnames;

      // create default timer (CPU time)
      readingTime = TimerFactory::createTimer(Timer::USER_TIME);
      soplex = nullptr;
      spx_alloc(soplex);
      new(soplex) SoPlexBase<R>();

      soplex->printVersion();
      MSG_INFO1(soplex->spxout, soplex->spxout << SOPLEX_COPYRIGHT << std::endl << std::endl);

      validation = nullptr;
      spx_alloc(validation);
      new(validation) Validation<R>();

      // no options were given
      if(argc <= 1)
      {
         printUsage(argv, 0);
         returnValue = 1;
         goto TERMINATE;
      }

      // read arguments from command line
      for(optidx = 1; optidx < argc; optidx++)
      {
         char* option = argv[optidx];

         // we reached <lpfile>
         if(option[0] != '-')
         {
            lpfilename = argv[optidx];
            continue;
         }

         // option string must start with '-', must contain at least two characters, and exactly two characters if and
         // only if it is -x, -y, -q, or -c
         if(option[0] != '-' || option[1] == '\0'
               || ((option[2] == '\0') != (option[1] == 'x' || option[1] == 'X' || option[1] == 'y'
                                           || option[1] == 'Y' || option[1] == 'q' || option[1] == 'c')))
         {
            printUsage(argv, optidx);
            returnValue = 1;
            goto TERMINATE_FREESTRINGS;
         }

         switch(option[1])
         {
         case '-' :
         {
            option = &option[2];

            // --readbas=<basfile> : read starting basis from file
            if(strncmp(option, "readbas=", 8) == 0)
            {
               if(readbasname == nullptr)
               {
                  char* filename = &option[8];
                  readbasname = new char[strlen(filename) + 1];
                  spxSnprintf(readbasname, strlen(filename) + 1, "%s", filename);
               }
            }
            // --writebas=<basfile> : write terminal basis to file
            else if(strncmp(option, "writebas=", 9) == 0)
            {
               if(writebasname == nullptr)
               {
                  char* filename = &option[9];
                  writebasname =  new char[strlen(filename) + 1];
                  spxSnprintf(writebasname, strlen(filename) + 1, "%s", filename);
               }
            }
            // --writefile=<lpfile> : write LP to file
            else if(strncmp(option, "writefile=", 10) == 0)
            {
               if(writefilename == nullptr)
               {
                  char* filename = &option[10];
                  writefilename = new char[strlen(filename) + 1];
                  spxSnprintf(writefilename, strlen(filename) + 1, "%s", filename);
               }
            }
            // --writedual=<lpfile> : write dual LP to a file
            else if(strncmp(option, "writedual=", 10) == 0)
            {
               if(writedualfilename == nullptr)
               {
                  char* dualfilename = &option[10];
                  writedualfilename = new char[strlen(dualfilename) + 1];
                  spxSnprintf(writedualfilename, strlen(dualfilename) + 1, "%s", dualfilename);
               }
            }
            // --loadset=<setfile> : load parameters from settings file
            else if(strncmp(option, "loadset=", 8) == 0)
            {
               if(loadsetname == nullptr)
               {
                  char* filename = &option[8];
                  loadsetname = new char[strlen(filename) + 1];
                  spxSnprintf(loadsetname, strlen(filename) + 1, "%s", filename);

                  if(!soplex->loadSettingsFile(loadsetname))
                  {
                     printUsage(argv, optidx);
                     returnValue = 1;
                     goto TERMINATE_FREESTRINGS;
                  }
                  else
                  {
                     // we need to start parsing again because some command line parameters might have been overwritten
                     optidx = 0;
                  }
               }
            }
            // --saveset=<setfile> : save parameters to settings file
            else if(strncmp(option, "saveset=", 8) == 0)
            {
               if(savesetname == nullptr)
               {
                  char* filename = &option[8];
                  savesetname = new char[strlen(filename) + 1];
                  spxSnprintf(savesetname, strlen(filename) + 1, "%s", filename);
               }
            }
            // --diffset=<setfile> : save modified parameters to settings file
            else if(strncmp(option, "diffset=", 8) == 0)
            {
               if(diffsetname == nullptr)
               {
                  char* filename = &option[8];
                  diffsetname = new char[strlen(filename) + 1];
                  spxSnprintf(diffsetname, strlen(filename) + 1, "%s", filename);
               }
            }
            // --readmode=<value> : choose reading mode for <lpfile> (0* - floating-point, 1 - rational)
            else if(strncmp(option, "readmode=", 9) == 0)
            {
               if(!soplex->setIntParam(soplex->READMODE, option[9] - '0'))
               {
                  printUsage(argv, optidx);
                  returnValue = 1;
                  goto TERMINATE_FREESTRINGS;
               }
            }
            // --solvemode=<value> : choose solving mode (0* - floating-point solve, 1 - auto, 2 - force iterative refinement)
            else if(strncmp(option, "solvemode=", 10) == 0)
            {
               if(!soplex->setIntParam(soplex->SOLVEMODE, option[10] - '0'))
               {
                  printUsage(argv, optidx);
                  returnValue = 1;
                  goto TERMINATE_FREESTRINGS;
               }
               // if the LP is parsed rationally and might be solved rationally, we choose automatic syncmode such that
               // the rational LP is kept after reading
               else if(soplex->intParam(soplex->READMODE) == soplex->READMODE_RATIONAL
                       && soplex->intParam(soplex->SOLVEMODE) != soplex->SOLVEMODE_REAL)
               {
                  soplex->setIntParam(soplex->SYNCMODE, soplex->SYNCMODE_AUTO);
               }
            }
            // --extsol=<value> : external solution for soplex to use for validation
            else if(strncmp(option, "extsol=", 7) == 0)
            {
               char* input = &option[7];

               if(!validation->updateExternalSolution(input))
               {
                  printUsage(argv, optidx);
                  returnValue = 1;
                  goto TERMINATE_FREESTRINGS;
               }
            }
            // --arithmetic=<value> : base arithmetic type, directly handled in main()
            else if(strncmp(option, "arithmetic=", 11) == 0)
            {
               continue;
            }
            // --precision=<value> : arithmetic precision, directly handled in main()
            else if(strncmp(option, "precision=", 10) == 0)
            {
               continue;
            }
            // --<type>:<name>=<val> :  change parameter value using syntax of settings file entries
            else if(!soplex->parseSettingsString(option))
            {
               printUsage(argv, optidx);
               returnValue = 1;
               goto TERMINATE_FREESTRINGS;
            }

            break;
         }

         case 't' :

            // -t<s> : set time limit to <s> seconds
            if(!soplex->setRealParam(soplex->TIMELIMIT, atoi(&option[2])))
            {
               printUsage(argv, optidx);
               returnValue = 1;
               goto TERMINATE_FREESTRINGS;
            }

            break;

         case 'i' :

            // -i<n> : set iteration limit to <n>
            if(!soplex->setIntParam(soplex->ITERLIMIT, atoi(&option[2])))
            {
               printUsage(argv, optidx);
               returnValue = 1;
               goto TERMINATE_FREESTRINGS;
            }

            break;

         case 'f' :

            // -f<eps> : set primal feasibility tolerance to <eps>
            if(!soplex->setRealParam(soplex->FEASTOL, atof(&option[2])))
            {
               printUsage(argv, optidx);
               returnValue = 1;
               goto TERMINATE_FREESTRINGS;
            }

            break;

         case 'o' :

            // -o<eps> : set dual feasibility (optimality) tolerance to <eps>
            if(!soplex->setRealParam(soplex->OPTTOL, atof(&option[2])))
            {
               printUsage(argv, optidx);
               returnValue = 1;
               goto TERMINATE_FREESTRINGS;
            }

            break;

         case 'l' :

            // l<eps> : set validation tolerance to <eps>
            if(!validation->updateValidationTolerance(&option[2]))
            {
               printUsage(argv, optidx);
               returnValue = 1;
               goto TERMINATE_FREESTRINGS;
            }

            break;

         case 's' :

            // -s<value> : choose simplifier/presolver (0 - off, 1 - internal, 2* - PaPILO)
            if(!soplex->setIntParam(soplex->SIMPLIFIER, option[2] - '0'))
            {
               printUsage(argv, optidx);
               returnValue = 1;
               goto TERMINATE_FREESTRINGS;
            }

            break;

         case 'g' :

            // -g<value> : choose scaling (0 - off, 1 - uni-equilibrium, 2* - bi-equilibrium, 3 - geometric, 4 - iterated geometric,  5 - least squares, 6 - geometric-equilibrium)
            if(!soplex->setIntParam(soplex->SCALER, option[2] - '0'))
            {
               printUsage(argv, optidx);
               returnValue = 1;
               goto TERMINATE_FREESTRINGS;
            }

            break;

         case 'p' :

            // -p<value> : choose pricing (0* - auto, 1 - dantzig, 2 - parmult, 3 - devex, 4 - quicksteep, 5 - steep)
            if(!soplex->setIntParam(soplex->PRICER, option[2] - '0'))
            {
               printUsage(argv, optidx);
               returnValue = 1;
               goto TERMINATE_FREESTRINGS;
            }

            break;

         case 'r' :

            // -r<value> : choose ratio tester (0 - textbook, 1 - harris, 2* - fast, 3 - boundflipping)
            if(!soplex->setIntParam(soplex->RATIOTESTER, option[2] - '0'))
            {
               printUsage(argv, optidx);
               returnValue = 1;
               goto TERMINATE_FREESTRINGS;
            }

            break;

         case 'v' :

            // -v<level> : set verbosity to <level> (0 - error, 3 - normal, 5 - high)
            if(!soplex->setIntParam(soplex->VERBOSITY, option[2] - '0'))
            {
               printUsage(argv, optidx);
               returnValue = 1;
               goto TERMINATE_FREESTRINGS;
            }

            break;

         case 'x' :
            // -x : print primal solution
            printPrimal = true;
            break;

         case 'X' :
            // -X : print primal solution with rationals
            printPrimalRational = true;
            break;

         case 'y' :
            // -y : print dual multipliers
            printDual = true;
            break;

         case 'Y' :
            // -Y : print dual multipliers with rationals
            printDualRational = true;
            break;

         case 'q' :
            // -q : display detailed statistics
            displayStatistics = true;
            break;

         case 'c' :
            // -c : perform final check of optimal solution in original problem
            checkSol = true;
            break;

         case 'h' :

            // -h : display all parameters
            if(!soplex->saveSettingsFile(0, false))
            {
               MSG_ERROR(std::cerr << "Error printing parameters\n");
            }

            break;

         //lint -fallthrough
         default :
         {
            printUsage(argv, optidx);
            returnValue = 1;
            goto TERMINATE_FREESTRINGS;
         }
         }
      }

      MSG_INFO1(soplex->spxout, soplex->printUserSettings();)

      // no LP file was given and no settings files are written
      if(lpfilename == nullptr && savesetname == nullptr && diffsetname == nullptr)
      {
         printUsage(argv, 0);
         returnValue = 1;
         goto TERMINATE_FREESTRINGS;
      }

      // ensure that syncmode is not manual
      if(soplex->intParam(soplex->SYNCMODE) == soplex->SYNCMODE_MANUAL)
      {
         MSG_ERROR(std::cerr <<
                   "Error: manual synchronization is invalid on command line.  Change parameter int:syncmode.\n");
         returnValue = 1;
         goto TERMINATE_FREESTRINGS;
      }

      // save settings files
      if(savesetname != nullptr)
      {
         MSG_INFO1(soplex->spxout, soplex->spxout << "Saving parameters to settings file <" << savesetname <<
                   "> . . .\n");

         if(!soplex->saveSettingsFile(savesetname, false))
         {
            MSG_ERROR(std::cerr << "Error writing parameters to file <" << savesetname << ">\n");
         }
      }

      if(diffsetname != nullptr)
      {
         MSG_INFO1(soplex->spxout, soplex->spxout << "Saving modified parameters to settings file <" <<
                   diffsetname << "> . . .\n");

         if(!soplex->saveSettingsFile(diffsetname, true))
         {
            MSG_ERROR(std::cerr << "Error writing modified parameters to file <" << diffsetname << ">\n");
         }
      }

      // no LP file given: exit after saving settings
      if(lpfilename == nullptr)
      {
         if(loadsetname != nullptr || savesetname != nullptr || diffsetname != nullptr)
         {
            MSG_INFO1(soplex->spxout, soplex->spxout << "\n");
         }

         goto TERMINATE_FREESTRINGS;
      }

      // measure time for reading LP file and basis file
      readingTime->start();

      // if the LP is parsed rationally and might be solved rationally, we choose automatic syncmode such that
      // the rational LP is kept after reading
      if(soplex->intParam(soplex->READMODE) == soplex->READMODE_RATIONAL
            && soplex->intParam(soplex->SOLVEMODE) != soplex->SOLVEMODE_REAL)
      {
         soplex->setIntParam(soplex->SYNCMODE, soplex->SYNCMODE_AUTO);
      }

      // read LP from input file
      MSG_INFO1(soplex->spxout, soplex->spxout << "Reading "
                << (soplex->intParam(soplex->READMODE) == soplex->READMODE_REAL ? "(real)" : "(rational)")
                << " LP file <" << lpfilename << "> . . .\n");

      if(!soplex->readFile(lpfilename, &rownames, &colnames))
      {
         MSG_ERROR(std::cerr << "Error while reading file <" << lpfilename << ">.\n");
         returnValue = 1;
         goto TERMINATE_FREESTRINGS;
      }

      // write LP if specified
      if(writefilename != nullptr)
      {
         if(!soplex->writeFile(writefilename, &rownames, &colnames))
         {
            MSG_ERROR(std::cerr << "Error while writing file <" << writefilename << ">.\n\n");
            returnValue = 1;
            goto TERMINATE_FREESTRINGS;
         }
         else
         {
            MSG_INFO1(soplex->spxout, soplex->spxout << "Written LP to file <" << writefilename << ">.\n\n");
         }
      }

      // write dual LP if specified
      if(writedualfilename != nullptr)
      {
         if(!soplex->writeDualFileReal(writedualfilename, &rownames, &colnames))
         {
            MSG_ERROR(std::cerr << "Error while writing dual file <" << writedualfilename << ">.\n\n");
            returnValue = 1;
            goto TERMINATE_FREESTRINGS;
         }
         else
         {
            MSG_INFO1(soplex->spxout, soplex->spxout << "Written dual LP to file <" << writedualfilename <<
                      ">.\n\n");
         }
      }

      // read basis file if specified
      if(readbasname != nullptr)
      {
         MSG_INFO1(soplex->spxout, soplex->spxout << "Reading basis file <" << readbasname << "> . . . ");

         if(!soplex->readBasisFile(readbasname, &rownames, &colnames))
         {
            MSG_ERROR(std::cerr << "Error while reading file <" << readbasname << ">.\n");
            returnValue = 1;
            goto TERMINATE_FREESTRINGS;
         }
      }

      readingTime->stop();

      MSG_INFO1(soplex->spxout,
                std::streamsize prec = soplex->spxout.precision();
                soplex->spxout << "Reading took "
                << std::fixed << std::setprecision(2) << readingTime->time()
                << std::scientific << std::setprecision(int(prec))
                << " seconds.\n\n");

      MSG_INFO1(soplex->spxout, soplex->spxout << "LP has " << soplex->numRows() << " rows "
                << soplex->numCols() << " columns and " << soplex->numNonzeros() << " nonzeros.\n\n");

      // solve the LP
      soplex->optimize();

      // print solution, check solution, and display statistics
      printPrimalSolution(*soplex, colnames, rownames, printPrimal, printPrimalRational);
      printDualSolution(*soplex, colnames, rownames, printDual, printDualRational);

      if(checkSol)
         checkSolution<R>(*soplex); // The type needs to get fixed here

      if(displayStatistics)
      {
         MSG_INFO1(soplex->spxout, soplex->spxout << "Statistics\n==========\n\n");
         soplex->printStatistics(soplex->spxout.getStream(SPxOut::INFO1));
      }

      if(validation->validate)
         validation->validateSolveReal(*soplex);

      // write basis file if specified
      if(writebasname != nullptr)
      {
         if(!soplex->hasBasis())
         {
            MSG_WARNING(soplex->spxout, soplex->spxout <<
                        "No basis information available.  Could not write file <" << writebasname << ">\n\n");
         }
         else if(!soplex->writeBasisFile(writebasname, &rownames, &colnames))
         {
            MSG_ERROR(std::cerr << "Error while writing file <" << writebasname << ">.\n\n");
            returnValue = 1;
            goto TERMINATE_FREESTRINGS;
         }
         else
         {
            MSG_INFO1(soplex->spxout, soplex->spxout << "Written basis information to file <" << writebasname <<
                      ">.\n\n");
         }
      }
   }
   catch(const SPxException& x)
   {
      MSG_ERROR(std::cerr << "Exception caught: " << x.what() << "\n");
      returnValue = 1;
      goto TERMINATE_FREESTRINGS;
   }

TERMINATE_FREESTRINGS:
   freeStrings(readbasname, writebasname, loadsetname, savesetname, diffsetname);

TERMINATE:

   // because EGlpNumClear() calls mpq_clear() for all mpq_t variables, we need to destroy all objects of class Rational
   // beforehand; hence all Rational objects and all data that uses Rational objects must be allocated dynamically via
   // spx_alloc() and freed here; disabling the list memory is crucial
   if(nullptr != soplex)
   {
      soplex->~SoPlexBase();
      spx_free(soplex);
   }

   if(nullptr != validation)
   {
      validation->~Validation();
      spx_free(validation);
   }

   if(nullptr != readingTime)
   {
      readingTime->~Timer();
      spx_free(readingTime);
   }

   return returnValue;
}

/// runs SoPlexBase command line
int main(int argc, char* argv[])
{
   int arithmetic = 0;
   int precision = 0;
   int optidx;

   // find out which precision/solvemode soplex should be run in. the rest happens in runSoPlex
   // no options were given
   if(argc <= 1)
   {
      printUsage(argv, 0);
      return 1;
   }

   // read arguments from command line
   for(optidx = 1; optidx < argc; optidx++)
   {
      char* option = argv[optidx];

      // we reached <lpfile>
      if(option[0] != '-')
         continue;

      // option string must start with '-', must contain at least two characters, and exactly two characters if and
      // only if it is -x, -y, -q, or -c
      if(option[0] != '-' || option[1] == '\0'
            || ((option[2] == '\0') != (option[1] == 'x' || option[1] == 'X' || option[1] == 'y'
                                        || option[1] == 'Y' || option[1] == 'q' || option[1] == 'c')))
      {
         printUsage(argv, optidx);
         return 1;
      }

      switch(option[1])
      {
      case '-' :
         option = &option[2];

         // --arithmetic=<value> : choose base arithmetic type (0 - double, 1 - quadprecision, 2 - higher multiprecision)
         // only need to do something here if multi or quad, the rest is handled in runSoPlex
         if(strncmp(option, "arithmetic=", 11) == 0)
         {
            if(option[11] == '1')
            {
#ifndef SOPLEX_WITH_FLOAT128
               MSG_ERROR(std::cerr <<
                         "Cannot set arithmetic type to quadprecision - Soplex compiled without quadprecision support\n";)
               printUsage(argv, 0);
               return 1;
#else
               arithmetic = 1;
#endif
            }
            else if(option[11] == '2')
            {
#ifndef SOPLEX_WITH_BOOST
               MSG_ERROR(std::cerr <<
                         "Cannot set arithmetic type to multiprecision - Soplex compiled without boost\n";)
               printUsage(argv, 0);
               return 1;
#else
               arithmetic = 2;

               // default precision in multiprecision solve is 50
               if(precision == 0)
                  precision = 50;

#endif
            }
         }
         // set precision
         else if(strncmp(option, "precision=", 10) == 0)
         {
            precision = atoi(option + 10);
#ifndef SOPLEX_WITH_BOOST
            MSG_ERROR(std::cerr << "Setting precision to non-default value without Boost has no effect\n";)
#endif
         }

         break;

      default:
         break;
      }
   }

   if(precision != 0 && arithmetic != 2)
   {
      MSG_ERROR(std::cerr <<
                "Setting precision to non-default value without enabling multiprecision solve has no effect\n";)
   }

   switch(arithmetic)
   {
   case 0:                 // double
      runSoPlex<Real>(argc, argv);
      break;

#ifdef SOPLEX_WITH_BOOST
#ifdef SOPLEX_WITH_FLOAT128

   case 1:                // quadprecision
#if BOOST_VERSION < 107000
      std::cerr << "Error: Boost version too old." << std:: endl <<
                "In order to use the quadprecision feature of SoPlex," <<
                " Boost Version 1.70.0 or higher is required." << std::endl << \
                "Included Boost version is " << BOOST_VERSION / 100000 << "."  // maj. version
                << BOOST_VERSION / 100 % 1000 << "."  // min. version
                << BOOST_VERSION % 100                // patch version;
                << std::endl;
#else
      using namespace boost::multiprecision;
      using Quad = boost::multiprecision::float128;
      runSoPlex<Quad>(argc, argv);
#endif
      break;
#endif

   case 2:                 // soplex mpf
      using namespace boost::multiprecision;

#if BOOST_VERSION < 107000
      std::cerr << "Error: Boost version too old." << std:: endl <<
                "In order to use the multiprecision feature of SoPlex," <<
                " Boost Version 1.70.0 or higher is required." << std::endl << \
                "Included Boost version is " << BOOST_VERSION / 100000 << "."  // maj. version
                << BOOST_VERSION / 100 % 1000 << "."  // min. version
                << BOOST_VERSION % 100                // patch version;
                << std::endl;
#else
#ifdef SOPLEX_WITH_MPFR

      // et_off means the expression templates options is turned off. TODO:
      // The documentation also mentions about static vs dynamic memory
      // allocation for the mpfr types. Is it relevant here? I probably also
      // need to have the mpfr_float_eto in the global soplex namespace
      using multiprecision = number<mpfr_float_backend<0>, et_off>;
      multiprecision::default_precision(precision);
      runSoPlex<multiprecision>(argc, argv);
#endif  // SOPLEX_WITH_MPFR

#ifdef SOPLEX_WITH_CPPMPF
      // It seems that precision cannot be set on run time for cpp_float
      // backend for boost::number. So a precision of 50 decimal points is
      // set.
      using multiprecision1 = number<cpp_dec_float<50>, et_off>;
      using multiprecision2 = number<cpp_dec_float<100>, et_off>;
      using multiprecision3 = number<cpp_dec_float<200>, et_off>;

      if(precision <= 50)
         runSoPlex<multiprecision1>(argc, argv);
      else if(precision <= 100)
         runSoPlex<multiprecision2>(argc, argv);
      else
         runSoPlex<multiprecision3>(argc, argv);

#endif  // SOPLEX_WITH_CPPMPF
#endif
      break;
#endif

   // coverity[dead_error_begin]
   default:
      std::cerr << "Wrong value for the arithmetic mode\n";
      return 0;
   }
}
