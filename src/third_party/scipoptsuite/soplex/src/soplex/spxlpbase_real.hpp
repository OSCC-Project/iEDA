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

/**@file  spxlpbase_real.hpp
 * @brief Saving LPs with R values in a form suitable for SoPlex.
 */

#include <assert.h>
#include <stdio.h>
#include <ctype.h>
#include <iostream>

#include "soplex/spxdefines.h"
#include "soplex/spxout.h"
#include "soplex/mpsinput.h"
#include "soplex/exceptions.h"
#include "soplex/spxscaler.h"

namespace soplex
{
/// Is \p c a \c space, \c tab, \c nl or \c cr ?
static inline bool LPFisSpace(int c)
{
   return (c == ' ') || (c == '\t') || (c == '\n') || (c == '\r');
}

/// Is there a number at the beginning of \p s ?
static inline bool LPFisValue(const char* s)
{
   return ((*s >= '0') && (*s <= '9')) || (*s == '+') || (*s == '-') || (*s == '.');
}


/// Is there a comparison operator at the beginning of \p s ?
static inline bool LPFisSense(const char* s)
{
   return (*s == '<') || (*s == '>') || (*s == '=');
}

template <class R>
void SPxLPBase<R>::unscaleLP()
{
   MSG_INFO3((*this->spxout), (*this->spxout) << "remove persistent scaling of LP" << std::endl;)

   if(lp_scaler)
   {
      lp_scaler->unscale(*this);
   }
   else
   {
      MSG_INFO3((*this->spxout), (*this->spxout) << "no LP scaler available" << std::endl;)
   }
}

template <class R> inline
void SPxLPBase<R>::computePrimalActivity(const VectorBase<R>& primal, VectorBase<R>& activity,
      const bool unscaled) const
{
   if(primal.dim() != nCols())
      throw SPxInternalCodeException("XSPXLP01 Primal vector for computing row activity has wrong dimension");

   if(activity.dim() != nRows())
      throw SPxInternalCodeException("XSPXLP03 Activity vector computing row activity has wrong dimension");

   int c;

   for(c = 0; c < nCols() && primal[c] == 0; c++)
      ;

   if(c >= nCols())
   {
      activity.clear();
      return;
   }

   DSVectorBase<R> tmp(nRows());

   if(unscaled && _isScaled)
   {
      lp_scaler->getColUnscaled(*this, c, tmp);
      activity = tmp;
   }
   else
      activity = colVector(c);

   activity *= primal[c];
   c++;

   for(; c < nCols(); c++)
   {
      if(primal[c] != 0)
      {
         if(unscaled && _isScaled)
         {
            lp_scaler->getColUnscaled(*this, c, tmp);
            activity.multAdd(primal[c], tmp);
         }
         else
            activity.multAdd(primal[c], colVector(c));
      }
   }
}

template <class R> inline
void SPxLPBase<R>::computeDualActivity(const VectorBase<R>& dual, VectorBase<R>& activity,
                                       const bool unscaled) const
{
   if(dual.dim() != nRows())
      throw SPxInternalCodeException("XSPXLP02 Dual vector for computing dual activity has wrong dimension");

   if(activity.dim() != nCols())
      throw SPxInternalCodeException("XSPXLP04 Activity vector computing dual activity has wrong dimension");

   int r;

   for(r = 0; r < nRows() && dual[r] == 0; r++)
      ;

   if(r >= nRows())
   {
      activity.clear();
      return;
   }

   DSVectorBase<R> tmp(nCols());

   if(unscaled && _isScaled)
   {
      lp_scaler->getRowUnscaled(*this, r, tmp);
      activity = tmp;
   }
   else
      activity = rowVector(r);

   activity *= dual[r];
   r++;

   for(; r < nRows(); r++)
   {
      if(dual[r] != 0)
      {
         if(unscaled && _isScaled)
         {
            lp_scaler->getRowUnscaled(*this, r, tmp);
            activity.multAdd(dual[r], tmp);
         }
         else
            activity.multAdd(dual[r], rowVector(r));
      }
   }
}

template <class R> inline
R SPxLPBase<R>::maxAbsNzo(bool unscaled) const
{
   R maxi = 0.0;

   if(unscaled && _isScaled)
   {
      assert(lp_scaler != nullptr);

      for(int i = 0; i < nCols(); ++i)
      {
         R m = lp_scaler->getColMaxAbsUnscaled(*this, i);

         if(m > maxi)
            maxi = m;
      }
   }
   else
   {
      for(int i = 0; i < nCols(); ++i)
      {
         R m = colVector(i).maxAbs();

         if(m > maxi)
            maxi = m;
      }
   }

   assert(maxi >= 0.0);

   return maxi;
}

template <class R> inline
R SPxLPBase<R>::minAbsNzo(bool unscaled) const
{
   R mini = R(infinity);

   if(unscaled && _isScaled)
   {
      assert(lp_scaler != nullptr);

      for(int i = 0; i < nCols(); ++i)
      {
         R m = lp_scaler->getColMinAbsUnscaled(*this, i);

         if(m < mini)
            mini = m;
      }
   }
   else
   {
      for(int i = 0; i < nCols(); ++i)
      {
         R m = colVector(i).minAbs();

         if(m < mini)
            mini = m;
      }
   }

   assert(mini >= 0.0);

   return mini;
}

/// Gets unscaled objective vector.
template <class R>
void SPxLPBase<R>::getObjUnscaled(VectorBase<R>& pobj) const
{
   if(_isScaled)
   {
      assert(lp_scaler);
      lp_scaler->getMaxObjUnscaled(*this, pobj);
   }
   else
   {
      pobj = LPColSetBase<R>::maxObj();
   }

   if(spxSense() == MINIMIZE)
      pobj *= -1.0;
}

/// Gets unscaled row vector of row \p i.
template <class R>
void SPxLPBase<R>::getRowVectorUnscaled(int i, DSVectorBase<R>& vec) const
{
   assert(i >= 0 && i < nRows());

   if(_isScaled)
      lp_scaler->getRowUnscaled(*this, i, vec);
   else
      vec = DSVectorBase<R>(LPRowSetBase<R>::rowVector(i));
}

/// Gets unscaled right hand side vector.
template <class R>
void SPxLPBase<R>::getRhsUnscaled(VectorBase<R>& vec) const
{
   if(_isScaled)
      lp_scaler->getRhsUnscaled(*this, vec);
   else
      vec = LPRowSetBase<R>::rhs();
}

/// Returns unscaled right hand side of row number \p i.
template <class R>
R SPxLPBase<R>::rhsUnscaled(int i) const
{
   assert(i >= 0 && i < nRows());

   if(_isScaled)
      return lp_scaler->rhsUnscaled(*this, i);
   else
      return LPRowSetBase<R>::rhs(i);
}

/// Returns unscaled right hand side of row with identifier \p id.
template <class R>
R SPxLPBase<R>::rhsUnscaled(const SPxRowId& id) const
{
   assert(id.isValid());
   return rhsUnscaled(number(id));
}

/// Returns unscaled left hand side vector.
template <class R>
void SPxLPBase<R>::getLhsUnscaled(VectorBase<R>& vec) const
{
   if(_isScaled)
      lp_scaler->getLhsUnscaled(*this, vec);
   else
      vec = LPRowSetBase<R>::lhs();
}

/// Returns unscaled left hand side of row number \p i.
template <class R>
R SPxLPBase<R>::lhsUnscaled(int i) const
{
   assert(i >= 0 && i < nRows());

   if(_isScaled)
      return lp_scaler->lhsUnscaled(*this, i);
   else
      return LPRowSetBase<R>::lhs(i);
}

/// Returns left hand side of row with identifier \p id.
template <class R>
R SPxLPBase<R>::lhsUnscaled(const SPxRowId& id) const
{
   assert(id.isValid());
   return lhsUnscaled(number(id));
}

/// Gets column vector of column \p i.
template <class R>
void SPxLPBase<R>::getColVectorUnscaled(int i, DSVectorBase<R>& vec) const
{
   assert(i >= 0 && i < nCols());

   if(_isScaled)
      lp_scaler->getColUnscaled(*this, i, vec);
   else
      vec = LPColSetBase<R>::colVector(i);
}

/// Gets column vector of column with identifier \p id.
template <class R>
void SPxLPBase<R>::getColVectorUnscaled(const SPxColId& id, DSVectorBase<R>& vec) const
{
   assert(id.isValid());
   getColVectorUnscaled(number(id), vec);
}

/// Returns unscaled objective value of column \p i.
template <class R>
R SPxLPBase<R>::objUnscaled(int i) const
{
   assert(i >= 0 && i < nCols());
   R res;

   if(_isScaled)
   {
      res = lp_scaler->maxObjUnscaled(*this, i);
   }
   else
   {
      res = maxObj(i);
   }

   if(spxSense() == MINIMIZE)
      res *= -1;

   return res;
}

/// Returns unscaled objective value of column with identifier \p id.
template <class R>
R SPxLPBase<R>::objUnscaled(const SPxColId& id) const
{
   assert(id.isValid());
   return objUnscaled(number(id));
}

/// Returns unscaled objective vector for maximization problem.
template <class R>
void SPxLPBase<R>::maxObjUnscaled(VectorBase<R>& vec) const
{
   if(_isScaled)
      lp_scaler->getMaxObjUnscaled(*this, vec);
   else
      vec = LPColSetBase<R>::maxObj();
}

/// Returns unscaled objective value of column \p i for maximization problem.
template <class R>
R SPxLPBase<R>::maxObjUnscaled(int i) const
{
   assert(i >= 0 && i < nCols());

   if(_isScaled)
      return lp_scaler->maxObjUnscaled(*this, i);
   else
      return LPColSetBase<R>::maxObj(i);
}

/// Returns unscaled objective value of column with identifier \p id for maximization problem.
template <class R>
R SPxLPBase<R>::maxObjUnscaled(const SPxColId& id) const
{
   assert(id.isValid());
   return maxObjUnscaled(number(id));
}

/// Returns unscaled upper bound vector
template <class R>
void SPxLPBase<R>::getUpperUnscaled(VectorBase<R>& vec) const
{
   if(_isScaled)
      lp_scaler->getUpperUnscaled(*this, vec);
   else
      vec = VectorBase<R>(LPColSetBase<R>::upper());
}

/// Returns unscaled upper bound of column \p i.
template <class R>
R SPxLPBase<R>::upperUnscaled(int i) const
{
   assert(i >= 0 && i < nCols());

   if(_isScaled)
      return lp_scaler->upperUnscaled(*this, i);
   else
      return LPColSetBase<R>::upper(i);
}

/// Returns unscaled upper bound of column with identifier \p id.
template <class R>
R SPxLPBase<R>::upperUnscaled(const SPxColId& id) const
{
   assert(id.isValid());
   return upperUnscaled(number(id));
}

/// Returns unscaled lower bound vector.
template <class R>
void SPxLPBase<R>::getLowerUnscaled(VectorBase<R>& vec) const
{
   if(_isScaled)
      lp_scaler->getLowerUnscaled(*this, vec);
   else
      vec = VectorBase<R>(LPColSetBase<R>::lower());
}

/// Returns unscaled lower bound of column \p i.
template<class R>
R SPxLPBase<R>::lowerUnscaled(int i) const
{
   assert(i >= 0 && i < nCols());

   if(_isScaled)
      return lp_scaler->lowerUnscaled(*this, i);
   else
      return LPColSetBase<R>::lower(i);
}

/// Returns unscaled lower bound of column with identifier \p id.
template <class R>
R SPxLPBase<R>::lowerUnscaled(const SPxColId& id) const
{
   assert(id.isValid());
   return lowerUnscaled(number(id));
}




// ---------------------------------------------------------------------------------------------------------------------
//  Specialization for reading LP format
// ---------------------------------------------------------------------------------------------------------------------

#define LPF_MAX_LINE_LEN  8192     ///< maximum length of a line (8190 + \\n + \\0)


/// Is there a possible column name at the beginning of \p s ?
static inline bool LPFisColName(const char* s)
{
   // strchr() gives a true for the null char.
   if(*s == '\0')
      return false;

   return ((*s >= 'A') && (*s <= 'Z'))
          || ((*s >= 'a') && (*s <= 'z'))
          || (strchr("!\"#$%&()/,;?@_'`{}|~", *s) != 0);
}


static inline bool LPFisInfinity(const char* s)
{
   return ((s[0] == '-') || (s[0] == '+'))
          && (tolower(s[1]) == 'i')
          && (tolower(s[2]) == 'n')
          && (tolower(s[3]) == 'f');
}



static inline bool LPFisFree(const char* s)
{
   return (tolower(s[0]) == 'f')
          && (tolower(s[1]) == 'r')
          && (tolower(s[2]) == 'e')
          && (tolower(s[3]) == 'e');
}



/// Read the next number and advance \p pos.
/** If only a sign is encountered, the number is assumed to be \c sign * 1.0.  This routine will not catch malformatted
 *  numbers like .e10 !
 */
template <class R>
static R LPFreadValue(char*& pos, SPxOut* spxout)
{
   assert(LPFisValue(pos));

   char        tmp[LPF_MAX_LINE_LEN];
   const char* s = pos;
   char*       t;
   R        value = 1.0;
   bool        has_digits = false;
   bool        has_emptyexponent = false;

   // 1. sign
   if((*s == '+') || (*s == '-'))
      s++;

   // 2. Digits before the decimal dot
   while((*s >= '0') && (*s <= '9'))
   {
      has_digits = true;
      s++;
   }

   // 3. Decimal dot
   if(*s == '.')
   {
      s++;

      // 4. If there was a dot, possible digit behind it
      while((*s >= '0') && (*s <= '9'))
      {
         has_digits = true;
         s++;
      }
   }

   // 5. Exponent
   if(tolower(*s) == 'e')
   {
      has_emptyexponent = true;
      s++;

      // 6. Exponent sign
      if((*s == '+') || (*s == '-'))
         s++;

      // 7. Exponent digits
      while((*s >= '0') && (*s <= '9'))
      {
         has_emptyexponent = false;
         s++;
      }
   }

   assert(s != pos);

   if(has_emptyexponent)
   {
      MSG_WARNING((*spxout), (*spxout) <<
                  "WLPFRD01 Warning: found empty exponent in LP file - check for forbidden variable names with initial 'e' or 'E'\n";
                 )
   }

   if(!has_digits)
      value = (*pos == '-') ? -1.0 : 1.0;
   else
   {
      for(t = tmp; pos != s; pos++)
         *t++ = *pos;

      *t = '\0';
      value = atof(tmp);
   }

   pos += s - pos;

   assert(pos == s);

   MSG_DEBUG(std::cout << "DLPFRD01 LPFreadValue = " << value << std::endl;)

   if(LPFisSpace(*pos))
      pos++;

   return value;
}



/// Read the next column name from the input.
/** The name read is looked up and if not found \p emptycol
 *  is added to \p colset. \p pos is advanced behind the name.
 *  @return The Index of the named column.
 */
template <class R>
static int LPFreadColName(char*& pos, NameSet* colnames, LPColSetBase<R>& colset,
                          const LPColBase<R>* emptycol, SPxOut* spxout)
{
   assert(LPFisColName(pos));
   assert(colnames != 0);

   char        name[LPF_MAX_LINE_LEN];
   const char* s = pos;
   int         i;
   int         colidx;

   // These are the characters that are not allowed in a column name.
   while((strchr("+-.<>= ", *s) == 0) && (*s != '\0'))
      s++;

   for(i = 0; pos != s; i++, pos++)
      name[i] = *pos;

   name[i] = '\0';

   if((colidx = colnames->number(name)) < 0)
   {
      // We only add the name if we got an empty column.
      if(emptycol == nullptr)
      {
         MSG_WARNING((*spxout), (*spxout) << "WLPFRD02 Unknown variable \"" << name << "\" ";)
      }
      else
      {
         colidx = colnames->num();
         colnames->add(name);
         colset.add(*emptycol);
      }
   }

   MSG_DEBUG(std::cout << "DLPFRD03 LPFreadColName [" << name << "] = " << colidx << std::endl;)

   if(LPFisSpace(*pos))
      pos++;

   return colidx;
}



/// Read the next <,>,=,==,<=,=<,>=,=> and advance \p pos.
static inline int LPFreadSense(char*& pos)
{
   assert(LPFisSense(pos));

   int sense = *pos++;

   if((*pos == '<') || (*pos == '>'))
      sense = *pos++;
   else if(*pos == '=')
      pos++;

   MSG_DEBUG(std::cout << "DLPFRD04 LPFreadSense = " << static_cast<char>(sense) << std::endl;)

   if(LPFisSpace(*pos))
      pos++;

   return sense;
}



/// Is the \p keyword present in \p buf ? If yes, advance \p pos.
/** \p keyword should be lower case. It can contain optional sections which are enclosed in '[' ']' like "min[imize]".
 */
static inline bool LPFhasKeyword(char*& pos, const char* keyword)
{
   int i;
   int k;

   assert(keyword != 0);

   for(i = 0, k = 0; keyword[i] != '\0'; i++, k++)
   {
      if(keyword[i] == '[')
      {
         i++;

         // Here we assumed that we have a ']' for the '['.
         while((tolower(pos[k]) == keyword[i]) && (pos[k] != '\0'))
         {
            k++;
            i++;
         }

         while(keyword[i] != ']')
            i++;

         --k;
      }
      else
      {
         if(keyword[i] != tolower(pos[k]))
            break;
      }
   }

   // we have to be at the end of the keyword and the word found on the line also has to end here.  Attention: The
   // LPFisSense is a kludge to allow LPFhasKeyword also to process Inf[inity] keywords in the bounds section.
   if(keyword[i] == '\0' && (pos[k] == '\0' || LPFisSpace(pos[k]) || LPFisSense(&pos[k])))
   {
      pos += k;

      MSG_DEBUG(std::cout << "DLPFRD05 LPFhasKeyword: " << keyword << std::endl;)

      return true;
   }

   return false;
}



/// If \p buf start with "name:" store the name in \p rownames and advance \p pos.
static inline bool LPFhasRowName(char*& pos, NameSet* rownames)
{
   const char* s = strchr(pos, ':');

   if(s == 0)
      return false;

   int dcolpos = int(s - pos);

   int end;
   int srt;

   // skip spaces between name and ":"
   for(end = dcolpos - 1; end >= 0; end--)
      if(pos[end] != ' ')
         break;

   // are there only spaces in front of the ":" ?
   if(end < 0)
   {
      pos = &(pos[dcolpos + 1]);
      return false;
   }

   // skip spaces in front of name
   for(srt = end - 1; srt >= 0; srt--)
      if(pos[srt] == ' ')
         break;

   // go back to the non-space character
   srt++;

   assert(srt <= end && pos[srt] != ' ');

   char name[LPF_MAX_LINE_LEN];
   int i;
   int k = 0;

   for(i = srt; i <= end; i++)
      name[k++] = pos[i];

   name[k] = '\0';

   if(rownames != 0)
      rownames->add(name);

   pos = &(pos[dcolpos + 1]);

   return true;
}


template <class R>
static R LPFreadInfinity(char*& pos)
{
   assert(LPFisInfinity(pos));

   R sense = (*pos == '-') ? -1.0 : 1.0;

   (void) LPFhasKeyword(++pos, "inf[inity]");

   return sense * R(infinity);
}


/// Read LP in "CPLEX LP File Format".
/** The specification is taken from the ILOG CPLEX 7.0 Reference Manual, Appendix E, Page 527.
 *
 *  This routine should read (most?) valid LP format files.  What it will not do, is find all cases where a file is ill
 *  formed.  If this happens it may complain and read nothing or read "something".
 *
 *  Problem: A line ending in '+' or '-' followed by a line starting with a number, will be regarded as an error.
 *
 *  The reader will accept the keyword INT[egers] as a synonym for GEN[erals] which is an undocumented feature in CPLEX.
 *
 *  A difference to the CPLEX reader, is that no name for the objective row is required.
 *
 * The manual says the maximum allowed line length is 255 characters, but CPLEX does not complain if the lines are
 * longer.
 *
 *  @return true if the file was read correctly
 */
template <class R> inline
bool SPxLPBase<R>::readLPF(
   std::istream& p_input,                ///< input stream.
   NameSet*      p_rnames,               ///< row names.
   NameSet*      p_cnames,               ///< column names.
   DIdxSet*      p_intvars)              ///< integer variables.
{
   enum
   {
      START, OBJECTIVE, CONSTRAINTS, BOUNDS, INTEGERS, BINARIES
   } section = START;

   NameSet* rnames;                      ///< row names.
   NameSet* cnames;                      ///< column names.

   LPColSetBase<R> cset;              ///< the set of columns read.
   LPRowSetBase<R> rset;              ///< the set of rows read.
   LPColBase<R> emptycol;             ///< reusable empty column.
   LPRowBase<R> row;                  ///< last assembled row.
   DSVectorBase<R> vec;               ///< last assembled vector (from row).

   R val = 1.0;
   int colidx;
   int sense = 0;

   char buf[LPF_MAX_LINE_LEN];
   char tmp[LPF_MAX_LINE_LEN];
   char line[LPF_MAX_LINE_LEN];
   int lineno = 0;
   bool unnamed = true;
   bool finished = false;
   bool other;
   bool have_value = true;
   int i;
   int k;
   char* s;
   char* pos;
   char* pos_old = 0;

   if(p_cnames)
      cnames = p_cnames;
   else
   {
      cnames = 0;
      spx_alloc(cnames);
      cnames = new(cnames) NameSet();
   }

   cnames->clear();

   if(p_rnames)
      rnames = p_rnames;
   else
   {
      try
      {
         rnames = 0;
         spx_alloc(rnames);
         rnames = new(rnames) NameSet();
      }
      catch(const SPxMemoryException& x)
      {
         if(!p_cnames)
         {
            cnames->~NameSet();
            spx_free(cnames);
         }

         throw x;
      }
   }

   rnames->clear();

   SPxLPBase<R>::clear(); // clear the LP.

   //--------------------------------------------------------------------------
   //--- Main Loop
   //--------------------------------------------------------------------------
   for(;;)
   {
      // 0. Read a line from the file.
      if(!p_input.getline(buf, sizeof(buf)))
      {
         if(strlen(buf) == LPF_MAX_LINE_LEN - 1)
         {
            MSG_ERROR(std::cerr << "ELPFRD06 Line exceeds " << LPF_MAX_LINE_LEN - 2
                      << " characters" << std::endl;)
         }
         else
         {
            MSG_ERROR(std::cerr << "ELPFRD07 No 'End' marker found" << std::endl;)
            finished = true;
         }

         break;
      }

      lineno++;
      i   = 0;
      pos = buf;

      MSG_DEBUG(std::cout << "DLPFRD08 Reading line " << lineno
                << " (pos=" << pos << ")" << std::endl;)

      // 1. Remove comments.
      if(0 != (s = strchr(buf, '\\')))
         * s = '\0';

      // 2. Look for keywords.
      if(section == START)
      {
         if(LPFhasKeyword(pos, "max[imize]"))
         {
            changeSense(SPxLPBase<R>::MAXIMIZE);
            section = OBJECTIVE;
         }
         else if(LPFhasKeyword(pos, "min[imize]"))
         {
            changeSense(SPxLPBase<R>::MINIMIZE);
            section = OBJECTIVE;
         }
      }
      else if(section == OBJECTIVE)
      {
         if(LPFhasKeyword(pos, "s[ubject][   ]t[o]")
               || LPFhasKeyword(pos, "s[uch][    ]t[hat]")
               || LPFhasKeyword(pos, "s[.][    ]t[.]")
               || LPFhasKeyword(pos, "lazy con[straints]"))
         {
            // store objective vector
            for(int j = vec.size() - 1; j >= 0; --j)
               cset.maxObj_w(vec.index(j)) = vec.value(j);

            // multiplication with -1 for minimization is done below
            vec.clear();
            have_value = true;
            val = 1.0;
            section = CONSTRAINTS;
         }
      }
      else if(section == CONSTRAINTS &&
              (LPFhasKeyword(pos, "s[ubject][   ]t[o]")
               || LPFhasKeyword(pos, "s[uch][    ]t[hat]")
               || LPFhasKeyword(pos, "s[.][    ]t[.]")))
      {
         have_value = true;
         val = 1.0;
      }
      else
      {
         if(LPFhasKeyword(pos, "lazy con[straints]"))
            ;
         else if(LPFhasKeyword(pos, "bound[s]"))
            section = BOUNDS;
         else if(LPFhasKeyword(pos, "bin[ary]"))
            section = BINARIES;
         else if(LPFhasKeyword(pos, "bin[aries]"))
            section = BINARIES;
         else if(LPFhasKeyword(pos, "gen[erals]"))
            section = INTEGERS;
         else if(LPFhasKeyword(pos, "int[egers]"))   // this is undocumented
            section = INTEGERS;
         else if(LPFhasKeyword(pos, "end"))
         {
            finished = true;
            break;
         }
         else if(LPFhasKeyword(pos, "s[ubject][   ]t[o]")  // second time
                 || LPFhasKeyword(pos, "s[uch][    ]t[hat]")
                 || LPFhasKeyword(pos, "s[.][    ]t[.]")
                 || LPFhasKeyword(pos, "lazy con[straints]"))
         {
            // In principle this has to checked for all keywords above,
            // otherwise we just ignore any half finished constraint
            if(have_value)
               goto syntax_error;

            have_value = true;
            val = 1.0;
         }
      }

      // 3a. Look for row names in objective and drop it.
      if(section == OBJECTIVE)
         LPFhasRowName(pos, 0);

      // 3b. Look for row name in constraint and store it.
      if(section == CONSTRAINTS)
         if(LPFhasRowName(pos, rnames))
            unnamed = false;

      // 4a. Remove initial spaces.
      while(LPFisSpace(pos[i]))
         i++;

      // 4b. remove spaces if they do not appear before the name of a vaiable.
      for(k = 0; pos[i] != '\0'; i++)
         if(!LPFisSpace(pos[i]) || LPFisColName(&pos[i + 1]))
            tmp[k++] = pos[i];

      tmp[k] = '\0';

      // 5. Is this an empty line ?
      if(tmp[0] == '\0')
         continue;

      // 6. Collapse sequences of '+' and '-'. e.g ++---+ => -
      for(i = 0, k = 0; tmp[i] != '\0'; i++)
      {
         while(((tmp[i] == '+') || (tmp[i] == '-')) && ((tmp[i + 1] == '+') || (tmp[i + 1] == '-')))
         {
            if(tmp[i++] == '-')
               tmp[i] = (tmp[i] == '-') ? '+' : '-';
         }

         line[k++] = tmp[i];
      }

      line[k] = '\0';

      //-----------------------------------------------------------------------
      //--- Line processing loop
      //-----------------------------------------------------------------------
      pos = line;

      MSG_DEBUG(std::cout << "DLPFRD09 pos=" << pos << std::endl;)

      // 7. We have something left to process.
      while((pos != 0) && (*pos != '\0'))
      {
         // remember our position, so we are sure we make progress.
         pos_old = pos;

         // now process the sections
         switch(section)
         {
         case OBJECTIVE:
            if(LPFisValue(pos))
            {
               R pre_sign = 1.0;

               /* Already having here a value could only result from being the first number in a constraint, or a sign
                * '+' or '-' as last token on the previous line.
                */
               if(have_value)
               {
                  if(NE(spxAbs(val), R(1.0)))
                     goto syntax_error;

                  if(EQ(val, R(-1.0)))
                     pre_sign = val;
               }

               /* non-finite coefficients are not allowed in the objective */
               if(LPFisInfinity(pos))
                  goto syntax_error;

               have_value = true;
               val = LPFreadValue<R>(pos, spxout) * pre_sign;
            }

            if(*pos == '\0')
               continue;

            if(!have_value || !LPFisColName(pos))
               goto syntax_error;

            have_value = false;
            colidx = LPFreadColName(pos, cnames, cset, &emptycol, spxout);
            vec.add(colidx, val);
            break;

         case CONSTRAINTS:
            if(LPFisValue(pos))
            {
               R pre_sign = 1.0;

               /* Already having here a value could only result from being the first number in a constraint, or a sign
                * '+' or '-' as last token on the previous line.
                */
               if(have_value)
               {
                  if(NE(spxAbs(val), R(1.0)))
                     goto syntax_error;

                  if(EQ(val, R(-1.0)))
                     pre_sign = val;
               }

               if(LPFisInfinity(pos))
               {
                  /* non-finite coefficients are not allowed */
                  if(sense == 0)
                     goto syntax_error;

                  val = LPFreadInfinity<R>(pos) * pre_sign;
               }
               else
                  val = LPFreadValue<R>(pos, spxout) * pre_sign;

               have_value = true;

               if(sense != 0)
               {
                  if(sense == '<')
                  {
                     row.setLhs(R(-infinity));
                     row.setRhs(val);
                  }
                  else if(sense == '>')
                  {
                     row.setLhs(val);
                     row.setRhs(R(infinity));
                  }
                  else
                  {
                     assert(sense == '=');

                     row.setLhs(val);
                     row.setRhs(val);
                  }

                  row.setRowVector(vec);
                  rset.add(row);
                  vec.clear();

                  if(!unnamed)
                     unnamed = true;
                  else
                  {
                     char name[16];
                     spxSnprintf(name, 16, "C%d", rset.num());
                     rnames->add(name);
                  }

                  have_value = true;
                  val = 1.0;
                  sense = 0;
                  pos = 0;
                  // next line
                  continue;
               }
            }

            if(*pos == '\0')
               continue;

            if(have_value)
            {
               if(LPFisColName(pos))
               {
                  colidx = LPFreadColName(pos, cnames, cset, &emptycol, spxout);

                  if(val != 0.0)
                  {
                     // Do we have this index already in the row?
                     int n = vec.pos(colidx);

                     // if not, add it
                     if(n < 0)
                        vec.add(colidx, val);
                     // if yes, add them up and remove the element if it amounts to zero
                     else
                     {
                        assert(vec.index(n) == colidx);

                        val += vec.value(n);

                        if(val == 0.0)
                           vec.remove(n);
                        else
                           vec.value(n) = val;

                        assert(cnames->has(colidx));

                        MSG_WARNING((*this->spxout), (*this->spxout) << "WLPFRD10 Duplicate index "
                                    << (*cnames)[colidx]
                                    << " in line " << lineno
                                    << std::endl;)
                     }
                  }

                  have_value = false;
               }
               else
               {
                  // We have a row like c1: <= 5 with no variables. We can not handle 10 <= 5; issue a syntax error.
                  if(val != 1.0)
                     goto syntax_error;

                  // If the next thing is not the sense we give up also.
                  if(!LPFisSense(pos))
                     goto syntax_error;

                  have_value = false;
               }
            }

            assert(!have_value);

            if(LPFisSense(pos))
               sense = LPFreadSense(pos);

            break;

         case BOUNDS:
            other = false;
            sense = 0;

            if(LPFisValue(pos))
            {
               val = LPFisInfinity(pos) ? LPFreadInfinity<R>(pos) : LPFreadValue<R>(pos, spxout);

               if(!LPFisSense(pos))
                  goto syntax_error;

               sense = LPFreadSense(pos);
               other = true;
            }

            if(!LPFisColName(pos))
               goto syntax_error;

            if((colidx = LPFreadColName<R>(pos, cnames, cset, nullptr, spxout)) < 0)
            {
               MSG_WARNING((*this->spxout), (*this->spxout) << "WLPFRD11 in Bounds section line "
                           << lineno << " ignored" << std::endl;)
               *pos = '\0';
               continue;
            }

            if(sense)
            {
               if(sense == '<')
                  cset.lower_w(colidx) = val;
               else if(sense == '>')
                  cset.upper_w(colidx) = val;
               else
               {
                  assert(sense == '=');
                  cset.lower_w(colidx) = val;
                  cset.upper_w(colidx) = val;
               }
            }

            if(LPFisFree(pos))
            {
               cset.lower_w(colidx) = R(-infinity);
               cset.upper_w(colidx) =  R(infinity);
               other = true;
               pos += 4;  // set position after the word "free"
            }
            else if(LPFisSense(pos))
            {
               sense = LPFreadSense(pos);
               other = true;

               if(!LPFisValue(pos))
                  goto syntax_error;

               val = LPFisInfinity(pos) ? LPFreadInfinity<R>(pos) : LPFreadValue<R>(pos, spxout);

               if(sense == '<')
                  cset.upper_w(colidx) = val;
               else if(sense == '>')
                  cset.lower_w(colidx) = val;
               else
               {
                  assert(sense == '=');
                  cset.lower_w(colidx) = val;
                  cset.upper_w(colidx) = val;
               }
            }

            /* Do we have only a single column name in the input line?  We could ignore this savely, but it is probably
             * a sign of some other error.
             */
            if(!other)
               goto syntax_error;

            break;

         case BINARIES:
         case INTEGERS:
            if((colidx = LPFreadColName<R>(pos, cnames, cset, 0, spxout)) < 0)
            {
               MSG_WARNING((*this->spxout), (*this->spxout) << "WLPFRD12 in Binary/General section line " << lineno
                           << " ignored" << std::endl;)
            }
            else
            {
               if(section == BINARIES)
               {
                  if(cset.lower(colidx) < 0.0)
                  {
                     cset.lower_w(colidx) = 0.0;
                  }

                  if(cset.upper(colidx) > 1.0)
                  {
                     cset.upper_w(colidx) = 1.0;
                  }
               }

               if(p_intvars != 0)
                  p_intvars->addIdx(colidx);
            }

            break;

         case START:
            MSG_ERROR(std::cerr << "ELPFRD13 This seems to be no LP format file" << std::endl;)
            goto syntax_error;

         default:
            throw SPxInternalCodeException("XLPFRD01 This should never happen.");
         }

         if(pos == pos_old)
            goto syntax_error;
      }
   }

   assert(isConsistent());

   addCols(cset);
   assert(isConsistent());

   addRows(rset);
   assert(isConsistent());

syntax_error:

   if(finished)
   {
      MSG_INFO2((*this->spxout), (*this->spxout) << "Finished reading " << lineno << " lines" <<
                std::endl;)
   }
   else
      MSG_ERROR(std::cerr << "ELPFRD15 Syntax error in line " << lineno << std::endl;)

      if(p_cnames == 0)
         spx_free(cnames);

   if(p_rnames == 0)
      spx_free(rnames);

   return finished;
}



// ---------------------------------------------------------------------------------------------------------------------
// Specialization for reading MPS format
// ---------------------------------------------------------------------------------------------------------------------

/// Process NAME section.
static inline void MPSreadName(MPSInput& mps, SPxOut* spxout)
{
   do
   {
      // This has to be the Line with the NAME section.
      if(!mps.readLine() || (mps.field0() == 0) || strcmp(mps.field0(), "NAME"))
         break;

      // Sometimes the name is omitted.
      mps.setProbName((mps.field1() == 0) ? "_MPS_" : mps.field1());

      MSG_INFO2((*spxout), (*spxout) << "IMPSRD01 Problem name   : " << mps.probName() << std::endl;)

      // This has to be a new section
      if(!mps.readLine() || (mps.field0() == 0))
         break;

      if(!strcmp(mps.field0(), "ROWS"))
         mps.setSection(MPSInput::ROWS);
      else if(!strncmp(mps.field0(), "OBJSEN", 6))
         mps.setSection(MPSInput::OBJSEN);
      else if(!strcmp(mps.field0(), "OBJNAME"))
         mps.setSection(MPSInput::OBJNAME);
      else
         break;

      return;
   }
   while(false);

   mps.syntaxError();
}



/// Process OBJSEN section. This Section is an ILOG extension.
static inline void MPSreadObjsen(MPSInput& mps)
{
   do
   {
      // This has to be the Line with MIN or MAX.
      if(!mps.readLine() || (mps.field1() == 0))
         break;

      if(!strcmp(mps.field1(), "MIN"))
         mps.setObjSense(MPSInput::MINIMIZE);
      else if(!strcmp(mps.field1(), "MAX"))
         mps.setObjSense(MPSInput::MAXIMIZE);
      else
         break;

      // Look for ROWS or OBJNAME Section
      if(!mps.readLine() || (mps.field0() == 0))
         break;

      if(!strcmp(mps.field0(), "ROWS"))
         mps.setSection(MPSInput::ROWS);
      else if(!strcmp(mps.field0(), "OBJNAME"))
         mps.setSection(MPSInput::OBJNAME);
      else
         break;

      return;
   }
   while(false);

   mps.syntaxError();
}



/// Process OBJNAME section. This Section is an ILOG extension.
static inline void MPSreadObjname(MPSInput& mps)
{
   do
   {
      // This has to be the Line with the name.
      if(!mps.readLine() || (mps.field1() == 0))
         break;

      mps.setObjName(mps.field1());

      // Look for ROWS Section
      if(!mps.readLine() || (mps.field0() == 0))
         break;

      if(strcmp(mps.field0(), "ROWS"))
         break;

      mps.setSection(MPSInput::ROWS);

      return;
   }
   while(false);

   mps.syntaxError();
}



/// Process ROWS section.
template <class R>
static void MPSreadRows(MPSInput& mps, LPRowSetBase<R>& rset, NameSet& rnames, SPxOut* spxout)
{
   LPRowBase<R> row;

   while(mps.readLine())
   {
      if(mps.field0() != 0)
      {
         MSG_INFO2((*spxout), (*spxout) << "IMPSRD02 Objective name : " << mps.objName() << std::endl;)

         if(strcmp(mps.field0(), "COLUMNS"))
            break;

         mps.setSection(MPSInput::COLUMNS);

         return;
      }

      if((mps.field1() == 0) || (mps.field2() == 0))
         break;

      if(*mps.field1() == 'N')
      {
         if(*mps.objName() == '\0')
            mps.setObjName(mps.field2());
      }
      else
      {
         if(rnames.has(mps.field2()))
            break;

         rnames.add(mps.field2());

         switch(*mps.field1())
         {
         case 'G':
            row.setLhs(0.0);
            row.setRhs(R(infinity));
            break;

         case 'E':
            row.setLhs(0.0);
            row.setRhs(0.0);
            break;

         case 'L':
            row.setLhs(R(-infinity));
            row.setRhs(0.0);
            break;

         default:
            mps.syntaxError();
            return;
         }

         rset.add(row);
      }

      assert((*mps.field1() == 'N') || (rnames.number(mps.field2()) == rset.num() - 1));
   }

   mps.syntaxError();
}



/// Process COLUMNS section.
template <class R>
static void MPSreadCols(MPSInput& mps, const LPRowSetBase<R>& rset, const NameSet&  rnames,
                        LPColSetBase<R>& cset, NameSet& cnames, DIdxSet* intvars)
{
   R val;
   int idx;
   char colname[MPSInput::MAX_LINE_LEN] = { '\0' };
   LPColBase<R> col(rset.num());
   DSVectorBase<R> vec;

   col.setObj(0.0);
   vec.clear();

   while(mps.readLine())
   {
      if(mps.field0() != 0)
      {
         if(strcmp(mps.field0(), "RHS"))
            break;

         if(colname[0] != '\0')
         {
            col.setColVector(vec);
            cset.add(col);
         }

         mps.setSection(MPSInput::RHS);

         return;
      }

      if((mps.field1() == 0) || (mps.field2() == 0) || (mps.field3() == 0))
         break;

      // new column?
      if(strcmp(colname, mps.field1()))
      {
         // first column?
         if(colname[0] != '\0')
         {
            col.setColVector(vec);
            cset.add(col);
         }

         // save copy of string (make sure string ends with \0)
         spxSnprintf(colname, MPSInput::MAX_LINE_LEN - 1, "%s", mps.field1());
         colname[MPSInput::MAX_LINE_LEN - 1] = '\0';

         int ncnames = cnames.size();
         cnames.add(colname);

         // check whether the new name is unique wrt previous column names
         if(cnames.size() <= ncnames)
         {
            MSG_ERROR(std::cerr << "ERROR in COLUMNS: duplicate column name or not column-wise ordering" <<
                      std::endl;)
            break;
         }

         vec.clear();
         col.setObj(0.0);
         col.setLower(0.0);
         col.setUpper(R(infinity));

         if(mps.isInteger())
         {
            assert(cnames.number(colname) == cset.num());

            if(intvars != 0)
               intvars->addIdx(cnames.number(colname));

            // for Integer variable the default bounds are 0/1
            col.setUpper(1.0);
         }
      }

      val = atof(mps.field3());

      if(!strcmp(mps.field2(), mps.objName()))
         col.setObj(val);
      else
      {
         if((idx = rnames.number(mps.field2())) < 0)
            mps.entryIgnored("Column", mps.field1(), "row", mps.field2());
         else if(val != 0.0)
            vec.add(idx, val);
      }

      if(mps.field5() != 0)
      {
         assert(mps.field4() != 0);

         val = atof(mps.field5());

         if(!strcmp(mps.field4(), mps.objName()))
            col.setObj(val);
         else
         {
            if((idx = rnames.number(mps.field4())) < 0)
               mps.entryIgnored("Column", mps.field1(), "row", mps.field4());
            else if(val != 0.0)
               vec.add(idx, val);
         }
      }
   }

   mps.syntaxError();
}



/// Process RHS section.
template <class R>
static void MPSreadRhs(MPSInput& mps, LPRowSetBase<R>& rset, const NameSet& rnames, SPxOut* spxout)
{
   char rhsname[MPSInput::MAX_LINE_LEN] = { '\0' };
   char addname[MPSInput::MAX_LINE_LEN] = { '\0' };
   int idx;
   R val;

   while(mps.readLine())
   {
      if(mps.field0() != 0)
      {
         MSG_INFO2((*spxout), (*spxout) << "IMPSRD03 RHS name       : " << rhsname  << std::endl;);

         if(!strcmp(mps.field0(), "RANGES"))
            mps.setSection(MPSInput::RANGES);
         else if(!strcmp(mps.field0(), "BOUNDS"))
            mps.setSection(MPSInput::BOUNDS);
         else if(!strcmp(mps.field0(), "ENDATA"))
            mps.setSection(MPSInput::ENDATA);
         else
            break;

         return;
      }

      if(((mps.field2() != 0) && (mps.field3() == 0)) || ((mps.field4() != 0) && (mps.field5() == 0)))
         mps.insertName("_RHS_");

      if((mps.field1() == 0) || (mps.field2() == 0) || (mps.field3() == 0))
         break;

      if(*rhsname == '\0')
         spxSnprintf(rhsname, MPSInput::MAX_LINE_LEN, "%s", mps.field1());

      if(strcmp(rhsname, mps.field1()))
      {
         if(strcmp(addname, mps.field1()))
         {
            assert(strlen(mps.field1()) < MPSInput::MAX_LINE_LEN);
            spxSnprintf(addname, MPSInput::MAX_LINE_LEN, "%s", mps.field1());
            MSG_INFO3((*spxout), (*spxout) << "IMPSRD07 RHS ignored    : " << addname << std::endl);
         }
      }
      else
      {
         if((idx = rnames.number(mps.field2())) < 0)
            mps.entryIgnored("RHS", mps.field1(), "row", mps.field2());
         else
         {
            val = atof(mps.field3());

            // LE or EQ
            if(rset.rhs(idx) < R(infinity))
               rset.rhs_w(idx) = val;

            // GE or EQ
            if(rset.lhs(idx) > R(-infinity))
               rset.lhs_w(idx) = val;
         }

         if(mps.field5() != 0)
         {
            if((idx = rnames.number(mps.field4())) < 0)
               mps.entryIgnored("RHS", mps.field1(), "row", mps.field4());
            else
            {
               val = atof(mps.field5());

               // LE or EQ
               if(rset.rhs(idx) < R(infinity))
                  rset.rhs_w(idx) = val;

               // GE or EQ
               if(rset.lhs(idx) > R(-infinity))
                  rset.lhs_w(idx) = val;
            }
         }
      }
   }

   mps.syntaxError();
}



/// Process RANGES section.
template <class R>
static void MPSreadRanges(MPSInput& mps,  LPRowSetBase<R>& rset, const NameSet& rnames,
                          SPxOut* spxout)
{
   char rngname[MPSInput::MAX_LINE_LEN] = { '\0' };
   int idx;
   R val;

   while(mps.readLine())
   {
      if(mps.field0() != 0)
      {
         MSG_INFO2((*spxout), (*spxout) << "IMPSRD04 Range name     : " << rngname << std::endl;);

         if(!strcmp(mps.field0(), "BOUNDS"))
            mps.setSection(MPSInput::BOUNDS);
         else if(!strcmp(mps.field0(), "ENDATA"))
            mps.setSection(MPSInput::ENDATA);
         else
            break;

         return;
      }

      if(((mps.field2() != 0) && (mps.field3() == 0)) || ((mps.field4() != 0) && (mps.field5() == 0)))
         mps.insertName("_RNG_");

      if((mps.field1() == 0) || (mps.field2() == 0) || (mps.field3() == 0))
         break;

      if(*rngname == '\0')
      {
         assert(strlen(mps.field1()) < MPSInput::MAX_LINE_LEN);
         spxSnprintf(rngname, MPSInput::MAX_LINE_LEN, "%s", mps.field1());
      }

      /* The rules are:
       * Row Sign   LHS             RHS
       * ----------------------------------------
       *  G   +/-   rhs             rhs + |range|
       *  L   +/-   rhs - |range|   rhs
       *  E   +     rhs             rhs + range
       *  E   -     rhs + range     rhs
       * ----------------------------------------
       */
      if(!strcmp(rngname, mps.field1()))
      {
         if((idx = rnames.number(mps.field2())) < 0)
            mps.entryIgnored("Range", mps.field1(), "row", mps.field2());
         else
         {
            val = atof(mps.field3());

            // EQ
            if((rset.lhs(idx) > R(-infinity)) && (rset.rhs_w(idx) <  R(infinity)))
            {
               assert(rset.lhs(idx) == rset.rhs(idx));

               if(val >= 0)
                  rset.rhs_w(idx) += val;
               else
                  rset.lhs_w(idx) += val;
            }
            else
            {
               // GE
               if(rset.lhs(idx) > R(-infinity))
                  rset.rhs_w(idx)  = rset.lhs(idx) + spxAbs(val);
               // LE
               else
                  rset.lhs_w(idx)  = rset.rhs(idx) - spxAbs(val);
            }
         }

         if(mps.field5() != 0)
         {
            if((idx = rnames.number(mps.field4())) < 0)
               mps.entryIgnored("Range", mps.field1(), "row", mps.field4());
            else
            {
               val = atof(mps.field5());

               // EQ
               if((rset.lhs(idx) > R(-infinity)) && (rset.rhs(idx) <  R(infinity)))
               {
                  assert(rset.lhs(idx) == rset.rhs(idx));

                  if(val >= 0)
                     rset.rhs_w(idx) += val;
                  else
                     rset.lhs_w(idx) += val;
               }
               else
               {
                  // GE
                  if(rset.lhs(idx) > R(-infinity))
                     rset.rhs_w(idx)  = rset.lhs(idx) + spxAbs(val);
                  // LE
                  else
                     rset.lhs_w(idx)  = rset.rhs(idx) - spxAbs(val);
               }
            }
         }
      }
   }

   mps.syntaxError();
}



/// Process BOUNDS section.
template <class R>
static void MPSreadBounds(MPSInput& mps, LPColSetBase<R>& cset, const NameSet& cnames,
                          DIdxSet* intvars, SPxOut* spxout)
{
   DIdxSet oldbinvars;
   char bndname[MPSInput::MAX_LINE_LEN] = { '\0' };
   int  idx;
   R val;

   while(mps.readLine())
   {
      if(mps.field0() != 0)
      {
         MSG_INFO2((*spxout), (*spxout) << "IMPSRD05 Bound name     : " << bndname << std::endl;)

         if(strcmp(mps.field0(), "ENDATA"))
            break;

         mps.setSection(MPSInput::ENDATA);

         return;
      }

      // Is the value field used ?
      if((!strcmp(mps.field1(), "LO"))
            || (!strcmp(mps.field1(), "UP"))
            || (!strcmp(mps.field1(), "FX"))
            || (!strcmp(mps.field1(), "LI"))
            || (!strcmp(mps.field1(), "UI")))
      {
         if((mps.field3() != 0) && (mps.field4() == 0))
            mps.insertName("_BND_", true);
      }
      else
      {
         if((mps.field2() != 0) && (mps.field3() == 0))
            mps.insertName("_BND_", true);
      }

      if((mps.field1() == 0) || (mps.field2() == 0) || (mps.field3() == 0))
         break;

      if(*bndname == '\0')
      {
         assert(strlen(mps.field2()) < MPSInput::MAX_LINE_LEN);
         spxSnprintf(bndname, MPSInput::MAX_LINE_LEN, "%s", mps.field2());
      }

      // Only read the first Bound in section
      if(!strcmp(bndname, mps.field2()))
      {
         if((idx = cnames.number(mps.field3())) < 0)
            mps.entryIgnored("column", mps.field3(), "bound", bndname);
         else
         {
            if(mps.field4() == 0)
               val = 0.0;
            else if(!strcmp(mps.field4(), "-Inf") || !strcmp(mps.field4(), "-inf"))
               val = R(-infinity);
            else if(!strcmp(mps.field4(), "Inf") || !strcmp(mps.field4(), "inf")
                    || !strcmp(mps.field4(), "+Inf") || !strcmp(mps.field4(), "+inf"))
               val = R(infinity);
            else
               val = atof(mps.field4());

            // ILOG extension (Integer Bound)
            if(mps.field1()[1] == 'I')
            {
               if(intvars != 0)
                  intvars->addIdx(idx);

               // if the variable has appeared in the MARKER section of the COLUMNS section then its default bounds were
               // set to 0,1; the first time it is declared integer we need to change to default bounds 0,R(infinity)
               if(oldbinvars.pos(idx) < 0)
               {
                  cset.upper_w(idx) = R(infinity);
                  oldbinvars.addIdx(idx);
               }
            }

            switch(*mps.field1())
            {
            case 'L':
               cset.lower_w(idx) = val;
               break;

            case 'U':
               cset.upper_w(idx) = val;
               break;

            case 'F':
               if(mps.field1()[1] == 'X')
               {
                  cset.lower_w(idx) = val;
                  cset.upper_w(idx) = val;
               }
               else
               {
                  cset.lower_w(idx) = R(-infinity);
                  cset.upper_w(idx) = R(infinity);
               }

               break;

            case 'M':
               cset.lower_w(idx) = R(-infinity);
               break;

            case 'P':
               cset.upper_w(idx) = R(infinity);
               break;

            // Ilog extension (Binary)
            case 'B':
               cset.lower_w(idx) = 0.0;
               cset.upper_w(idx) = 1.0;

               if(intvars != 0)
                  intvars->addIdx(idx);

               break;

            default:
               mps.syntaxError();
               return;
            }
         }
      }
   }

   mps.syntaxError();
}



/// Read LP in MPS File Format.
/**
 *  The specification is taken from the IBM Optimization Library Guide and Reference, online available at
 *  http://www.software.ibm.com/sos/features/libuser.htm and from the ILOG CPLEX 7.0 Reference Manual, Appendix E, Page
 *  531.
 *
 *  This routine should read all valid MPS format files.  What it will not do, is find all cases where a file is ill
 *  formed.  If this happens it may complain and read nothing or read "something".
 *
 *  @return true if the file was read correctly.
 */
const int Init_Cols = 10000; ///< initialy allocated columns.
const int Init_NZos = 100000; ///< initialy allocated non zeros.
template <class R> inline
bool SPxLPBase<R>::readMPS(
   std::istream& p_input,           ///< input stream.
   NameSet*      p_rnames,          ///< row names.
   NameSet*      p_cnames,          ///< column names.
   DIdxSet*      p_intvars)         ///< integer variables.
{
   LPRowSetBase<R>& rset = *this;
   LPColSetBase<R>& cset = *this;
   NameSet* rnames;
   NameSet* cnames;

   if(p_cnames)
      cnames = p_cnames;
   else
   {
      cnames = 0;
      spx_alloc(cnames);
      cnames = new(cnames) NameSet();
   }

   cnames->clear();

   if(p_rnames)
      rnames = p_rnames;
   else
   {
      try
      {
         rnames = 0;
         spx_alloc(rnames);
         rnames = new(rnames) NameSet();
      }
      catch(const SPxMemoryException& x)
      {
         if(!p_cnames)
         {
            cnames->~NameSet();
            spx_free(cnames);
         }

         throw x;
      }
   }

   rnames->clear();

   SPxLPBase<R>::clear(); // clear the LP.

   cset.memRemax(Init_NZos);
   cset.reMax(Init_Cols);

   MPSInput mps(p_input);

   MPSreadName(mps, spxout);

   if(mps.section() == MPSInput::OBJSEN)
      MPSreadObjsen(mps);

   if(mps.section() == MPSInput::OBJNAME)
      MPSreadObjname(mps);

   if(mps.section() == MPSInput::ROWS)
      MPSreadRows(mps, rset, *rnames, spxout);

   addedRows(rset.num());

   if(mps.section() == MPSInput::COLUMNS)
      MPSreadCols(mps, rset, *rnames, cset, *cnames, p_intvars);

   if(mps.section() == MPSInput::RHS)
      MPSreadRhs(mps, rset, *rnames, spxout);

   if(mps.section() == MPSInput::RANGES)
      MPSreadRanges(mps, rset, *rnames, spxout);

   if(mps.section() == MPSInput::BOUNDS)
      MPSreadBounds(mps, cset, *cnames, p_intvars, spxout);

   if(mps.section() != MPSInput::ENDATA)
      mps.syntaxError();

   if(mps.hasError())
      clear();
   else
   {
      changeSense(mps.objSense() == MPSInput::MINIMIZE ? SPxLPBase<R>::MINIMIZE : SPxLPBase<R>::MAXIMIZE);

      MSG_INFO2((*spxout), (*spxout) << "IMPSRD06 Objective sense: " << ((mps.objSense() ==
                MPSInput::MINIMIZE) ? "Minimize\n" : "Maximize\n"));

      added2Set(
         *(reinterpret_cast<SVSetBase<R>*>(static_cast<LPRowSetBase<R>*>(this))),
         *(reinterpret_cast<SVSetBase<R>*>(static_cast<LPColSetBase<R>*>(this))),
         cset.num());
      addedCols(cset.num());

      assert(isConsistent());
   }

   if(p_cnames == 0)
   {
      cnames->~NameSet();
      spx_free(cnames);
   }

   if(p_rnames == 0)
   {
      rnames->~NameSet();
      spx_free(rnames);
   }

   return !mps.hasError();
}



// ---------------------------------------------------------------------------------------------------------------------
// Specialization for writing LP format
// ---------------------------------------------------------------------------------------------------------------------

// get the name of a row or construct one
template <class R>
static const char* LPFgetRowName(
   const SPxLPBase<R>& p_lp,
   int                    p_idx,
   const NameSet*         p_rnames,
   char*                  p_buf,
   int                    p_num_written_rows
)
{
   assert(p_buf != 0);
   assert(p_idx >= 0);
   assert(p_idx <  p_lp.nRows());

   if(p_rnames != 0)
   {
      DataKey key = p_lp.rId(p_idx);

      if(p_rnames->has(key))
         return (*p_rnames)[key];
   }

   spxSnprintf(p_buf, 16, "C%d", p_num_written_rows);

   return p_buf;
}



// get the name of a column or construct one
template <class R>
static const char* getColName(
   const SPxLPBase<R>& p_lp,
   int                    p_idx,
   const NameSet*         p_cnames,
   char*                  p_buf
)
{
   assert(p_buf != 0);
   assert(p_idx >= 0);
   assert(p_idx <  p_lp.nCols());

   if(p_cnames != 0)
   {
      DataKey key = p_lp.cId(p_idx);

      if(p_cnames->has(key))
         return (*p_cnames)[key];
   }

   spxSnprintf(p_buf, 16, "x%d", p_idx);

   return p_buf;
}



// write an SVectorBase<R>
#define NUM_ENTRIES_PER_LINE 5
template <class R>
static void LPFwriteSVector(
   const SPxLPBase<R>&   p_lp,       ///< the LP
   std::ostream&            p_output,   ///< output stream
   const NameSet*           p_cnames,   ///< column names
   const SVectorBase<R>& p_svec)     ///< vector to write
{

   char name[16];
   int num_coeffs = 0;

   for(int j = 0; j < p_lp.nCols(); ++j)
   {
      const R coeff = p_svec[j];

      if(coeff == 0)
         continue;

      if(num_coeffs == 0)
         p_output << coeff << " " << getColName(p_lp, j, p_cnames, name);
      else
      {
         // insert a line break every NUM_ENTRIES_PER_LINE columns
         if(num_coeffs % NUM_ENTRIES_PER_LINE == 0)
            p_output << "\n\t";

         if(coeff < 0)
            p_output << " - " << -coeff;
         else
            p_output << " + " << coeff;

         p_output << " " << getColName(p_lp, j, p_cnames, name);
      }

      ++num_coeffs;
   }
}



// write the objective
template <class R>
static void LPFwriteObjective(
   const SPxLPBase<R>& p_lp,       ///< the LP
   std::ostream&          p_output,   ///< output stream
   const NameSet*         p_cnames    ///< column names
)
{

   const int sense = p_lp.spxSense();

   p_output << ((sense == SPxLPBase<R>::MINIMIZE) ? "Minimize\n" : "Maximize\n");
   p_output << "  obj: ";

   const VectorBase<R>& obj = p_lp.maxObj();
   DSVectorBase<R> svec(obj.dim());
   svec.operator = (obj);
   svec *= R(sense);
   LPFwriteSVector(p_lp, p_output, p_cnames, svec);
   p_output << "\n";
}



// write non-ranged rows
template <class R>
static void LPFwriteRow(
   const SPxLPBase<R>&   p_lp,       ///< the LP
   std::ostream&            p_output,   ///< output stream
   const NameSet*           p_cnames,   ///< column names
   const SVectorBase<R>& p_svec,     ///< vector of the row
   const R&              p_lhs,      ///< lhs of the row
   const R&              p_rhs       ///< rhs of the row
)
{

   LPFwriteSVector(p_lp, p_output, p_cnames, p_svec);

   if(p_lhs == p_rhs)
      p_output << " = " << p_rhs;
   else if(p_lhs <= R(-infinity))
      p_output << " <= " << p_rhs;
   else
   {
      assert(p_rhs >= R(infinity));
      p_output << " >= " << p_lhs;
   }

   p_output << "\n";
}



// write all rows
template <class R>
static void LPFwriteRows(
   const SPxLPBase<R>& p_lp,       ///< the LP
   std::ostream&          p_output,   ///< output stream
   const NameSet*         p_rnames,   ///< row names
   const NameSet*         p_cnames   ///< column names
)
{

   char name[16];

   p_output << "Subject To\n";

   for(int i = 0; i < p_lp.nRows(); ++i)
   {
      const R lhs = p_lp.lhs(i);
      const R rhs = p_lp.rhs(i);

      if(lhs > R(-infinity) && rhs < R(infinity) && lhs != rhs)
      {
         // ranged row -> write two non-ranged rows
         p_output << " " << LPFgetRowName(p_lp, i, p_rnames, name, i) << "_1 : ";
         LPFwriteRow(p_lp, p_output, p_cnames, p_lp.rowVector(i), lhs, R(infinity));

         p_output << " " << LPFgetRowName(p_lp, i, p_rnames, name, i) << "_2 : ";
         LPFwriteRow(p_lp, p_output, p_cnames, p_lp.rowVector(i), R(-infinity), rhs);
      }
      else
      {
         p_output << " " << LPFgetRowName(p_lp, i, p_rnames, name, i) << " : ";
         LPFwriteRow(p_lp, p_output, p_cnames, p_lp.rowVector(i), lhs, rhs);
      }
   }
}



// write the variable bounds
// (the default bounds 0 <= x <= R(infinity) are not written)
template <class R>
static void LPFwriteBounds(
   const SPxLPBase<R>&   p_lp,       ///< the LP to write
   std::ostream&            p_output,   ///< output stream
   const NameSet*           p_cnames    ///< column names
)
{

   char name[16];

   p_output << "Bounds\n";

   for(int j = 0; j < p_lp.nCols(); ++j)
   {
      const R lower = p_lp.lower(j);
      const R upper = p_lp.upper(j);

      if(lower == upper)
      {
         p_output << "  "   << getColName(p_lp, j, p_cnames, name) << " = "  << upper << '\n';
      }
      else if(lower > R(-infinity))
      {
         if(upper < R(infinity))
         {
            // range bound
            if(lower != 0)
               p_output << "  "   << lower << " <= "
                        << getColName(p_lp, j, p_cnames, name)
                        << " <= " << upper << '\n';
            else
               p_output << "  "   << getColName(p_lp, j, p_cnames, name)
                        << " <= " << upper << '\n';
         }
         else if(lower != 0)
            p_output << "  " << lower << " <= "
                     << getColName(p_lp, j, p_cnames, name)
                     << '\n';
      }
      else if(upper < R(infinity))
         p_output << "   -Inf <= "
                  << getColName(p_lp, j, p_cnames, name)
                  << " <= " << upper << '\n';
      else
         p_output << "  "   << getColName(p_lp, j, p_cnames, name)
                  << " free\n";
   }
}



// write the generals section
template <class R>
static void LPFwriteGenerals(
   const SPxLPBase<R>&   p_lp,         ///< the LP to write
   std::ostream&            p_output,     ///< output stream
   const NameSet*           p_cnames,     ///< column names
   const DIdxSet*           p_intvars     ///< integer variables
)
{

   char name[16];

   if(p_intvars == NULL || p_intvars->size() <= 0)
      return;  // no integer variables

   p_output << "Generals\n";

   for(int j = 0; j < p_lp.nCols(); ++j)
      if(p_intvars->pos(j) >= 0)
         p_output << "  " << getColName(p_lp, j, p_cnames, name) << "\n";
}



/// Write LP in LP Format.
template <class R> inline
void SPxLPBase<R>::writeLPF(
   std::ostream&  p_output,          ///< output stream
   const NameSet* p_rnames,          ///< row names
   const NameSet* p_cnames,          ///< column names
   const DIdxSet* p_intvars          ///< integer variables
) const
{
   SPxOut::setScientific(p_output, 16);

   LPFwriteObjective(*this, p_output, p_cnames);
   LPFwriteRows(*this, p_output, p_rnames, p_cnames);
   LPFwriteBounds(*this, p_output, p_cnames);
   LPFwriteGenerals(*this, p_output, p_cnames, p_intvars);

   p_output << "End" << std::endl;
}



// ---------------------------------------------------------------------------------------------------------------------
// Specialization for writing MPS format
// ---------------------------------------------------------------------------------------------------------------------

template <class R>
static void MPSwriteRecord(
   std::ostream&  os,
   const char*    indicator,
   const char*    name,
   const char*    name1  = nullptr,
   const R     value1 = 0.0,
   const char*    name2  = nullptr,
   const R     value2 = 0.0
)
{
   char buf[81];

   spxSnprintf(buf, sizeof(buf), " %-2.2s %-8.8s", (indicator == 0) ? "" : indicator,
               (name == 0)      ? "" : name);
   os << buf;

   if(name1 != nullptr)
   {
      spxSnprintf(buf, sizeof(buf), "%-8.8s  %.15" REAL_FORMAT, name1, (Real) value1);
      os << buf;

      if(name2 != 0)
      {
         spxSnprintf(buf, sizeof(buf), "   %-8.8s  %.15" REAL_FORMAT, name2, (Real) value2);
         os << buf;
      }
   }

   os << std::endl;
}



template <class R>
static R MPSgetRHS(R left, R right)
{
   R rhsval;

   if(left > R(-infinity))   /// This includes ranges
      rhsval = left;
   else if(right <  R(infinity))
      rhsval = right;
   else
      throw SPxInternalCodeException("XMPSWR01 This should never happen.");

   return rhsval;
}


template <class R>
static const char* MPSgetRowName(
   const SPxLPBase<R>& lp,
   int                   idx,
   const NameSet*        rnames,
   char*                 buf
)
{
   assert(buf != 0);
   assert(idx >= 0);
   assert(idx <  lp.nRows());

   if(rnames != 0)
   {
      DataKey key = lp.rId(idx);

      if(rnames->has(key))
         return (*rnames)[key];
   }

   spxSnprintf(buf, 16, "C%d", idx);

   return buf;
}



/// Write LP in MPS format.
/** @note There will always be a BOUNDS section, even if there are no bounds.
 */
template <class R> inline
void SPxLPBase<R>::writeMPS(
   std::ostream&  p_output,          ///< output stream.
   const NameSet* p_rnames,          ///< row names.
   const NameSet* p_cnames,          ///< column names.
   const DIdxSet* p_intvars          ///< integer variables.
) const
{

   const char*    indicator;
   char           name [16];
   char           name1[16];
   char           name2[16];
   bool           has_ranges = false;
   int            i;
   int            k;

   SPxOut::setScientific(p_output, 16);
   // --- NAME Section ---
   p_output << "NAME          MPSDATA" << std::endl;

   // --- ROWS Section ---
   p_output << "ROWS" << std::endl;

   for(i = 0; i < nRows(); i++)
   {
      if(lhs(i) == rhs(i))
         indicator = "E";
      else if((lhs(i) > R(-infinity)) && (rhs(i) < R(infinity)))
      {
         indicator = "E";
         has_ranges = true;
      }
      else if(lhs(i) > R(-infinity))
         indicator = "G";
      else if(rhs(i) <  R(infinity))
         indicator = "L";
      else
         throw SPxInternalCodeException("XMPSWR02 This should never happen.");

      MPSwriteRecord<R>(p_output, indicator, MPSgetRowName(*this, i, p_rnames, name));
   }

   MPSwriteRecord<R>(p_output, "N", "MINIMIZE");

   // --- COLUMNS Section ---
   p_output << "COLUMNS" << std::endl;

   bool has_intvars = (p_intvars != 0) && (p_intvars->size() > 0);

   for(int j = 0; j < (has_intvars ? 2 : 1); j++)
   {
      bool is_intrun = has_intvars && (j == 1);

      if(is_intrun)
         p_output << "    MARK0001  'MARKER'                 'INTORG'" << std::endl;

      for(i = 0; i < nCols(); i++)
      {
         bool is_intvar = has_intvars && (p_intvars->pos(i) >= 0);

         if((is_intrun && !is_intvar) || (!is_intrun &&  is_intvar))
            continue;

         const SVectorBase<R>& col = colVector(i);
         int colsize2 = (col.size() / 2) * 2;

         assert(colsize2 % 2 == 0);

         for(k = 0; k < colsize2; k += 2)
            MPSwriteRecord(p_output, 0, getColName(*this, i, p_cnames, name),
                           MPSgetRowName(*this, col.index(k), p_rnames, name1), col.value(k),
                           MPSgetRowName(*this, col.index(k + 1), p_rnames, name2), col.value(k + 1));

         if(colsize2 != col.size())
            MPSwriteRecord(p_output, 0, getColName(*this, i, p_cnames, name),
                           MPSgetRowName(*this, col.index(k), p_rnames, name1), col.value(k));

         if(isNotZero(maxObj(i)))
            MPSwriteRecord(p_output, 0, getColName(*this, i, p_cnames, name), "MINIMIZE", -maxObj(i));
      }

      if(is_intrun)
         p_output << "    MARK0001  'MARKER'                 'INTEND'" << std::endl;
   }

   // --- RHS Section ---
   p_output << "RHS" << std::endl;

   i = 0;

   while(i < nRows())
   {
      R rhsval1 = 0.0;
      R rhsval2 = 0.0;

      for(; i < nRows(); i++)
         if((rhsval1 = MPSgetRHS(lhs(i), rhs(i))) != 0.0)
            break;

      if(i < nRows())
      {
         for(k = i + 1; k < nRows(); k++)
         {
            if((rhsval2 = MPSgetRHS(lhs(k), rhs(k))) != 0.0)
               break;
         }

         if(k < nRows())
         {
            MPSwriteRecord(p_output, 0, "RHS", MPSgetRowName(*this, i, p_rnames, name1), rhsval1,
                           MPSgetRowName(*this, k, p_rnames, name2), rhsval2);
         }
         else
            MPSwriteRecord(p_output, 0, "RHS", MPSgetRowName(*this, i, p_rnames, name1), rhsval1);

         i = k + 1;
      }
   }

   // --- RANGES Section ---
   if(has_ranges)
   {
      p_output << "RANGES" << std::endl;

      for(i = 0; i < nRows(); i++)
      {
         if((lhs(i) > R(-infinity)) && (rhs(i) < R(infinity)))
            MPSwriteRecord(p_output, "", "RANGE", MPSgetRowName(*this, i, p_rnames, name1), rhs(i) - lhs(i));
      }
   }

   // --- BOUNDS Section ---
   p_output << "BOUNDS" << std::endl;

   for(i = 0; i < nCols(); i++)
   {
      // skip variables that do not appear in the objective function or any constraint
      const SVectorBase<R>& col = colVector(i);

      if(col.size() == 0 && isZero(maxObj(i)))
         continue;

      if(lower(i) == upper(i))
      {
         MPSwriteRecord(p_output, "FX", "BOUND", getColName(*this, i, p_cnames, name1), lower(i));
         continue;
      }

      if((lower(i) <= R(-infinity)) && (upper(i) >= R(infinity)))
      {
         MPSwriteRecord<R>(p_output, "FR", "BOUND", getColName(*this, i, p_cnames, name1));
         continue;
      }

      if(lower(i) != 0.0)
      {
         if(lower(i) > R(-infinity))
            MPSwriteRecord(p_output, "LO", "BOUND", getColName(*this, i, p_cnames, name1), lower(i));
         else
            MPSwriteRecord<R>(p_output, "MI", "BOUND", getColName(*this, i, p_cnames, name1));
      }

      if(has_intvars && (p_intvars->pos(i) >= 0))
      {
         // Integer variables have default upper bound 1.0, but we should write
         // it nevertheless since CPLEX seems to assume R(infinity) otherwise.
         MPSwriteRecord(p_output, "UP", "BOUND", getColName(*this, i, p_cnames, name1), upper(i));
      }
      else
      {
         // Continous variables have default upper bound R(infinity)
         if(upper(i) < R(infinity))
            MPSwriteRecord(p_output, "UP", "BOUND", getColName(*this, i, p_cnames, name1), upper(i));
      }
   }

   // --- ENDATA Section ---
   p_output << "ENDATA" << std::endl;

   // Output warning when writing a maximisation problem
   if(spxSense() == SPxLPBase<R>::MAXIMIZE)
   {
      MSG_WARNING((*spxout), (*spxout) <<
                  "XMPSWR03 Warning: objective function inverted when writing maximization problem in MPS file format\n");
   }
}



/// Building the dual problem from a given LP
/// @note primalRows must be as large as the number of unranged primal rows + 2 * the number of ranged primal rows.
///       dualCols must have the identical size to the primal rows.
template <class R> inline
void SPxLPBase<R>::buildDualProblem(SPxLPBase<R>& dualLP, SPxRowId primalRowIds[],
                                    SPxColId primalColIds[],
                                    SPxRowId dualRowIds[], SPxColId dualColIds[], int* nprimalrows, int* nprimalcols, int* ndualrows,
                                    int* ndualcols)
{
   // Setting up the primalrowids and dualcolids arrays if not given as parameters
   if(primalRowIds == 0 || primalColIds == 0 || dualRowIds == 0 || dualColIds == 0)
   {
      DataArray < SPxRowId > primalrowids(2 * nRows());
      DataArray < SPxColId > primalcolids(2 * nCols());
      DataArray < SPxRowId > dualrowids(2 * nCols());
      DataArray < SPxColId > dualcolids(2 * nRows());
      int numprimalrows = 0;
      int numprimalcols = 0;
      int numdualrows = 0;
      int numdualcols = 0;

      buildDualProblem(dualLP, primalrowids.get_ptr(), primalcolids.get_ptr(), dualrowids.get_ptr(),
                       dualcolids.get_ptr(), &numprimalrows, &numprimalcols, &numdualrows, &numdualcols);

      if(primalRowIds != 0)
      {
         primalRowIds = primalrowids.get_ptr();
         (*nprimalrows) = numprimalrows;
      }

      if(primalColIds != 0)
      {
         primalColIds = primalcolids.get_ptr();
         (*nprimalcols) = numprimalcols;
      }

      if(dualRowIds != 0)
      {
         dualRowIds = dualrowids.get_ptr();
         (*ndualrows) = numdualrows;
      }

      if(dualColIds != 0)
      {
         dualColIds = dualcolids.get_ptr();
         (*ndualcols) = numdualcols;
      }

      return;
   }

   // setting the sense of the dual LP
   if(spxSense() == MINIMIZE)
      dualLP.changeSense(MAXIMIZE);
   else
      dualLP.changeSense(MINIMIZE);

   LPRowSetBase<R> dualrows(nCols());
   LPColSetBase<R> dualcols(nRows());
   DSVectorBase<R> col(1);

   int numAddedRows = 0;
   int numVarBoundCols = 0;
   int primalrowsidx = 0;
   int primalcolsidx = 0;

   for(int i = 0; i < nCols(); ++i)
   {
      primalColIds[primalcolsidx] = cId(i);
      primalcolsidx++;

      if(lower(i) <= R(-infinity) && upper(i) >= R(infinity))   // unrestricted variables
      {
         dualrows.create(0, obj(i), obj(i));
         numAddedRows++;
      }
      else if(lower(i) <= R(-infinity))   // no lower bound is set, indicating a <= 0 variable
      {
         if(isZero(upper(i)))   // standard bound variable
         {
            if(spxSense() == MINIMIZE)
               dualrows.create(0, obj(i), R(infinity));
            else
               dualrows.create(0, R(-infinity), obj(i));
         }
         else // additional upper bound on the variable
         {
            col.add(numAddedRows, 1.0);

            if(spxSense() == MINIMIZE)
            {
               dualrows.create(0, obj(i), obj(i));
               dualcols.add(upper(i), R(-infinity), col, 0.0);
            }
            else
            {
               dualrows.create(0, obj(i), obj(i));
               dualcols.add(upper(i), 0.0, col, R(infinity));
            }

            col.clear();

            numVarBoundCols++;
         }

         numAddedRows++;
      }
      else if(upper(i) >= R(infinity))   // no upper bound set, indicating a >= 0 variable
      {
         if(isZero(lower(i)))   // standard bound variable
         {
            if(spxSense() == MINIMIZE)
               dualrows.create(0, R(-infinity), obj(i));
            else
               dualrows.create(0, obj(i), R(infinity));
         }
         else // additional lower bound on the variable
         {
            col.add(numAddedRows, 1.0);

            if(spxSense() == MINIMIZE)
            {
               dualrows.create(0, obj(i), obj(i));
               dualcols.add(lower(i), 0.0, col, R(infinity));
            }
            else
            {
               dualrows.create(0, obj(i), obj(i));
               dualcols.add(lower(i), R(-infinity), col, 0.0);
            }

            col.clear();

            numVarBoundCols++;
         }

         numAddedRows++;
      }
      else if(NE(lower(i), upper(i)))   // a boxed variable
      {
         if(isZero(lower(i)))   // variable bounded between 0 and upper(i)
         {
            col.add(numAddedRows, 1.0);

            if(spxSense() == MINIMIZE)
            {
               dualrows.create(0, R(-infinity), obj(i));
               dualcols.add(upper(i), R(-infinity), col, 0.0);
            }
            else
            {
               dualrows.create(0, obj(i), R(infinity));
               dualcols.add(upper(i), 0.0, col, R(infinity));
            }

            col.clear();

            numVarBoundCols++;
         }
         else if(isZero(upper(i)))   // variable bounded between lower(i) and 0
         {
            col.add(numAddedRows, 1.0);

            if(spxSense() == MINIMIZE)
            {
               dualrows.create(0, obj(i), R(infinity));
               dualcols.add(lower(i), 0.0, col, R(infinity));
            }
            else
            {
               dualrows.create(0, R(-infinity), obj(i));
               dualcols.add(lower(i), R(-infinity), col, 0.0);
            }

            col.clear();

            numVarBoundCols++;
         }
         else // variable bounded between lower(i) and upper(i)
         {
            dualrows.create(0, obj(i), obj(i));

            col.add(numAddedRows, 1.0);

            if(spxSense() == MINIMIZE)
            {
               dualcols.add(lower(i), 0.0, col, R(infinity));
               dualcols.add(upper(i), R(-infinity), col, 0.0);
            }
            else
            {
               dualcols.add(lower(i), R(-infinity), col, 0.0);
               dualcols.add(upper(i), 0.0, col, R(infinity));
            }

            col.clear();

            numVarBoundCols += 2;
         }

         numAddedRows++;
      }
      else
      {
         assert(lower(i) == upper(i));

         dualrows.create(0, obj(i), obj(i));

         col.add(numAddedRows, 1.0);
         dualcols.add(lower(i), 0, col, R(infinity));
         dualcols.add(lower(i), R(-infinity), col, 0);
         col.clear();

         numVarBoundCols += 2;
         numAddedRows++;
      }
   }

   // adding the empty rows to the dual LP
   dualLP.addRows(dualrows);

   // setting the dual row ids for the related primal cols.
   // this assumes that the rows are added in sequential order.
   for(int i = 0; i < primalcolsidx; i++)
      dualRowIds[i] = dualLP.rId(i);

   (*nprimalcols) = primalcolsidx;
   (*ndualrows) = primalcolsidx;

   // iterating over each of the rows to create dual columns
   for(int i = 0; i < nRows(); ++i)
   {
      // checking the type of the row
      switch(rowType(i))
      {
      case LPRowBase<R>::RANGE: // range constraint, requires the addition of two dual variables
         assert(lhs(i) > R(-infinity));
         assert(rhs(i) < R(infinity));

         if(spxSense() == MINIMIZE)
         {
            primalRowIds[primalrowsidx] = rId(i); // setting the rowid for the primal row
            primalrowsidx++;
            dualcols.add(lhs(i), 0.0, rowVector(i), R(infinity));

            primalRowIds[primalrowsidx] = rId(i); // setting the rowid for the primal row
            primalrowsidx++;
            dualcols.add(rhs(i), R(-infinity), rowVector(i), 0.0);
         }
         else
         {
            primalRowIds[primalrowsidx] = rId(i); // setting the rowid for the primal row
            primalrowsidx++;
            dualcols.add(lhs(i), R(-infinity), rowVector(i), 0.0);

            primalRowIds[primalrowsidx] = rId(i); // setting the rowid for the primal row
            primalrowsidx++;
            dualcols.add(rhs(i), 0.0, rowVector(i), R(infinity));
         }

         break;

      case LPRowBase<R>::GREATER_EQUAL: // >= constraint
         assert(lhs(i) > R(-infinity));
         primalRowIds[primalrowsidx] = rId(i); // setting the rowid for the primal row
         primalrowsidx++;

         if(spxSense() == MINIMIZE)
            dualcols.add(lhs(i), 0.0, rowVector(i), R(infinity));
         else
            dualcols.add(lhs(i), R(-infinity), rowVector(i), 0.0);

         break;

      case LPRowBase<R>::LESS_EQUAL: // <= constriant
         assert(rhs(i) < R(infinity));
         primalRowIds[primalrowsidx] = rId(i); // setting the rowid for the primal row
         primalrowsidx++;

         if(spxSense() == MINIMIZE)
            dualcols.add(rhs(i), R(-infinity), rowVector(i), 0.0);
         else
            dualcols.add(rhs(i), 0.0, rowVector(i), R(infinity));

         break;

      case LPRowBase<R>::EQUAL: // Equality constraint
         assert(EQ(lhs(i), rhs(i)));
         primalRowIds[primalrowsidx] = rId(i); // setting the rowid for the primal row
         primalrowsidx++;
         dualcols.add(rhs(i), R(-infinity), rowVector(i), R(infinity));
         break;

      default:
         throw SPxInternalCodeException("XLPFRD01 This should never happen.");
      }
   }

   // adding the filled columns to the dual LP
   dualLP.addCols(dualcols);

   // setting the dual column ids for the related primal rows.
   // this assumes that the columns are added in sequential order.
   for(int i = 0; i < primalrowsidx; i++)
      dualColIds[i] = dualLP.cId(i + numVarBoundCols);

   (*nprimalrows) = primalrowsidx;
   (*ndualcols) = primalrowsidx;
}

} // namespace soplex
