/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program                         */
/*          GCG --- Generic Column Generation                                */
/*                  a Dantzig-Wolfe decomposition based extension            */
/*                  of the branch-cut-and-price framework                    */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/* Copyright (C) 2010-2022 Operations Research, RWTH Aachen University       */
/*                         Zuse Institute Berlin (ZIB)                       */
/*                                                                           */
/* This program is free software; you can redistribute it and/or             */
/* modify it under the terms of the GNU Lesser General Public License        */
/* as published by the Free Software Foundation; either version 3            */
/* of the License, or (at your option) any later version.                    */
/*                                                                           */
/* This program is distributed in the hope that it will be useful,           */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/* GNU Lesser General Public License for more details.                       */
/*                                                                           */
/* You should have received a copy of the GNU Lesser General Public License  */
/* along with this program; if not, write to the Free Software               */
/* Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.*/
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   class_indexpartition.cpp
 * @brief  generalization of ConsPartition and VarPartition
 * @author Julius Hense
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "class_indexpartition.h"

#include <cassert>
#include <sstream>
#include <algorithm>

namespace gcg {

/** local methods */

struct sort_pred {
    bool operator()(const std::pair<int,int> &left, const std::pair<int,int> &right) {
        return left.second < right.second;
    }
};

/** constructor */
IndexPartition::IndexPartition(
   SCIP*          _scip,
   const char*    givenName,
   int            givenNClasses,
   int            givenNIndices
) :
   scip(_scip),  name(std::string(givenName)), nClasses(givenNClasses), nIndices(givenNIndices), indicesToClasses(givenNIndices, -1),
   classNames(givenNClasses, ""),
   classDescriptions(givenNClasses, ""),
   classDecompInfo( nClasses, 0 )
{
}

/** copy constructor */
IndexPartition::IndexPartition(const IndexPartition* toCopy)
{
   assert( toCopy != NULL );
   scip = toCopy->scip;
   name = toCopy->name;
   nClasses = toCopy->nClasses;
   nIndices = toCopy->nIndices;
   indicesToClasses = toCopy->indicesToClasses;
   classNames.assign(nClasses, "");
   classDescriptions.assign(nClasses, "");
   classDecompInfo.assign(nClasses, 0);
   for ( int i = 0; i < nClasses; ++i )
   {
      classNames[i] = toCopy->classNames[i];
      classDescriptions[i] = toCopy->classDescriptions[i];
      classDecompInfo[i] = toCopy->classDecompInfo[i];
   }
}

/** destructor */
IndexPartition::~IndexPartition()
{
}

/** creates a new class, returns index of the class */
int IndexPartition::addClass(const char* givenName, const char* givenDesc)
{
   assert((int) classNames.size() == nClasses);
   assert((int) classDescriptions.size() == nClasses);
   assert((int) classDecompInfo.size() == nClasses);

   std::string givenClassName = std::string(givenName);

   classNames.push_back(givenClassName);
   classDescriptions.push_back(givenDesc);
   classDecompInfo.push_back(0);

   ++nClasses;

   return nClasses - 1;
}

/** assigns an index to a class */
void IndexPartition::assignIndexToClass(int givenIndex, int givenClassindex)
{
   assert(0 <= givenIndex && givenIndex < nIndices);
   assert(-1 <= givenClassindex && givenClassindex < nClasses);

   indicesToClasses[givenIndex] = givenClassindex;
}

/** returns true if the other partition has an equivalent index structure,
 *  meaning that the partition of the set of constraints is the same ignoring the concrete classindices, classnames, etc. */
bool IndexPartition::isDuplicateOf(IndexPartition* otherPartition )
{
   std::vector<int> classMapping ( getNClasses(), -1 );

   /* check whether number of indices and classes is the same */
   assert(getNIndices() == otherPartition->getNIndices() );
   if( getNClasses() != otherPartition->getNClasses() )
      return false;

   /* check whether index classes in this partition are subsets of classes in current partition */
   for( int i = 0; i < getNIndices(); ++i )
   {
      if( isIndexClassified(i) )
      {
         int compClass = getClassOfIndex(i);

         if( classMapping[compClass] == -1 )
            classMapping[compClass] = otherPartition->getClassOfIndex(i);
         else if( classMapping[compClass] != otherPartition->getClassOfIndex(i) )
            return false;
      }
      else if( otherPartition->isIndexClassified(i) )
      {
         return false;
      }
   }

   /* check whether index classes in this partition are strict subsets of classes in current partition */
   for( size_t c = 0; c < classMapping.size(); ++c )
   {
      if( classMapping[c] != -1 )
      {
         for( size_t j = c + 1; j < classMapping.size(); ++j )
         {
            if( classMapping[c] == classMapping[j] )
               return false;
         }
      }
   }

   return true;
}

/** returns a vector containing all possible subsets of the given classindices */
std::vector<std::vector<int>> IndexPartition::getAllSubsets(std::vector<int>& givenClassindices)
{
   std::vector<std::vector<int>> subsets;
   std::vector<int> empty;
   subsets.push_back(empty);

   for( int& givenClassindice : givenClassindices )
   {
      std::vector< std::vector<int> > subsetTemp = subsets;

      for( auto& j : subsetTemp )
         j.push_back(givenClassindice);

      for( auto& j : subsetTemp )
         subsets.push_back(j);
   }
   return subsets;
}

/** returns the decomposition info of the a class */
int IndexPartition::getClassDecompInfo(int givenClassindex)
{
   assert(0 <= givenClassindex && givenClassindex < nClasses);

   return classDecompInfo[givenClassindex];
}

/** returns the information text of a class */
const char* IndexPartition::getClassDescription(int givenClassindex)
{
   assert(0 <= givenClassindex && givenClassindex < nClasses);

   return classDescriptions[givenClassindex].c_str();
}

/** returns the name of a class */
const char* IndexPartition::getClassName(int givenClassindex)
{
   assert(0 <= givenClassindex && givenClassindex < nClasses);

   return classNames[givenClassindex].c_str();
}

/** returns the name of the class an index is assigned to */
const char* IndexPartition::getClassNameOfIndex(int givenIndex)
{
   assert(0 <= givenIndex && givenIndex < nIndices);
   assert(0 <= indicesToClasses[givenIndex] && indicesToClasses[givenIndex] < nClasses);

   return classNames[indicesToClasses[givenIndex]].c_str();
}


/** returns the index of the class an index is assigned to */
int IndexPartition::getClassOfIndex(int givenIndex)
{
   assert(0 <= givenIndex && givenIndex < nIndices);

   return indicesToClasses[givenIndex];
}

/** returns vector containing the assigned class of each index */
std::vector<int>& IndexPartition::getIndicesToClasses()
{
   return indicesToClasses;
}


/** returns the name of the partition */
const char* IndexPartition::getName()
{
   return name.c_str();
}


/** returns the number of classes the partition provides */
int IndexPartition::getNClasses()
{
   return nClasses;
}

/** returns the number of indices */
int IndexPartition::getNIndices()
{
   return nIndices;
}

/** returns a vector with the numbers of indices that are assigned to the classes */
std::vector<int> IndexPartition::getNIndicesOfClasses()
{
   std::vector<int> nIndicesOfClasses( nClasses, 0 );

   if ( nClasses == 0 )
      return nIndicesOfClasses;

   for ( int i = 0; i < nIndices; ++i)
   {
      if ( indicesToClasses[i] != -1 )
         ++nIndicesOfClasses[indicesToClasses[i]];
   }
   return nIndicesOfClasses;
}

/** returns whether an index is already assigned to a class */
bool IndexPartition::isIndexClassified(int givenIndex)
{
   assert(0 <= givenIndex && givenIndex < nIndices);

   return indicesToClasses[givenIndex] != -1;
}

/** returns a class index mapping for creating a new partition */
std::vector<int> IndexPartition::reduceClasses(int givenMaxNumber)
{
   assert( givenMaxNumber > 0 );

   if ( getNClasses() <= givenMaxNumber || nClasses >= 2*givenMaxNumber)
      return std::vector<int>(0);

   std::vector<int> classindexmapping(nClasses, 0);
   int enlargedclass = nClasses - givenMaxNumber;

   /* count number of indices per class */
   std::vector<std::pair<int,int>> nmembers( nClasses, std::pair<int,int>(0,0) );
   for( int i = 0; i < nClasses; ++i )
   {
      nmembers[i].first = i;
   }

   std::vector<int>::const_iterator iter = indicesToClasses.begin();
   std::vector<int>::const_iterator iterend = indicesToClasses.end();
   for( ; iter < iterend; ++iter )
   {
      if ( *iter != -1 )
         nmembers[*iter].second++;
   }

   /* map the classes with high numbers of assigned indices to new class indices */
   std::sort( nmembers.begin(), nmembers.end(), sort_pred());
   for( int i = 1; i < givenMaxNumber; ++i )
   {
      classindexmapping[nmembers[enlargedclass + i].first] = i;
   }

   return classindexmapping;
}

/** removes all classes which do not have any assigned indices (classindices may change)
 *  returns number of removed classes */
int IndexPartition::removeEmptyClasses()
{
   if ( nClasses == 0 )
      return 0;

   /* firstly, find empty classes */
   std::vector<int> toDelete(0);
   std::vector<int> nIndicesPerClasses(nClasses, 0);

   for ( int i = 0; i < nIndices; ++i )
   {
      if ( indicesToClasses[i] != -1 )
         ++nIndicesPerClasses[indicesToClasses[i]];
   }

   for ( int i = 0; i < nClasses; ++i )
   {
      if ( nIndicesPerClasses[i] == 0 )
      {
         toDelete.push_back(i);
      }
   }

   /* secondly, update data */
   for ( size_t i = 0; i < toDelete.size(); ++i )
   {
      int classindex = toDelete[toDelete.size() - 1 - i];

      for ( int j = 0; j < nIndices; ++j )
      {
         assert( indicesToClasses[j] != classindex );
         if ( indicesToClasses[j] > classindex )
            --indicesToClasses[j];
      }
      classNames.erase(classNames.begin() + classindex);
      classDescriptions.erase(classDescriptions.begin() + classindex);
      classDecompInfo.erase(classDecompInfo.begin() + classindex);
      --nClasses;
   }

   return toDelete.size();
}

/** sets the decomposition info of the a class */
void IndexPartition::setClassDecompInfo(int givenClassindex, int givenDecompInfo)
{
   assert(0 <= givenClassindex && givenClassindex < nClasses);

   classDecompInfo[givenClassindex] = givenDecompInfo;
}

/** sets the information text of a class */
void IndexPartition::setClassDescription(int givenClassindex, const char* givenDesc)
{
   assert(0 <= givenClassindex && givenClassindex < nClasses);

   classDescriptions[givenClassindex] = std::string(givenDesc);
}

/** sets the name of a class */
void IndexPartition::setClassName(int givenClassindex, const char* givenName)
{
   assert(0 <= givenClassindex && givenClassindex < nClasses);

   classNames[givenClassindex] = std::string(givenName);
}

} /* namespace gcg */

