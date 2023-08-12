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

/**@file   gcgsorttpl.c
 * @brief  template functions for sorting
 * @author Michael Winkler
 * @author Tobias Achterberg
 * @author Tobias Oelschlaegel
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

/* template parameters that have to be passed in as #define's:
 * #define GCGSORTTPL_NAMEEXT      <ext>      extension to be used for SCIP method names, for example DownIntRealPtr
 * #define GCGSORTTPL_KEYTYPE      <type>     data type of the key array
 * #define GCGSORTTPL_FIELD1TYPE   <type>     data type of first additional array which should be sorted in the same way (optional)
 * #define GCGSORTTPL_FIELD2TYPE   <type>     data type of second additional array which should be sorted in the same way (optional)
 * #define GCGSORTTPL_FIELD3TYPE   <type>     data type of third additional array which should be sorted in the same way (optional)
 * #define GCGSORTTPL_FIELD4TYPE   <type>     data type of fourth additional array which should be sorted in the same way (optional)
 * #define GCGSORTTPL_FIELD5TYPE   <type>     data type of fifth additional array which should be sorted in the same way (optional)
 * #define GCGSORTTPL_FIELD6TYPE   <type>     data type of fifth additional array which should be sorted in the same way (optional)
 * #define GCGSORTTPL_PTRCOMP                 ptrcomp method should be used for comparisons (optional)
 * #define GCGSORTTPL_INDCOMP                 indcomp method should be used for comparisons (optional)
 * #define GCGSORTTPL_BACKWARDS               should the array be sorted other way around
 */

/** compares two element indices
 **  result:
 **    < 0: ind1 comes before (is better than) ind2
 **    = 0: both indices have the same value
 **    > 0: ind2 comes after (is worse than) ind2
 **/
#define GCG_DECL_SORTINDCOMP(x) int x (void* userdata, void* dataptr, int ind1, int ind2)

/** compares two data element pointers
 **  result:
 **    < 0: elem1 comes before (is better than) elem2
 **    = 0: both elements have the same value
 **    > 0: elem2 comes after (is worse than) elem2
 **/
#define GCG_DECL_SORTPTRCOMP(x) int x (void* userdata, void* elem1, void* elem2)


#define GCGSORTTPL_SHELLSORTMAX 25

#ifndef GCGSORTTPL_NAMEEXT
#error You need to define GCGSORTTPL_NAMEEXT.
#endif
#ifndef GCGSORTTPL_KEYTYPE
#error You need to define GCGSORTTPL_KEYTYPE.
#endif

#ifdef GCGSORTTPL_EXPANDNAME
#undef GCGSORTTPL_EXPANDNAME
#endif
#ifdef GCGSORTTPL_NAME
#undef GCGSORTTPL_NAME
#endif

/* enabling and disabling additional lines in the code */
#ifdef GCGSORTTPL_FIELD1TYPE
#define GCGSORTTPL_HASFIELD1(x)    x
#define GCGSORTTPL_HASFIELD1PAR(x) x,
#else
#define GCGSORTTPL_HASFIELD1(x)    /**/
#define GCGSORTTPL_HASFIELD1PAR(x) /**/
#endif
#ifdef GCGSORTTPL_FIELD2TYPE
#define GCGSORTTPL_HASFIELD2(x)    x
#define GCGSORTTPL_HASFIELD2PAR(x) x,
#else
#define GCGSORTTPL_HASFIELD2(x)    /**/
#define GCGSORTTPL_HASFIELD2PAR(x) /**/
#endif
#ifdef GCGSORTTPL_FIELD3TYPE
#define GCGSORTTPL_HASFIELD3(x)    x
#define GCGSORTTPL_HASFIELD3PAR(x) x,
#else
#define GCGSORTTPL_HASFIELD3(x)    /**/
#define GCGSORTTPL_HASFIELD3PAR(x) /**/
#endif
#ifdef GCGSORTTPL_FIELD4TYPE
#define GCGSORTTPL_HASFIELD4(x)    x
#define GCGSORTTPL_HASFIELD4PAR(x) x,
#else
#define GCGSORTTPL_HASFIELD4(x)    /**/
#define GCGSORTTPL_HASFIELD4PAR(x) /**/
#endif
#ifdef GCGSORTTPL_FIELD5TYPE
#define GCGSORTTPL_HASFIELD5(x)    x
#define GCGSORTTPL_HASFIELD5PAR(x) x,
#else
#define GCGSORTTPL_HASFIELD5(x)    /**/
#define GCGSORTTPL_HASFIELD5PAR(x) /**/
#endif
#ifdef GCGSORTTPL_FIELD6TYPE
#define GCGSORTTPL_HASFIELD6(x)    x
#define GCGSORTTPL_HASFIELD6PAR(x) x,
#else
#define GCGSORTTPL_HASFIELD6(x)    /**/
#define GCGSORTTPL_HASFIELD6PAR(x) /**/
#endif
#ifdef GCGSORTTPL_PTRCOMP
#define GCGSORTTPL_HASPTRCOMP(x)    x
#define GCGSORTTPL_HASPTRCOMPPAR(x) x,
#else
#define GCGSORTTPL_HASPTRCOMP(x)    /**/
#define GCGSORTTPL_HASPTRCOMPPAR(x) /**/
#endif
#ifdef GCGSORTTPL_INDCOMP
#define GCGSORTTPL_HASINDCOMP(x)    x
#define GCGSORTTPL_HASINDCOMPPAR(x) x,
#else
#define GCGSORTTPL_HASINDCOMP(x)    /**/
#define GCGSORTTPL_HASINDCOMPPAR(x) /**/
#endif


/* the two-step macro definition is needed, such that macro arguments
 * get expanded by prescan of the C preprocessor (see "info cpp",
 * chapter 3.10.6: Argument Prescan)
 */
#define GCGSORTTPL_EXPANDNAME(method, methodname) \
   method ## methodname
#define GCGSORTTPL_NAME(method, methodname) \
  GCGSORTTPL_EXPANDNAME(method, methodname)

/* comparator method */
#ifdef GCGSORTTPL_PTRCOMP
#ifdef GCGSORTTPL_BACKWARDS
#define GCGSORTTPL_ISBETTER(x,y) (ptrcomp(userdata, (x), (y)) > 0)
#define GCGSORTTPL_ISWORSE(x,y) (ptrcomp(userdata, (x), (y)) < 0)
#else
#define GCGSORTTPL_ISBETTER(x,y) (ptrcomp(userdata, (x), (y)) < 0)
#define GCGSORTTPL_ISWORSE(x,y) (ptrcomp(userdata, (x), (y)) > 0)
#endif
#else
#ifdef GCGSORTTPL_INDCOMP
#ifdef GCGSORTTPL_BACKWARDS
#define GCGSORTTPL_ISBETTER(x,y) (indcomp(userdata, dataptr, (x), (y)) > 0)
#define GCGSORTTPL_ISWORSE(x,y) (indcomp(userdata, dataptr, (x), (y)) < 0)
#else
#define GCGSORTTPL_ISBETTER(x,y) (indcomp(userdata, dataptr, (x), (y)) < 0)
#define GCGSORTTPL_ISWORSE(x,y) (indcomp(userdata, dataptr, (x), (y)) > 0)
#endif
#else
#ifdef GCGSORTTPL_BACKWARDS
#define GCGSORTTPL_ISBETTER(x,y) ((x) > (y))
#define GCGSORTTPL_ISWORSE(x,y) ((x) < (y))
#else
#define GCGSORTTPL_ISBETTER(x,y) ((x) < (y))
#define GCGSORTTPL_ISWORSE(x,y) ((x) > (y))
#endif
#endif
#endif

/* swapping two variables */
#define GCGSORTTPL_SWAP(T,x,y) \
   {                \
      T temp = x;   \
      x = y;        \
      y = temp;     \
   }


/** shell-sort an array of data elements; use it only for arrays smaller than 25 entries */
static
void GCGSORTTPL_NAME(gcgsorttpl_shellSort, GCGSORTTPL_NAMEEXT)
(
   GCGSORTTPL_KEYTYPE*      key,                /**< pointer to data array that defines the order */
   GCGSORTTPL_HASFIELD1PAR(  GCGSORTTPL_FIELD1TYPE*    field1 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD2PAR(  GCGSORTTPL_FIELD2TYPE*    field2 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD3PAR(  GCGSORTTPL_FIELD3TYPE*    field3 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD4PAR(  GCGSORTTPL_FIELD4TYPE*    field4 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD5PAR(  GCGSORTTPL_FIELD5TYPE*    field5 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD6PAR(  GCGSORTTPL_FIELD6TYPE*    field6 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASPTRCOMPPAR( GCG_DECL_SORTPTRCOMP((*ptrcomp)) )  /**< data element comparator */
   GCGSORTTPL_HASINDCOMPPAR( GCG_DECL_SORTINDCOMP((*indcomp)) )  /**< data element comparator */
   GCGSORTTPL_HASINDCOMPPAR( void*                  dataptr    )  /**< pointer to data field that is given to the external compare method */
   void*                 userdata,           /**< userdata that is supplied to the comparator function */
   int                   start,              /**< starting index */
   int                   end                 /**< ending index */
   )
{
   static const int incs[3] = {1, 5, 19}; /* sequence of increments */
   int k;

   assert(start <= end);

   for( k = 2; k >= 0; --k )
   {
      int h = incs[k];
      int first = h + start;
      int i;

      for( i = first; i <= end; ++i )
      {
         int j;
         GCGSORTTPL_KEYTYPE tempkey = key[i];

         GCGSORTTPL_HASFIELD1( GCGSORTTPL_FIELD1TYPE tempfield1 = field1[i]; )
         GCGSORTTPL_HASFIELD2( GCGSORTTPL_FIELD2TYPE tempfield2 = field2[i]; )
         GCGSORTTPL_HASFIELD3( GCGSORTTPL_FIELD3TYPE tempfield3 = field3[i]; )
         GCGSORTTPL_HASFIELD4( GCGSORTTPL_FIELD4TYPE tempfield4 = field4[i]; )
         GCGSORTTPL_HASFIELD5( GCGSORTTPL_FIELD5TYPE tempfield5 = field5[i]; )
         GCGSORTTPL_HASFIELD6( GCGSORTTPL_FIELD6TYPE tempfield6 = field6[i]; )

         j = i;
         while( j >= first && GCGSORTTPL_ISBETTER(tempkey, key[j-h]) )
         {
            key[j] = key[j-h];
            GCGSORTTPL_HASFIELD1( field1[j] = field1[j-h]; )
            GCGSORTTPL_HASFIELD2( field2[j] = field2[j-h]; )
            GCGSORTTPL_HASFIELD3( field3[j] = field3[j-h]; )
            GCGSORTTPL_HASFIELD4( field4[j] = field4[j-h]; )
            GCGSORTTPL_HASFIELD5( field5[j] = field5[j-h]; )
            GCGSORTTPL_HASFIELD6( field6[j] = field6[j-h]; )
            j -= h;
         }

         key[j] = tempkey;
         GCGSORTTPL_HASFIELD1( field1[j] = tempfield1; )
         GCGSORTTPL_HASFIELD2( field2[j] = tempfield2; )
         GCGSORTTPL_HASFIELD3( field3[j] = tempfield3; )
         GCGSORTTPL_HASFIELD4( field4[j] = tempfield4; )
         GCGSORTTPL_HASFIELD5( field5[j] = tempfield5; )
         GCGSORTTPL_HASFIELD6( field6[j] = tempfield6; )
      }
   }
}


/** quick-sort an array of pointers; pivot is the medial element */
static
void GCGSORTTPL_NAME(gcgsorttpl_qSort, GCGSORTTPL_NAMEEXT)
(
   GCGSORTTPL_KEYTYPE*      key,                /**< pointer to data array that defines the order */
   GCGSORTTPL_HASFIELD1PAR(  GCGSORTTPL_FIELD1TYPE*    field1 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD2PAR(  GCGSORTTPL_FIELD2TYPE*    field2 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD3PAR(  GCGSORTTPL_FIELD3TYPE*    field3 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD4PAR(  GCGSORTTPL_FIELD4TYPE*    field4 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD5PAR(  GCGSORTTPL_FIELD5TYPE*    field5 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD6PAR(  GCGSORTTPL_FIELD6TYPE*    field6 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASPTRCOMPPAR( GCG_DECL_SORTPTRCOMP((*ptrcomp)) )  /**< data element comparator */
   GCGSORTTPL_HASINDCOMPPAR( GCG_DECL_SORTINDCOMP((*indcomp)) )  /**< data element comparator */
   GCGSORTTPL_HASINDCOMPPAR( void*                  dataptr    )  /**< pointer to data field that is given to the external compare method */
   void*                 userdata,           /**< userdata that is supplied to the comparator function */
   int                   start,              /**< starting index */
   int                   end,                /**< ending index */
   SCIP_Bool             type                /**< TRUE, if quick-sort should start with with key[lo] < pivot <= key[hi], key[lo] <= pivot < key[hi] otherwise */
   )
{
   assert(start <= end);

   /* use quick-sort for long lists */
   while( end - start >= GCGSORTTPL_SHELLSORTMAX )
   {
      GCGSORTTPL_KEYTYPE pivotkey;
      int lo;
      int hi;
      int mid;

      /* select pivot element */
      mid = (start+end)/2;
      pivotkey = key[mid];

      /* partition the array into elements < pivot [start,hi] and elements >= pivot [lo,end] */
      lo = start;
      hi = end;
      for( ;; )
      {
         if( type )
         {
            while( lo < end && GCGSORTTPL_ISBETTER(key[lo], pivotkey) )
               lo++;
            while( hi > start && !GCGSORTTPL_ISBETTER(key[hi], pivotkey) )
               hi--;
         }
         else
         {
            while( lo < end && !GCGSORTTPL_ISWORSE(key[lo], pivotkey) )
               lo++;
            while( hi > start && GCGSORTTPL_ISWORSE(key[hi], pivotkey) )
               hi--;
         }

         if( lo >= hi )
            break;

         GCGSORTTPL_SWAP(GCGSORTTPL_KEYTYPE, key[lo], key[hi]);
         GCGSORTTPL_HASFIELD1( GCGSORTTPL_SWAP(GCGSORTTPL_FIELD1TYPE, field1[lo], field1[hi]); )
         GCGSORTTPL_HASFIELD2( GCGSORTTPL_SWAP(GCGSORTTPL_FIELD2TYPE, field2[lo], field2[hi]); )
         GCGSORTTPL_HASFIELD3( GCGSORTTPL_SWAP(GCGSORTTPL_FIELD3TYPE, field3[lo], field3[hi]); )
         GCGSORTTPL_HASFIELD4( GCGSORTTPL_SWAP(GCGSORTTPL_FIELD4TYPE, field4[lo], field4[hi]); )
         GCGSORTTPL_HASFIELD5( GCGSORTTPL_SWAP(GCGSORTTPL_FIELD5TYPE, field5[lo], field5[hi]); )
         GCGSORTTPL_HASFIELD6( GCGSORTTPL_SWAP(GCGSORTTPL_FIELD6TYPE, field6[lo], field6[hi]); )

         lo++;
         hi--;
      }
      assert((hi == lo-1) || (type && hi == start) || (!type && lo == end));

      /* skip entries which are equal to the pivot element (three partitions, <, =, > than pivot)*/
      if( type )
      {
         while( lo < end && !GCGSORTTPL_ISBETTER(pivotkey, key[lo]) )
            lo++;

         /* make sure that we have at least one element in the smaller partition */
         if( lo == start )
         {
            /* everything is greater or equal than the pivot element: move pivot to the left (degenerate case) */
            assert(!GCGSORTTPL_ISBETTER(key[mid], pivotkey)); /* the pivot element did not change its position */
            assert(!GCGSORTTPL_ISBETTER(pivotkey, key[mid]));
            GCGSORTTPL_SWAP(GCGSORTTPL_KEYTYPE, key[lo], key[mid]);
            GCGSORTTPL_HASFIELD1( GCGSORTTPL_SWAP(GCGSORTTPL_FIELD1TYPE, field1[lo], field1[mid]); )
            GCGSORTTPL_HASFIELD2( GCGSORTTPL_SWAP(GCGSORTTPL_FIELD2TYPE, field2[lo], field2[mid]); )
            GCGSORTTPL_HASFIELD3( GCGSORTTPL_SWAP(GCGSORTTPL_FIELD3TYPE, field3[lo], field3[mid]); )
            GCGSORTTPL_HASFIELD4( GCGSORTTPL_SWAP(GCGSORTTPL_FIELD4TYPE, field4[lo], field4[mid]); )
            GCGSORTTPL_HASFIELD5( GCGSORTTPL_SWAP(GCGSORTTPL_FIELD5TYPE, field5[lo], field5[mid]); )
            GCGSORTTPL_HASFIELD6( GCGSORTTPL_SWAP(GCGSORTTPL_FIELD6TYPE, field6[lo], field6[mid]); )
            lo++;
         }
      }
      else
      {
         while( hi > start && !GCGSORTTPL_ISWORSE(pivotkey, key[hi]) )
            hi--;

         /* make sure that we have at least one element in the smaller partition */
         if( hi == end )
         {
            /* everything is greater or equal than the pivot element: move pivot to the left (degenerate case) */
            assert(!GCGSORTTPL_ISBETTER(key[mid], pivotkey)); /* the pivot element did not change its position */
            assert(!GCGSORTTPL_ISBETTER(pivotkey, key[mid]));
            GCGSORTTPL_SWAP(GCGSORTTPL_KEYTYPE, key[hi], key[mid]);
            GCGSORTTPL_HASFIELD1( GCGSORTTPL_SWAP(GCGSORTTPL_FIELD1TYPE, field1[hi], field1[mid]); )
            GCGSORTTPL_HASFIELD2( GCGSORTTPL_SWAP(GCGSORTTPL_FIELD2TYPE, field2[hi], field2[mid]); )
            GCGSORTTPL_HASFIELD3( GCGSORTTPL_SWAP(GCGSORTTPL_FIELD3TYPE, field3[hi], field3[mid]); )
            GCGSORTTPL_HASFIELD4( GCGSORTTPL_SWAP(GCGSORTTPL_FIELD4TYPE, field4[hi], field4[mid]); )
            GCGSORTTPL_HASFIELD5( GCGSORTTPL_SWAP(GCGSORTTPL_FIELD5TYPE, field5[hi], field5[mid]); )
            GCGSORTTPL_HASFIELD6( GCGSORTTPL_SWAP(GCGSORTTPL_FIELD6TYPE, field6[hi], field6[mid]); )
            hi--;
         }
      }

      /* sort the smaller partition by a recursive call, sort the larger part without recursion */
      if( hi - start <= end - lo )
      {
         /* sort [start,hi] with a recursive call */
         if( start < hi )
         {
            GCGSORTTPL_NAME(gcgsorttpl_qSort, GCGSORTTPL_NAMEEXT)
               (key,
                GCGSORTTPL_HASFIELD1PAR(field1)
                GCGSORTTPL_HASFIELD2PAR(field2)
                GCGSORTTPL_HASFIELD3PAR(field3)
                GCGSORTTPL_HASFIELD4PAR(field4)
                GCGSORTTPL_HASFIELD5PAR(field5)
                GCGSORTTPL_HASFIELD6PAR(field6)
                GCGSORTTPL_HASPTRCOMPPAR(ptrcomp)
                GCGSORTTPL_HASINDCOMPPAR(indcomp)
                GCGSORTTPL_HASINDCOMPPAR(dataptr)
                  userdata, start, hi, !type);
         }

         /* now focus on the larger part [lo,end] */
         start = lo;
      }
      else
      {
         if( lo < end )
         {
            /* sort [lo,end] with a recursive call */
            GCGSORTTPL_NAME(gcgsorttpl_qSort, GCGSORTTPL_NAMEEXT)
               (key,
                GCGSORTTPL_HASFIELD1PAR(field1)
                GCGSORTTPL_HASFIELD2PAR(field2)
                GCGSORTTPL_HASFIELD3PAR(field3)
                GCGSORTTPL_HASFIELD4PAR(field4)
                GCGSORTTPL_HASFIELD5PAR(field5)
                GCGSORTTPL_HASFIELD6PAR(field6)
                GCGSORTTPL_HASPTRCOMPPAR(ptrcomp)
                GCGSORTTPL_HASINDCOMPPAR(indcomp)
                GCGSORTTPL_HASINDCOMPPAR(dataptr)
                  userdata, lo, end, !type);
         }

         /* now focus on the larger part [start,hi] */
         end = hi;
      }
      type = !type;
   }

   /* use shell sort on the remaining small list */
   if( end - start >= 1 )
   {
      GCGSORTTPL_NAME(gcgsorttpl_shellSort, GCGSORTTPL_NAMEEXT)
         (key,
            GCGSORTTPL_HASFIELD1PAR(field1)
            GCGSORTTPL_HASFIELD2PAR(field2)
            GCGSORTTPL_HASFIELD3PAR(field3)
            GCGSORTTPL_HASFIELD4PAR(field4)
            GCGSORTTPL_HASFIELD5PAR(field5)
            GCGSORTTPL_HASFIELD6PAR(field6)
            GCGSORTTPL_HASPTRCOMPPAR(ptrcomp)
            GCGSORTTPL_HASINDCOMPPAR(indcomp)
            GCGSORTTPL_HASINDCOMPPAR(dataptr)
            userdata, start, end);
   }
}

#ifndef NDEBUG
/** verifies that an array is indeed sorted */
static
void GCGSORTTPL_NAME(gcgsorttpl_checkSort, GCGSORTTPL_NAMEEXT)
(
   GCGSORTTPL_KEYTYPE*      key,                /**< pointer to data array that defines the order */
   GCGSORTTPL_HASPTRCOMPPAR( GCG_DECL_SORTPTRCOMP((*ptrcomp)) )  /**< data element comparator */
   GCGSORTTPL_HASINDCOMPPAR( GCG_DECL_SORTINDCOMP((*indcomp)) )  /**< data element comparator */
   GCGSORTTPL_HASINDCOMPPAR( void*                  dataptr    )  /**< pointer to data field that is given to the external compare method */
   void*                 userdata,           /**< userdata that is supplied to the comparator function */
   int                   len                 /**< length of the array */
   )
{
   int i;

   for( i = 0; i < len-1; i++ )
   {
      assert(!GCGSORTTPL_ISBETTER(key[i+1], key[i]));
   }
}
#endif

/** SCIPsort...(): sorts array 'key' and performs the same permutations on the additional 'field' arrays */
void GCGSORTTPL_NAME(GCGsort, GCGSORTTPL_NAMEEXT)
(
   GCGSORTTPL_KEYTYPE*      key,                /**< pointer to data array that defines the order */
   GCGSORTTPL_HASFIELD1PAR(  GCGSORTTPL_FIELD1TYPE*    field1 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD2PAR(  GCGSORTTPL_FIELD2TYPE*    field2 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD3PAR(  GCGSORTTPL_FIELD3TYPE*    field3 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD4PAR(  GCGSORTTPL_FIELD4TYPE*    field4 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD5PAR(  GCGSORTTPL_FIELD5TYPE*    field5 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD6PAR(  GCGSORTTPL_FIELD6TYPE*    field6 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASPTRCOMPPAR( GCG_DECL_SORTPTRCOMP((*ptrcomp)) )  /**< data element comparator */
   GCGSORTTPL_HASINDCOMPPAR( GCG_DECL_SORTINDCOMP((*indcomp)) )  /**< data element comparator */
   GCGSORTTPL_HASINDCOMPPAR( void*                  dataptr    )  /**< pointer to data field that is given to the external compare method */
   void*                 userdata,           /** userdata that is supplied to the comparator function */
   int                   len                 /**< length of arrays */
   )
{
   /* ignore the trivial cases */
   if( len <= 1 )
      return;

   /* use shell sort on the remaining small list */
   if( len <= GCGSORTTPL_SHELLSORTMAX )
   {
      GCGSORTTPL_NAME(gcgsorttpl_shellSort, GCGSORTTPL_NAMEEXT)
         (key,
            GCGSORTTPL_HASFIELD1PAR(field1)
            GCGSORTTPL_HASFIELD2PAR(field2)
            GCGSORTTPL_HASFIELD3PAR(field3)
            GCGSORTTPL_HASFIELD4PAR(field4)
            GCGSORTTPL_HASFIELD5PAR(field5)
            GCGSORTTPL_HASFIELD6PAR(field6)
            GCGSORTTPL_HASPTRCOMPPAR(ptrcomp)
            GCGSORTTPL_HASINDCOMPPAR(indcomp)
            GCGSORTTPL_HASINDCOMPPAR(dataptr)
            userdata, 0, len-1);
   }
   else
   {
      GCGSORTTPL_NAME(gcgsorttpl_qSort, GCGSORTTPL_NAMEEXT)
         (key,
            GCGSORTTPL_HASFIELD1PAR(field1)
            GCGSORTTPL_HASFIELD2PAR(field2)
            GCGSORTTPL_HASFIELD3PAR(field3)
            GCGSORTTPL_HASFIELD4PAR(field4)
            GCGSORTTPL_HASFIELD5PAR(field5)
            GCGSORTTPL_HASFIELD6PAR(field6)
            GCGSORTTPL_HASPTRCOMPPAR(ptrcomp)
            GCGSORTTPL_HASINDCOMPPAR(indcomp)
            GCGSORTTPL_HASINDCOMPPAR(dataptr)
            userdata, 0, len-1, TRUE);
   }
#ifndef NDEBUG
   GCGSORTTPL_NAME(gcgsorttpl_checkSort, GCGSORTTPL_NAMEEXT)
      (key,
       GCGSORTTPL_HASPTRCOMPPAR(ptrcomp)
       GCGSORTTPL_HASINDCOMPPAR(indcomp)
       GCGSORTTPL_HASINDCOMPPAR(dataptr)
       userdata,
       len);
#endif
}

#ifdef GCGSORTTPL_SORTED_VEC
/** SCIPsortedvecInsert...(): adds an element to a sorted multi-vector;
 *  This method does not do any memory allocation! It assumes that the arrays are large enough
 *  to store the additional values.
 */
void GCGSORTTPL_NAME(GCGsortedvecInsert, GCGSORTTPL_NAMEEXT)
(
   GCGSORTTPL_KEYTYPE*      key,                /**< pointer to data array that defines the order */
   GCGSORTTPL_HASFIELD1PAR(  GCGSORTTPL_FIELD1TYPE*    field1 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD2PAR(  GCGSORTTPL_FIELD2TYPE*    field2 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD3PAR(  GCGSORTTPL_FIELD3TYPE*    field3 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD4PAR(  GCGSORTTPL_FIELD4TYPE*    field4 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD5PAR(  GCGSORTTPL_FIELD5TYPE*    field5 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD6PAR(  GCGSORTTPL_FIELD6TYPE*    field6 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASPTRCOMPPAR( GCG_DECL_SORTPTRCOMP((*ptrcomp)) )  /**< data element comparator */
   GCGSORTTPL_HASINDCOMPPAR( GCG_DECL_SORTINDCOMP((*indcomp)) )  /**< data element comparator */
   GCGSORTTPL_HASINDCOMPPAR( void*                  dataptr    )  /**< pointer to data field that is given to the external compare method */
   GCGSORTTPL_KEYTYPE       keyval,             /**< key value of new element */
   GCGSORTTPL_HASFIELD1PAR(  GCGSORTTPL_FIELD1TYPE     field1val  )  /**< field1 value of new element */
   GCGSORTTPL_HASFIELD2PAR(  GCGSORTTPL_FIELD2TYPE     field2val  )  /**< field1 value of new element */
   GCGSORTTPL_HASFIELD3PAR(  GCGSORTTPL_FIELD3TYPE     field3val  )  /**< field1 value of new element */
   GCGSORTTPL_HASFIELD4PAR(  GCGSORTTPL_FIELD4TYPE     field4val  )  /**< field1 value of new element */
   GCGSORTTPL_HASFIELD5PAR(  GCGSORTTPL_FIELD5TYPE     field5val  )  /**< field1 value of new element */
   GCGSORTTPL_HASFIELD6PAR(  GCGSORTTPL_FIELD6TYPE     field6val  )  /**< field1 value of new element */
   void*                 userdata,           /**< userdata that is supplied to the comparator function */
   int*                  len,                /**< pointer to length of arrays (will be increased by 1) */
   int*                  pos                 /**< pointer to store the insert position, or NULL */
   )
{
   int j;

   for( j = *len; j > 0 && GCGSORTTPL_ISBETTER(keyval, key[j-1]); j-- )
   {
      key[j] = key[j-1];
      GCGSORTTPL_HASFIELD1( field1[j] = field1[j-1]; )
      GCGSORTTPL_HASFIELD2( field2[j] = field2[j-1]; )
      GCGSORTTPL_HASFIELD3( field3[j] = field3[j-1]; )
      GCGSORTTPL_HASFIELD4( field4[j] = field4[j-1]; )
      GCGSORTTPL_HASFIELD5( field5[j] = field5[j-1]; )
      GCGSORTTPL_HASFIELD6( field6[j] = field6[j-1]; )
   }

   key[j] = keyval;
   GCGSORTTPL_HASFIELD1( field1[j] = field1val; )
   GCGSORTTPL_HASFIELD2( field2[j] = field2val; )
   GCGSORTTPL_HASFIELD3( field3[j] = field3val; )
   GCGSORTTPL_HASFIELD4( field4[j] = field4val; )
   GCGSORTTPL_HASFIELD5( field5[j] = field5val; )
   GCGSORTTPL_HASFIELD6( field6[j] = field6val; )

   (*len)++;

   if( pos != NULL )
      (*pos) = j;
}

/** SCIPsortedvecDelPos...(): deletes an element at a given position from a sorted multi-vector */
void GCGSORTTPL_NAME(GCGsortedvecDelPos, GCGSORTTPL_NAMEEXT)
(
   GCGSORTTPL_KEYTYPE*      key,                /**< pointer to data array that defines the order */
   GCGSORTTPL_HASFIELD1PAR(  GCGSORTTPL_FIELD1TYPE*    field1 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD2PAR(  GCGSORTTPL_FIELD2TYPE*    field2 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD3PAR(  GCGSORTTPL_FIELD3TYPE*    field3 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD4PAR(  GCGSORTTPL_FIELD4TYPE*    field4 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD5PAR(  GCGSORTTPL_FIELD5TYPE*    field5 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASFIELD6PAR(  GCGSORTTPL_FIELD6TYPE*    field6 )      /**< additional field that should be sorted in the same way */
   GCGSORTTPL_HASPTRCOMPPAR( GCG_DECL_SORTPTRCOMP((*ptrcomp)) )  /**< data element comparator */
   GCGSORTTPL_HASINDCOMPPAR( GCG_DECL_SORTINDCOMP((*indcomp)) )  /**< data element comparator */
   GCGSORTTPL_HASINDCOMPPAR( void*                  dataptr    )  /**< pointer to data field that is given to the external compare method */
   void*                 userdata,           /**< userdata that is supplied to the comparator function */
   int                   pos,                /**< array position of element to be deleted */
   int*                  len                 /**< pointer to length of arrays (will be decreased by 1) */
   )
{
   int j;

   assert(0 <= pos && pos < *len);

   (*len)--;

   for( j = pos; j < *len; j++ )
   {
      key[j] = key[j+1];
      GCGSORTTPL_HASFIELD1( field1[j] = field1[j+1]; )
      GCGSORTTPL_HASFIELD2( field2[j] = field2[j+1]; )
      GCGSORTTPL_HASFIELD3( field3[j] = field3[j+1]; )
      GCGSORTTPL_HASFIELD4( field4[j] = field4[j+1]; )
      GCGSORTTPL_HASFIELD5( field5[j] = field5[j+1]; )
      GCGSORTTPL_HASFIELD6( field6[j] = field6[j+1]; )
   }
}


/* The SCIPsortedvecFind...() method only has needs the key array but not the other field arrays. In order to
 * avoid defining the same method multiple times, only include this method if we do not have any additional fields.
 */
#ifndef GCGSORTTPL_FIELD1TYPE

/** SCIPsortedvecFind...(): Finds the position at which 'val' is located in the sorted vector by binary search.
 *  If the element exists, the method returns TRUE and stores the position of the element in '*pos'.
 *  If the element does not exist, the method returns FALSE and stores the position of the element that follows
 *  'val' in the ordering in '*pos', i.e., '*pos' is the position at which 'val' would be inserted.
 *  Note that if the element is not found, '*pos' may be equal to len if all existing elements are smaller than 'val'.
 */
SCIP_Bool GCGSORTTPL_NAME(GCGsortedvecFind, GCGSORTTPL_NAMEEXT)
(
   GCGSORTTPL_KEYTYPE*      key,                /**< pointer to data array that defines the order */
   GCGSORTTPL_HASPTRCOMPPAR( GCG_DECL_SORTPTRCOMP((*ptrcomp)) )  /**< data element comparator */
   GCGSORTTPL_HASINDCOMPPAR( GCG_DECL_SORTINDCOMP((*indcomp)) )  /**< data element comparator */
   GCGSORTTPL_HASINDCOMPPAR( void*                  dataptr    )  /**< pointer to data field that is given to the external compare method */
   GCGSORTTPL_KEYTYPE       val,                /**< data field to find position for */
   void*                 userdata,           /**< userdata that is supplied to the comparator function */
   int                   len,                /**< length of array */
   int*                  pos                 /**< pointer to store the insert position */
   )
{
   int left;
   int right;

   assert(key != NULL);
   assert(pos != NULL);

   left = 0;
   right = len-1;
   while( left <= right )
   {
      int middle;

      middle = (left+right)/2;
      assert(0 <= middle && middle < len);

      if( GCGSORTTPL_ISBETTER(val, key[middle]) )
         right = middle-1;
      else if( GCGSORTTPL_ISBETTER(key[middle], val) )
         left = middle+1;
      else
      {
         *pos = middle;
         return TRUE;
      }
   }
   assert(left == right+1);

   *pos = left;
   return FALSE;
}

#endif
#endif /* GCGSORTTPL_SORTED_VEC */

/* undefine template parameters and local defines */
#undef GCGSORTTPL_NAMEEXT
#undef GCGSORTTPL_KEYTYPE
#undef GCGSORTTPL_FIELD1TYPE
#undef GCGSORTTPL_FIELD2TYPE
#undef GCGSORTTPL_FIELD3TYPE
#undef GCGSORTTPL_FIELD4TYPE
#undef GCGSORTTPL_FIELD5TYPE
#undef GCGSORTTPL_FIELD6TYPE
#undef GCGSORTTPL_PTRCOMP
#undef GCGSORTTPL_INDCOMP
#undef GCGSORTTPL_HASFIELD1
#undef GCGSORTTPL_HASFIELD2
#undef GCGSORTTPL_HASFIELD3
#undef GCGSORTTPL_HASFIELD4
#undef GCGSORTTPL_HASFIELD5
#undef GCGSORTTPL_HASFIELD6
#undef GCGSORTTPL_HASPTRCOMP
#undef GCGSORTTPL_HASINDCOMP
#undef GCGSORTTPL_HASFIELD1PAR
#undef GCGSORTTPL_HASFIELD2PAR
#undef GCGSORTTPL_HASFIELD3PAR
#undef GCGSORTTPL_HASFIELD4PAR
#undef GCGSORTTPL_HASFIELD5PAR
#undef GCGSORTTPL_HASFIELD6PAR
#undef GCGSORTTPL_HASPTRCOMPPAR
#undef GCGSORTTPL_HASINDCOMPPAR
#undef GCGSORTTPL_ISBETTER
#undef GCGSORTTPL_ISWORSE
#undef GCGSORTTPL_SWAP
#undef GCGSORTTPL_SHELLSORTMAX
#undef GCGSORTTPL_BACKWARDS
#undef GCGSORTTPL_SORTED_VEC
