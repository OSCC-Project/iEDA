/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*               This file is part of the program and library                */
/*    PaPILO --- Parallel Presolve for Integer and Linear Optimization       */
/*                                                                           */
/* Copyright (C) 2020-2022 Konrad-Zuse-Zentrum                               */
/*                     fuer Informationstechnik Berlin                       */
/*                                                                           */
/* This program is free software: you can redistribute it and/or modify      */
/* it under the terms of the GNU Lesser General Public License as published  */
/* by the Free Software Foundation, either version 3 of the License, or      */
/* (at your option) any later version.                                       */
/*                                                                           */
/* This program is distributed in the hope that it will be useful,           */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/* GNU Lesser General Public License for more details.                       */
/*                                                                           */
/* You should have received a copy of the GNU Lesser General Public License  */
/* along with this program.  If not, see <https://www.gnu.org/licenses/>.    */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef _PAPILO_CORE_MATRIX_BUFFER_HPP_
#define _PAPILO_CORE_MATRIX_BUFFER_HPP_

#include "papilo/core/SparseStorage.hpp"
#include "papilo/misc/Vec.hpp"
#include <cassert>
#include <cstdint>

namespace papilo
{

// for doubles, entry is 32bytes which makes two entries per cache line
// the entries are added to a vector and linked to two trees. One
// allows traversal for row major and one for column major storage order
template <typename REAL>
struct MatrixEntry
{
   REAL val;
   int row;
   int col;

   struct TreeHook
   {
      int left;
      int right;
   };

   TreeHook row_major;
   TreeHook col_major;

   MatrixEntry() {}

   MatrixEntry( int _row, int _col, const REAL& _val )
       : val( _val ), row( _row ), col( _col )
   {
      row_major.left = 0;
      row_major.right = 0;
      col_major.left = 0;
      col_major.right = 0;
   }
};

template <bool RowMajor>
struct GetNodeProperty;

/// data structure for sparse matrix entries that allows efficient row
/// major and column major traversal
template <typename REAL>
struct MatrixBuffer
{
   template <bool RowMajor>
   int
   splay( int n, int t )
   {
      using Node = GetNodeProperty<RowMajor>;

      /* Simple top down splay, not requiring i to be in the tree t.  */
      /* What it does is described above.                             */

      assert( t != 0 );

      int l = 0;
      int r = 0;
      int y;

      for( ;; )
      {
         if( Node::lesser( entries[n], entries[t] ) )
         {
            int t_left = Node::left( entries[t] );
            if( t_left == 0 )
               break;
            if( Node::lesser( entries[n], entries[t_left] ) )
            {
               // rotate right
               y = t_left;
               Node::left( entries[t] ) = Node::right( entries[y] );
               Node::right( entries[y] ) = t;
               t = y;
               if( Node::left( entries[t] ) == 0 )
                  break;
            }

            // link right
            Node::left( entries[r] ) = t;

            r = t;
            t = Node::left( entries[t] );
         }
         else
         {
            int t_right = Node::right( entries[t] );
            if( t_right == 0 )
               break;
            if( Node::lesser( entries[t_right], entries[n] ) )
            {
               // rotate left
               y = t_right;
               Node::right( entries[t] ) = Node::left( entries[y] );
               Node::left( entries[y] ) = t;
               t = y;
               if( Node::right( entries[t] ) == 0 )
                  break;
            }

            // link left
            Node::right( entries[l] ) = t;

            l = t;
            t = Node::right( entries[t] );
         }
      }

      // assemble
      Node::right( entries[l] ) = Node::left( entries[t] );
      Node::left( entries[r] ) = Node::right( entries[t] );
      Node::left( entries[t] ) = Node::right( entries[0] );
      Node::right( entries[t] ) = Node::left( entries[0] );

      Node::left( entries[0] ) = 0;
      Node::right( entries[0] ) = 0;

      return t;
   }

   template <bool RowMajor>
   void
   splay_insert( int n, int t )
   {
      using Node = GetNodeProperty<RowMajor>;

      t = splay<RowMajor>( n, t );
      // make n the new root
      if( Node::lesser( entries[n], entries[t] ) )
      {
         Node::left( entries[n] ) = Node::left( entries[t] );
         Node::right( entries[n] ) = t;
         Node::left( entries[t] ) = 0;
      }
      else
      {
         Node::right( entries[n] ) = Node::right( entries[t] );
         Node::left( entries[n] ) = t;
         Node::right( entries[t] ) = 0;
      }
   }

   /// link node n into the tree for the given storage order
   template <bool RowMajor>
   void
   link( int n )
   {
      using Node = GetNodeProperty<RowMajor>;

      // currnode is the pointer of the parent node
      // where n will be inserted, starting with the root pointer
      int* currnode = &Node::root( this );

      // get priority of n
      uint32_t prio = Node::priority( entries[n] );

      // after this loop n will be the subtree that needs
      // to be linked to *currnode. That is either a
      // leaf node or a treap where n is the root
      while( *currnode != 0 )
      {
         uint32_t p = Node::priority( entries[*currnode] );

         if( p < prio )
         {
            // if we found a node with a smaller priority we do
            // a top down splay to make n the root of this sub tree
            // and stop insertion here.
            int t = splay<RowMajor>( n, *currnode );

            // make n the new root
            if( Node::lesser( entries[n], entries[t] ) )
            {
               Node::left( entries[n] ) = Node::left( entries[t] );
               Node::right( entries[n] ) = t;
               Node::left( entries[t] ) = 0;
            }
            else
            {
               Node::right( entries[n] ) = Node::right( entries[t] );
               Node::left( entries[n] ) = t;
               Node::right( entries[t] ) = 0;
            }

            break;
         }

         // go further down the tree
         if( Node::lesser( entries[n], entries[*currnode] ) )
            currnode = &Node::left( entries[*currnode] );
         else
            currnode = &Node::right( entries[*currnode] );
      }

      // update the parents pointer to the the tree now rooted at n
      // (could be the root pointer)
      *currnode = n;
   }

   bool
   empty() const
   {
      return entries.size() == 1;
   }

   void
   clear()
   {
      entries.resize( 1 );
      col_major_root = 0;
      row_major_root = 0;
   }

   void
   addEntry( int row, int col, const REAL& val )
   {
      int n = entries.size();
      entries.emplace_back( row, col, val );

      this->template link<true>( n );
      this->template link<false>( n );
   }

   template <bool RowMajor>
   MatrixEntry<REAL>*
   findEntry( int row, int col )
   {
      using Node = GetNodeProperty<RowMajor>;
      MatrixEntry<REAL> entry( row, col, REAL{ 0 } );

      int k = Node::root( this );

      while( k != 0 )
      {
         if( Node::lesser( entry, entries[k] ) )
            k = Node::left( entries[k] );
         else if( Node::lesser( entries[k], entry ) )
            k = Node::right( entries[k] );
         else
         {
            assert( entries[k].row == row );
            assert( entries[k].col == col );
            return &entries[k];
         }
      }

      return nullptr;
   }

   void
   addEntrySafe( int row, int col, const REAL& val )
   {
      MatrixEntry<REAL>* entry = findEntry<true>( row, col );

      if( entry != NULL )
         entry->val += val;
      else
         addEntry( row, col, val );
   }

   template <bool RowMajor>
   const MatrixEntry<REAL>*
   begin( SmallVec<int, 32>& stack ) const
   {
      using Node = GetNodeProperty<RowMajor>;

      stack.clear();
      stack.push_back( 0 );

      int k = Node::root( this );

      while( k != 0 )
      {
         stack.push_back( k );
         k = Node::left( entries[k] );
      }

      return &entries[stack.back()];
   }

   template <bool RowMajor>
   const MatrixEntry<REAL>*
   beginStart( SmallVec<int, 32>& stack, int row, int col ) const
   {
      using Node = GetNodeProperty<RowMajor>;

      stack.clear();
      stack.push_back( 0 );

      int k = Node::root( this );

      assert( row == -1 || col == -1 );

      MatrixEntry<REAL> dummy( row, col, REAL{ 0 } );

      while( k != 0 )
      {
         if( Node::lesser( dummy, entries[k] ) )
         {
            stack.push_back( k );
            k = Node::left( entries[k] );
         }
         else
         {
            k = Node::right( entries[k] );
         }
      }

      return &entries[stack.back()];
   }

   template <bool RowMajor>
   const MatrixEntry<REAL>*
   next( SmallVec<int, 32>& stack ) const
   {
      using Node = GetNodeProperty<RowMajor>;

      int k = stack.back();
      stack.pop_back();

      k = Node::right( entries[k] );
      while( k != 0 )
      {
         stack.push_back( k );
         k = Node::left( entries[k] );
      }

      return &entries[stack.back()];
   }

   const MatrixEntry<REAL>*
   end() const
   {
      return &entries[0];
   }

   void
   reserve( int nnz )
   {
      entries.reserve( nnz + 1 );
   }

   void
   startBadge()
   {
      badge_start = entries.size();
   }

   void
   addBadgeEntry( int row, int col, const REAL& val )
   {
      assert( badge_start >= 0 && badge_start <= (int)entries.size() );
      entries.emplace_back( row, col, val );
   }

   void
   discardBadge()
   {
      assert( badge_start >= 0 && badge_start <= entries.size() );
      entries.resize( badge_start );
      badge_start = -1;
   }

   void
   finishBadge()
   {
      assert( badge_start >= 0 && badge_start <= (int)entries.size() );

      for( int i = badge_start; i != (int)entries.size(); ++i )
      {
         this->template link<true>( i );
         this->template link<false>( i );
      }

      badge_start = -1;
   }

   int
   getNnz() const
   {
      return entries.size() - 1;
   }

   SparseStorage<REAL>
   buildCSR(
       int nrows, int ncols,
       double spareRatio = SparseStorage<REAL>::DEFAULT_SPARE_RATIO,
       int mininterrowspace = SparseStorage<REAL>::DEFAULT_MIN_INTER_ROW_SPACE )
   {
      int nnz = getNnz();

      SparseStorage<REAL> csrStorage( nrows, ncols, nnz, spareRatio,
                                      mininterrowspace );

      SmallVec<int, 32> stack;

      REAL* values = csrStorage.getValues();
      int* columns = csrStorage.getColumns();
      IndexRange* rowranges = csrStorage.getRowRanges();

      const MatrixEntry<REAL>* it = this->begin<true>( stack );
      const MatrixEntry<REAL>* end = this->end();

      int k = 0;

      for( int i = 0; i != nrows; ++i )
      {
         rowranges[i].start = k;

         while( it != end && it->row == i )
         {
            values[k] = it->val;
            columns[k] = it->col;

            ++k;

            it = this->next<true>( stack );
         }

         rowranges[i].end = k;

         if( k != rowranges[i].start )
         {
            int rowsize = k - rowranges[i].start;
            k += csrStorage.computeRowAlloc( rowsize ) - rowsize;
         }
      }

      rowranges[nrows].start = csrStorage.getNAlloc();
      rowranges[nrows].end = csrStorage.getNAlloc();

      return csrStorage;
   }

   SparseStorage<REAL>
   buildCSC(
       int nrows, int ncols,
       double spareRatio = SparseStorage<REAL>::DEFAULT_SPARE_RATIO,
       int minintercolspace = SparseStorage<REAL>::DEFAULT_MIN_INTER_ROW_SPACE )
   {
      int nnz = getNnz();

      SparseStorage<REAL> cscStorage( ncols, nrows, nnz, spareRatio,
                                      minintercolspace );

      SmallVec<int, 32> stack;

      REAL* values = cscStorage.getValues();
      int* rows = cscStorage.getColumns();
      IndexRange* colranges = cscStorage.getRowRanges();

      const MatrixEntry<REAL>* it = this->begin<false>( stack );
      const MatrixEntry<REAL>* end = this->end();

      int k = 0;

      for( int i = 0; i != ncols; ++i )
      {
         colranges[i].start = k;

         while( it != end && it->col == i )
         {
            values[k] = it->val;
            rows[k] = it->row;

            ++k;

            it = this->next<false>( stack );
         }

         colranges[i].end = k;

         if( k != colranges[i].start )
         {
            int colsize = k - colranges[i].start;
            k += cscStorage.computeRowAlloc( colsize ) - colsize;
         }
      }

      colranges[ncols].start = cscStorage.getNAlloc();
      colranges[ncols].end = cscStorage.getNAlloc();

      return cscStorage;
   }

   MatrixBuffer()
   {
      // root is the dummy NULL node
      row_major_root = 0;
      col_major_root = 0;

      // insert dummy NULL node at position 0
      entries.emplace_back( -1, -1, 0 );
   }

   int badge_start = -1;
   int row_major_root;
   int col_major_root;
   Vec<MatrixEntry<REAL>> entries;
};

template <>
struct GetNodeProperty<true>
{
   template <typename REAL>
   static uint32_t
   priority( const MatrixEntry<REAL>& e )
   {
      uint64_t x = uint64_t( uint32_t( e.row ) ) << 32 | uint32_t( e.col );
      return uint32_t( ( x * UINT64_C( 0x9e3779b97f4a7c15 ) ) >> 32 );
   }

   template <typename REAL>
   static bool
   lesser( const MatrixEntry<REAL>& a, const MatrixEntry<REAL>& b )
   {
      return a.row < b.row || ( a.row == b.row && a.col < b.col );
   }

   template <typename REAL>
   static int&
   left( MatrixEntry<REAL>& e )
   {
      return e.row_major.left;
   }

   template <typename REAL>
   static int&
   right( MatrixEntry<REAL>& e )
   {
      return e.row_major.right;
   }

   template <typename REAL>
   static int&
   root( MatrixBuffer<REAL>* thisptr )
   {
      return thisptr->row_major_root;
   }

   template <typename REAL>
   static const int&
   left( const MatrixEntry<REAL>& e )
   {
      return e.row_major.left;
   }

   template <typename REAL>
   static const int&
   right( const MatrixEntry<REAL>& e )
   {
      return e.row_major.right;
   }

   template <typename REAL>
   static const int&
   root( const MatrixBuffer<REAL>* thisptr )
   {
      return thisptr->row_major_root;
   }
};

template <>
struct GetNodeProperty<false>
{

   template <typename REAL>
   static uint32_t
   priority( const MatrixEntry<REAL>& e )
   {
      uint64_t x = uint64_t( uint32_t( e.col ) ) << 32 | uint32_t( e.row );
      return uint32_t( ( x * UINT64_C( 0x9e3779b97f4a7c15 ) ) >> 32 );
   }

   template <typename REAL>
   static bool
   lesser( const MatrixEntry<REAL>& a, const MatrixEntry<REAL>& b )
   {
      return a.col < b.col || ( a.col == b.col && a.row < b.row );
   }

   template <typename REAL>
   static int&
   left( MatrixEntry<REAL>& e )
   {
      return e.col_major.left;
   }

   template <typename REAL>
   static int&
   right( MatrixEntry<REAL>& e )
   {
      return e.col_major.right;
   }

   template <typename REAL>
   static int&
   root( MatrixBuffer<REAL>* thisptr )
   {
      return thisptr->col_major_root;
   }

   template <typename REAL>
   static const int&
   left( const MatrixEntry<REAL>& e )
   {
      return e.col_major.left;
   }

   template <typename REAL>
   static const int&
   right( const MatrixEntry<REAL>& e )
   {
      return e.col_major.right;
   }

   template <typename REAL>
   static const int&
   root( const MatrixBuffer<REAL>* thisptr )
   {
      return thisptr->col_major_root;
   }
};

} // namespace papilo

#endif
