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

/**@file   xternal.c
 * @brief  documentation page for GCG's C-API (no other pages)
 */

/**@defgroup PUBLICAPI Public API of GCG
 * @brief methods and headers of the public C-API of \GCG
 *
 * The public API of \GCG is separated into a Core API and a Plugin API.
 * The first contains all methods that can be accessed by including the header gcg.h.
 * The Plugin API is a collection of methods that are provided by the default plugins of \GCG.
 * The Plugin API is provided by gcgplugins.c."
 *
 *
 */

/**@defgroup PUBLICCOREAPI Core API
* @ingroup PUBLICAPI
* @brief methods and headers of the plugin-independent C-API provided by \GCG.
*
* In order facilitate the navigation through the core API of \GCG, it is structured into different modules.
*/

/**@defgroup DATASTRUCTURES Data Structures
  * @ingroup PUBLICCOREAPI
  * @brief Commonly used data structures
  */

/**@defgroup TYPEDEFINITIONS Type Definitions
  * @ingroup PUBLICCOREAPI
  * @brief Type definitions and callback declarations
  */


/**@defgroup BLISS Bliss
  * @ingroup PUBLICCOREAPI
  * @brief Methods concerning BLISS
  */

/**@defgroup DECOMP Decomposition
  * @ingroup PUBLICCOREAPI
  * @brief Public methods concerning the decomposition.
  *
  */

/**@defgroup HEURISTICS Heuristics
  * @ingroup PUBLICCOREAPI
  * @brief Public methods concerning heuristics.
  */

/**@defgroup PRICING_PUB Pricing
  * @ingroup PUBLICCOREAPI
  * @brief All pricing-related public functionalities.
  *
  */

/**@defgroup PRICINGJOB Pricing Job
  * @ingroup PRICING_PUB
  */

/**@defgroup PRICINGPROB Pricing Problem
  * @ingroup PRICING_PUB
  */

/**@defgroup SEPARATORS_PUB Separators
  * @ingroup PUBLICCOREAPI
  * @brief Public methods for separators.
  */

/**@defgroup MISC Miscellaneous
  * @ingroup PUBLICCOREAPI
  * @brief Public methods from the scip_misc.c file.
  */




/**@defgroup PUBLICPLUGINAPI Plugin API of GCG
  * @ingroup PUBLICAPI
  * @brief core API extensions provided by the default plugins of \GCG.
  *
  * All of the modules listed below provide functions that are allowed to be used by user-written extensions of \GCG.
  */

  /**@defgroup BENDERS Benders' Decomposition
   * @ingroup PUBLICPLUGINAPI
   * @brief This page contains a description of all methods and files provided by the Benders' decomposition.
   *
   */

  /**@defgroup BRANCHINGRULES Branching Rules
   * @ingroup PUBLICPLUGINAPI
   * @brief This page contains a list of all branching rule which are currently available.
   *
   * A detailed description what a branching rule does and how to add a branching rule to \GCG can be found
   * \ref own-branching-rule "here".
   */

  /**@defgroup CONSHDLRS  Constraint Handler
   * @ingroup PUBLICPLUGINAPI
   * @brief This page contains a list of all constraint handlers which are currently available.
   *
   * A detailed description what a constraint handler does and how to add a constraint handler to \GCG can be found
   * in the SCIP documentation.
   */

  /**@defgroup DETECTORS Detectors
   * @ingroup PUBLICPLUGINAPI
   * @brief This page contains a list of all detectors which are currently available.
   *
   * A detailed description what a detector does and how to add a detector to \GCG can be found
   * \ref detection "here".
   */

  /**@defgroup CLASSIFIERS Classifiers
   * @ingroup PUBLICPLUGINAPI
   * @brief This page contains a list of all classifiers which are currently available.
   *
   * A detailed description what a classifier does can be found \ref classifiers "here"
   * and a guide on how to add a classifier to \GCG can be found \ref own-classifier "here".
   *
   */

  /**@defgroup DIALOGS Dialogs
   * @ingroup PUBLICPLUGINAPI
   * @brief This page contains a list of all dialogs which are currently available.
   *
   * A detailed description what a dialog does and how to add a dialog to \GCG can be found
   * n the SCIP documentation.
   */

  /**@defgroup DISPLAYS Displays
   * @ingroup PUBLICPLUGINAPI
   * @brief This page contains a list of all displays (output columns)  which are currently available.
   *
   * A detailed description what a display does and how to add a display to \GCG can be found
   * in the SCIP documentation.
   *
   */

  /**@defgroup FILEREADERS File Readers
   * @ingroup PUBLICPLUGINAPI
   * @brief This page contains a list of all file readers which are currently available.
   *
   * A detailed description what a file reader does and how to add a file reader to \GCG can be found
   * in the SCIP documentation.
   */

  /**@defgroup NODESELECTORS Node Selectors
   * @ingroup PUBLICPLUGINAPI
   * @brief This page contains a list of all node selectors which are currently available.
   *
   * A detailed description what a node selector does and how to add a node selector to \GCG can be found
   * in the SCIP documentation.
   */

   /**@defgroup PRICING Pricing
    * @ingroup PUBLICPLUGINAPI
    * @brief This page contains a list of all pricers, pricing solvers and the pricing jobs and problem structures.
    *
    */

  /**@defgroup PRICERS Pricers
   * @ingroup PRICING
   * @brief This page contains a list of all pricers which are currently available.
   *
   * Per default there exist no variable pricer. A detailed description what a variable pricer does and how to add a
   * variable pricer to \GCG can be found in the SCIP documentation.
   */

  /**@defgroup PRICINGSOLVERS Pricing Solvers
   * @ingroup PRICING
   * @brief This page contains a list of all pricing solvers which are currently available.
   *
   * A detailed description what a pricing solver does and how to add a pricing solver to \GCG can be found
   * \ref pricing "here".
   */

   /**@defgroup PRIMALHEURISTICS Primal Heuristics
   * @ingroup PUBLICPLUGINAPI
   * @brief This page contains a list of all primal heuristics which are currently available.
   *
   * A detailed description what a primal heuristic does and how to add a primal heuristic to \GCG can be found
   * \ref own-primal-heuristic "here".
   */

   /**@defgroup DIVINGHEURISTICS Diving Heuristics
   * @ingroup PRIMALHEURISTICS
   * @brief This page contains a list of all diving heuristics which are currently available.
   *
   * A detailed description what a diving heuristic does can be found
   * \ref diving-heuristics "here".
   */

  /**@defgroup RELAXATORS Relaxators
   * @ingroup PUBLICPLUGINAPI
   * @brief This page contains a list of all relaxators which are currently available.
   */

  /**@defgroup SEPARATORS Separators
   * @ingroup PUBLICPLUGINAPI
   * @brief This page contains a list of all separators  which are currently available.
   *
   * A detailed description what a separator does and how to add a separator to \GCG can be found
   * in the SCIP documentation.
   */

  /**@defgroup TYPEDEFINITIONS Type Definitions
   * @ingroup PUBLICCOREAPI
   * This page lists headers containing branch-and-price specific public methods provided by \GCG.
   *
   * All of the headers listed below include functions that are allowed to be called by external users. Besides those
   * functions it is also valid to call methods that are listed in one of the headers of the (default) \GCG plug-ins; in
   * particular, this holds for relax_gcg.h and pricer_gcg.h.
   *
   */

  /**\@defgroup INTERNALAPI Internal API of \GCG
   * \@brief internal API methods that should only be used by the core of \GCG
   *
   * This page lists the header files of internal API methods. In contrast to the public API, these internal methods
   * should not be used by user plugins and extensions of \GCG. Please consult
   * \ref PUBLICCOREAPI "the Core API" and \ref PUBLICPLUGINAPI "Plugin API" for the complete API available to user plugins.
   *
   */
