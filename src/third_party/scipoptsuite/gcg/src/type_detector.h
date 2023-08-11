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

/**@file   type_detector.h
 * @ingroup TYPEDEFINITIONS
 * @brief  type definitions for detectors in GCG projects
 * @author Martin Bergner
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_TYPE_DETECTOR_H__
#define GCG_TYPE_DETECTOR_H__

#include "scip/type_retcode.h"
#include "scip/type_scip.h"
#include "scip/type_result.h"
#include "type_decomp.h"


typedef struct DEC_Detector DEC_DETECTOR;
typedef struct DEC_DetectorData DEC_DETECTORDATA;


struct Partialdec_Detection_Data;
typedef struct Partialdec_Detection_Data PARTIALDEC_DETECTION_DATA;


/** destructor of detector to free user data (called when GCG is exiting)
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - detector        : detector data structure
 */
#define DEC_DECL_FREEDETECTOR(x) SCIP_RETCODE x (SCIP* scip, DEC_DETECTOR* detector)

/**
 * detector initialization method (called after problem was transformed)
 * It can be used to fill the detector data with needed information. The implementation is optional.
 *
 * input:
 *  - scip            : SCIP data structure
 *  - detector        : detector data structure
 */
#define DEC_DECL_INITDETECTOR(x) SCIP_RETCODE x (SCIP* scip, DEC_DETECTOR* detector)

/**
 * detector deinitialization method (called before the transformed problem is freed)
 * It can be used to clean up the data created in DEC_DECL_INITDETECTOR. The implementation is optional.
 *
 * input:
 *  - scip            : SCIP data structure
 *  - detector        : detector data structure
 */
#define DEC_DECL_EXITDETECTOR(x) SCIP_RETCODE x (SCIP* scip, DEC_DETECTOR* detector)


/**
 * given a partialdec (incomplete decomposition) the detector
 * tries to find refined partialdec and stores it
 *
 * input:
 *  - scip                 : SCIP data structure
 *  - detector             : pointer to detector
 *  - partialdecdetectiondata   : pointer to partialdec propagation data structure
 *  - result               : pointer where to store the result
 *
 * possible return values for result:
 *  - SCIP_SUCCESS    : the method completed and found decompositions
 *  - SCIP_DIDNOTFIND : the method completed without finding a decomposition
 *  - SCIP_DIDNOTRUN  : the method did not run
 */
#define DEC_DECL_PROPAGATEPARTIALDEC(x) SCIP_RETCODE x (SCIP* scip, DEC_DETECTOR* detector, PARTIALDEC_DETECTION_DATA* partialdecdetectiondata, SCIP_RESULT* result)


/**
 * given a partialdec (incomplete decomposition) the detector
 * tries to find finished partialdecs and stores them
 *
 * input:
 *  - scip                  : SCIP data structure
 *  - detector              : pointer to detector
 *  - partialdecdetectiondata    : pointer to partialdec propagation data structure
 *  - result                : pointer where to store the result
 *
 * possible return values for result:
 *  - SCIP_SUCCESS    : the method completed and found decompositions
 *  - SCIP_DIDNOTFIND : the method completed without finding a decomposition
 *  - SCIP_DIDNOTRUN  : the method did not run
 */
#define DEC_DECL_FINISHPARTIALDEC(x) SCIP_RETCODE x (SCIP* scip, DEC_DETECTOR* detector, PARTIALDEC_DETECTION_DATA* partialdecdetectiondata, SCIP_RESULT* result)

/**
 * given a complete partialdec (complete decomposition) the detector
 * postprocess the partialdec in order to find a different yet promising partialdec
 *
 * input:
 *  - scip                  : SCIP data structure
 *  - detector              : pointer to detector
 *  - partialdecdetectiondata    : pointer to partialdec propagation data structure
 *  - result                : pointer where to store the result
 *
 * possible return values for result:
 *  - SCIP_SUCCESS    : the method completed and found decompositions
 *  - SCIP_DIDNOTFIND : the method completed without finding a decomposition
 *  - SCIP_DIDNOTRUN  : the method did not run
 */
#define DEC_DECL_POSTPROCESSPARTIALDEC(x) SCIP_RETCODE x (SCIP* scip, DEC_DETECTOR* detector, PARTIALDEC_DETECTION_DATA* partialdecdetectiondata, SCIP_RESULT* result)

/**
 * set the parameter of a detector to values according to fast emphasis and size of the instance
 *  input:
 *  - scip            : SCIP data structure
 *  - detector        : pointer to detector
 *  - result          : pointer where to store the result
 *
 * possible return values for result:
 *  - SCIP_SUCCESS    : the method completed
 *  - SCIP_DIDNOTRUN  : the method did not run
 */
#define DEC_DECL_SETPARAMFAST(x) SCIP_RETCODE x (SCIP* scip, DEC_DETECTOR* detector, SCIP_RESULT* result)

/**
 * set the parameter of a detector to values according to aggressive emphasis and size of the instance
 *  input:
 *  - scip            : SCIP data structure
 *  - detector        : pointer to detector
 *  - result          : pointer where to store the result
 *
 * possible return values for result:
 *  - SCIP_SUCCESS    : the method completed
 *  - SCIP_DIDNOTRUN  : the method did not run
 */
#define DEC_DECL_SETPARAMAGGRESSIVE(x) SCIP_RETCODE x (SCIP* scip, DEC_DETECTOR* detector, SCIP_RESULT* result)

/**
 * set the parameter of a detector to values according to default emphasis and size of the instance
 *  input:
 *  - scip            : SCIP data structure
 *  - detector        : pointer to detector
 *  - result          : pointer where to store the result
 *
 * possible return values for result:
 *  - SCIP_SUCCESS    : the method completed
 *  - SCIP_DIDNOTRUN  : the method did not run
 */
#define DEC_DECL_SETPARAMDEFAULT(x) SCIP_RETCODE x (SCIP* scip, DEC_DETECTOR* detector, SCIP_RESULT* result)



#endif
