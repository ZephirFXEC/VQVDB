/*
 * Copyright (c) 2025, Enzo Crema
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See the LICENSE file in the project root for full license text.
 */
#include <OP/OP_OperatorTable.h>
#include <UT/UT_DSOVersion.h>

#include "SOP_VQVDB_Decoder.hpp"
#include "SOP_VQVDB_Encoder.hpp"

// This is the single entry point Houdini will call for this DSO
void newSopOperator(OP_OperatorTable* table) {
	newSopOperator_Encoder(table);
	newSopOperator_Decoder(table);
}