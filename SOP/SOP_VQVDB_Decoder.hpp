//
// Created by zphrfx on 22/06/2025.
//

#pragma once

#include <PRM/PRM_TemplateBuilder.h>
#include <SOP/SOP_Node.h>
#include <SOP/SOP_NodeVerb.h>

#include "SOP_VQVDB_Decoder.proto.h"

class SOP_VQVDB_Decoder final : public SOP_Node {
public:
	SOP_VQVDB_Decoder(OP_Network* net, const char* name, OP_Operator* op) : SOP_Node(net, name, op) {}

	~SOP_VQVDB_Decoder() override = default;

	static PRM_Template* buildTemplates();

	static OP_Node* myConstructor(OP_Network* net, const char* name, OP_Operator* op) { return new SOP_VQVDB_Decoder(net, name, op); }

	OP_ERROR cookMySop(OP_Context& context) override { return cookMyselfAsVerb(context); }

	const SOP_NodeVerb* cookVerb() const override;


	const char* inputLabel(unsigned idx) const override {
		switch (idx) {
			case 0:
				return "Input Grids";
			default:
				return "Input Grids";
		}
	}
};

class SOP_VQVDB_DecoderCache final : public SOP_NodeCache {
public:
	SOP_VQVDB_DecoderCache() : SOP_NodeCache() {}
	~SOP_VQVDB_DecoderCache() override = default;
};

class SOP_VQVDB_DecoderVerb final : public SOP_NodeVerb {
public:
	SOP_VQVDB_DecoderVerb() = default;
	~SOP_VQVDB_DecoderVerb() override = default;
	[[nodiscard]] SOP_NodeParms* allocParms() const override { return new SOP_VQVDB_DecoderParms; }
	[[nodiscard]] SOP_NodeCache* allocCache() const override { return new SOP_VQVDB_DecoderCache(); }
	[[nodiscard]] UT_StringHolder name() const override { return "VQVDB_Decoder"; }

	SOP_NodeVerb::CookMode cookMode(const SOP_NodeParms* parms) const override { return SOP_NodeVerb::COOK_GENERATOR; }

	void cook(const SOP_NodeVerb::CookParms& cookparms) const override;

	static const SOP_NodeVerb::Register<SOP_VQVDB_DecoderVerb> theVerb;
	static const char* const theDsFile;
};
