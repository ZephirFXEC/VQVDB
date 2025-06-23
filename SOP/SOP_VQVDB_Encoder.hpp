//
// Created by zphrfx on 22/06/2025.
//

#pragma once

#include <PRM/PRM_TemplateBuilder.h>
#include <SOP/SOP_Node.h>
#include <SOP/SOP_NodeVerb.h>

#include "SOP_VQVDB_Encoder.proto.h"

class SOP_VQVDB_Encoder final : public SOP_Node {
   public:
	SOP_VQVDB_Encoder(OP_Network* net, const char* name, OP_Operator* op) : SOP_Node(net, name, op) {}

	~SOP_VQVDB_Encoder() override = default;

	static PRM_Template* buildTemplates();

	static OP_Node* myConstructor(OP_Network* net, const char* name, OP_Operator* op) { return new SOP_VQVDB_Encoder(net, name, op); }

	OP_ERROR cookMySop(OP_Context& context) override { return cookMyselfAsVerb(context); }

	const SOP_NodeVerb* cookVerb() const override;


	const char* inputLabel(unsigned idx) const override {
		switch (idx) {
			case 0:
				return "Input Grids";
			default:
				return "Sourcing Grids";
		}
	}
};

class SOP_VQVDB_EncoderCache final : public SOP_NodeCache {
   public:
	SOP_VQVDB_EncoderCache() : SOP_NodeCache() {}
	~SOP_VQVDB_EncoderCache() override = default;
};

class SOP_VQVDB_EncoderVerb final : public SOP_NodeVerb {
   public:
	SOP_VQVDB_EncoderVerb() = default;
	~SOP_VQVDB_EncoderVerb() override = default;
	[[nodiscard]] SOP_NodeParms* allocParms() const override { return new SOP_VQVDB_EncoderParms; }
	[[nodiscard]] SOP_NodeCache* allocCache() const override { return new SOP_VQVDB_EncoderCache(); }
	[[nodiscard]] UT_StringHolder name() const override { return "VQVDB_Encoder"; }

	SOP_NodeVerb::CookMode cookMode(const SOP_NodeParms* parms) const override { return SOP_NodeVerb::COOK_DUPLICATE; }

	void cook(const SOP_NodeVerb::CookParms& cookparms) const override;

	static const SOP_NodeVerb::Register<SOP_VQVDB_EncoderVerb> theVerb;
	static const char* const theDsFile;
};
