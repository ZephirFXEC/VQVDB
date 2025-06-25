#include "SOP_VQVDB_Decoder.hpp"

#include <GU/GU_Detail.h>
#include <UT/UT_DSOVersion.h>

#include "Utils/Utils.hpp"
#include "VQVDB/VQVAECodec.hpp"

void newSopOperator(OP_OperatorTable* table) {
	table->addOperator(new OP_Operator("vqvdb_decoder", "VQVDB Decoder", SOP_VQVDB_Decoder::myConstructor,
	                                   SOP_VQVDB_Decoder::buildTemplates(), 0, 0, nullptr, OP_FLAG_GENERATOR));
}


const char* const SOP_VQVDB_DecoderVerb::theDsFile = R"THEDSFILE(
{
    name        "SOP_VQVDB_Decoder"
    label       "VQ-VDB Decoder"

    parm {
        name    "vdbname"
        label   "VDB Grid Name"
        type    string
        default { "density" }
    }
    parm {
        name    "inputfile"
        label   "Input File (.vqvdb)"
        type    file
    }
    parm {
        name    "batchsize"
        label   "GPU Batch Size"
        type    integer
        default { 64 }
        range   { 1 8192 }
    }
}
)THEDSFILE";


PRM_Template* SOP_VQVDB_Decoder::buildTemplates() {
	static PRM_TemplateBuilder templ("SOP_VQVDB_Decoder.cpp", SOP_VQVDB_DecoderVerb::theDsFile);
	return templ.templates();
}
const SOP_NodeVerb::Register<SOP_VQVDB_DecoderVerb> SOP_VQVDB_DecoderVerb::theVerb;

const SOP_NodeVerb* SOP_VQVDB_Decoder::cookVerb() const { return SOP_VQVDB_DecoderVerb::theVerb.get(); }

void SOP_VQVDB_DecoderVerb::cook(const CookParms& cookparms) const {
	auto& sopparms = cookparms.parms<SOP_VQVDB_DecoderParms>();
	GU_Detail* gdp = cookparms.gdh().gdpNC();


	const auto& in_path = sopparms.getInputfile();

	if (in_path.empty()) {
		return;
	}
	openvdb::FloatGrid::Ptr output_grid = openvdb::FloatGrid::create();

	try {
		// --- Run Decoder ---
		cookparms.sopAddMessage(SOP_MESSAGE, "Starting VQ-VDB decoding...");

		VQVAECodec decoder("C:/Users/zphrfx/Desktop/hdk/VQVDB/models/vqvae_scripted.pt");

		decoder.decompress(in_path.data(), output_grid, sopparms.getBatchsize());
	} catch (const std::exception& e) {
		cookparms.sopAddError(SOP_MESSAGE, e.what());
	}

	GU_PrimVDB::buildFromGrid(*gdp, output_grid, nullptr, sopparms.getVdbname());
}
