#include <UT/UT_DSOVersion.h>

#include "SOP_VQVDB_Encoder.hpp"

#include <GU/GU_Detail.h>

#include "Utils/Utils.hpp"
#include "VQVDB_CPP/VQVAE_Encoder.hpp"

void newSopOperator(OP_OperatorTable* table) {
	table->addOperator(new OP_Operator("vqvdb_encoder", "VQVDB Encoder", SOP_VQVDB_Encoder::myConstructor,
	                                   SOP_VQVDB_Encoder::buildTemplates(), 1, 1, nullptr));
}


const char* const SOP_VQVDB_EncoderVerb::theDsFile = R"THEDSFILE(
{
    name        "SOP_VQVDB_Encoder"
    label       "VQ-VDB Encoder"

    parm {
        name    "vdbname"
        label   "VDB Grid Name"
        type    string
        default { "density" }
    }
    parm {
        name    "outputpath"
        label   "Output File (.vqvdb)"
        type    file
    }
    parm {
        name    "batchsize"
        label   "GPU Batch Size"
        type    integer
        default { 64 }
        range   { 1 1024 }
    }
    parm {
        name    "execute"
        label   "Encode and Save to Disk"
        type    toggle
    }
}
)THEDSFILE";


PRM_Template* SOP_VQVDB_Encoder::buildTemplates() {
	static PRM_TemplateBuilder templ("SOP_VQVDB_Encoder.cpp", SOP_VQVDB_EncoderVerb::theDsFile);
	return templ.templates();
}
const SOP_NodeVerb::Register<SOP_VQVDB_EncoderVerb> SOP_VQVDB_EncoderVerb::theVerb;

const SOP_NodeVerb* SOP_VQVDB_Encoder::cookVerb() const { return SOP_VQVDB_EncoderVerb::theVerb.get(); }

void SOP_VQVDB_EncoderVerb::cook(const CookParms& cookparms) const {
	auto& sopparms = cookparms.parms<SOP_VQVDB_EncoderParms>();
	const GU_Detail* input_gdp = cookparms.inputGeo(0);

	if (sopparms.getExecute() == 0) {
		return;
	}

	try {
		// --- Get Parameters ---
		const std::string out_path{sopparms.getOutputpath()};
		const int batch_size = sopparms.getBatchsize();

		// --- Validate Parameters ---
		if (out_path.empty()) {
			cookparms.sopAddError(SOP_MESSAGE, "Model path and/or Output path must be specified.");
		}
		if (!input_gdp) {
			cookparms.sopAddError(SOP_MESSAGE, "No input geometry connected.");
		}

		// --- Load Grid ---
		openvdb::FloatGrid::Ptr grid;
		if (auto err = loadGrid<openvdb::FloatGrid>(input_gdp, grid, sopparms.getVdbname()); err != UT_ERROR_NONE) {
			cookparms.sopAddError(SOP_MESSAGE, "Failed to load VDB grid from input.");
		}
		if (!grid) {
			cookparms.sopAddError(SOP_MESSAGE, "VDB grid is null or not found.");
		}

		// --- Run Encoder ---
		cookparms.sopAddMessage(SOP_MESSAGE, "Starting VQ-VDB encoding...");

		VQVAEEncoder encoder("C:/Users/zphrfx/Desktop/hdk/VQVDB/models/vqvae_scripted.pt");

		encoder.compress(grid, out_path);

		cookparms.sopAddMessage(SOP_MESSAGE, ("Successfully saved to " + out_path).c_str());

	} catch (const std::exception& e) {
		cookparms.sopAddError(SOP_MESSAGE, e.what());
	}

	// This makes the toggle behave like a one-shot button.
	// We do this OUTSIDE the try-catch block to ensure it always runs.
	cookparms.getNode()->setInt("execute", 0, 0, 0);  // (parm_name, index, time, value)
}
