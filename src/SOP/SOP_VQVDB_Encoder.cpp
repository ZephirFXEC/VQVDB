#include "SOP_VQVDB_Encoder.hpp"

#include <GU/GU_Detail.h>
#include <UT/UT_DSOVersion.h>

#include "Backend/TorchBackend.hpp"
#include "Utils/Utils.hpp"
#include "VQVAECodec.hpp"

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


bool SOP_VQVDB_EncoderCache::initializeCodec() {
	// If codec already exists, do nothing.
	if (codec_) {
		return true;
	}

	try {
		auto backend = std::make_shared<TorchBackend>();
		codec_ = std::make_unique<VQVAECodec>(backend);
	} catch (const std::exception& e) {
		codec_.reset();
		return false;
	}

	return true;
}

void SOP_VQVDB_EncoderVerb::cook(const CookParms& cookparms) const {
	auto& sopparms = cookparms.parms<SOP_VQVDB_EncoderParms>();
	const auto sopcache = dynamic_cast<SOP_VQVDB_EncoderCache*>(cookparms.cache());

	// init codec if not already initialized
	if (!sopcache || !sopcache->initializeCodec()) {
		cookparms.sopAddError(SOP_MESSAGE, "Failed to initialize VQVDB codec.");
		return;
	}


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
			return;
		}
		if (!input_gdp) {
			cookparms.sopAddError(SOP_MESSAGE, "No input geometry connected.");
			return;
		}

		// --- Load Grid ---
		openvdb::FloatGrid::Ptr grid;
		if (auto err = loadGrid<openvdb::FloatGrid>(input_gdp, grid, sopparms.getVdbname()); err != UT_ERROR_NONE) {
			cookparms.sopAddError(SOP_MESSAGE, "Failed to load VDB grid from input.");
		}
		if (!grid) {
			cookparms.sopAddError(SOP_MESSAGE, "VDB grid is null or not found.");
			return;
		}

		// --- Run Encoder ---
		cookparms.sopAddMessage(SOP_MESSAGE, "Starting VQ-VDB encoding...");


		sopcache->codec_->compress(grid, out_path, batch_size);

		cookparms.sopAddMessage(SOP_MESSAGE, ("Successfully saved to " + out_path).c_str());

	} catch (const std::exception& e) {
		cookparms.sopAddError(SOP_MESSAGE, e.what());
	}

	// This makes the toggle behave like a one-shot button.
	// We do this OUTSIDE the try-catch block to ensure it always runs.
	cookparms.getNode()->setInt("execute", 0, 0, 0);  // (parm_name, index, time, value)
}
