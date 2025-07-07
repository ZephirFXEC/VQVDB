#include "SOP_VQVDB_Decoder.hpp"

#include <GU/GU_Detail.h>
#include <UT/UT_DSOVersion.h>

#include "Backend/TorchBackend.hpp"
#include "Utils/Utils.hpp"
#include "VQVAECodec.hpp"

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

bool SOP_VQVDB_DecoderCache::initializeCodec() {
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

void SOP_VQVDB_DecoderVerb::cook(const CookParms& cookparms) const {
	auto& sopparms = cookparms.parms<SOP_VQVDB_DecoderParms>();
	const auto sopcache = dynamic_cast<SOP_VQVDB_DecoderCache*>(cookparms.cache());

	GU_Detail* gdp = cookparms.gdh().gdpNC();

	// init codec if not already initialized
	if (!sopcache || !sopcache->initializeCodec()) {
		cookparms.sopAddError(SOP_MESSAGE, "Failed to initialize VQVDB codec.");
		return;
	}


	const auto& in_path = sopparms.getInputfile();

	if (in_path.empty()) {
		return;
	}
	openvdb::FloatGrid::Ptr output_grid = openvdb::FloatGrid::create();

	try {
		// --- Run Decoder ---
		cookparms.sopAddMessage(SOP_MESSAGE, "Starting VQ-VDB decoding...");

		const VQVAECodec codec(std::make_shared<TorchBackend>());

		sopcache->codec_->decompress(in_path.data(), output_grid, sopparms.getBatchsize());
	} catch (const std::exception& e) {
		cookparms.sopAddError(SOP_MESSAGE, e.what());
	}

	GU_PrimVDB::buildFromGrid(*gdp, output_grid, nullptr, sopparms.getVdbname());
}
