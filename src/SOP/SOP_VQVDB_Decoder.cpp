#include "SOP_VQVDB_Decoder.hpp"

#include <GU/GU_Detail.h>
#include <UT/UT_DSOVersion.h>

#include "../orchestrator/VQVAECodec.hpp"
#include "Utils/Utils.hpp"

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
	if (codec_) {
		return true;
	}

	try {
		CodecConfig config;
		config.device = CodecConfig::Device::CUDA;
		config.source = EmbeddedModel{};

		std::unique_ptr<IVQVAECodec> backend = IVQVAECodec::create(config, BackendType::ONNX);
		if (!backend) {
			return false;
		}

		codec_ = std::make_unique<VQVAECodec>(std::move(backend));

	} catch (const std::exception& e) {
		std::cerr << "Caught exception during codec initialization: " << e.what() << std::endl;
		codec_.reset();
		return false;
	}

	return true;
}

void SOP_VQVDB_DecoderVerb::cook(const CookParms& cookparms) const {
	auto& sopparms = cookparms.parms<SOP_VQVDB_DecoderParms>();
	const auto sopcache = dynamic_cast<SOP_VQVDB_DecoderCache*>(cookparms.cache());

	if (!sopcache || !sopcache->initializeCodec()) {
		cookparms.sopAddError(SOP_MESSAGE, "Failed to initialize VQ-VDB codec backend.");
		return;
	}

	const std::filesystem::path in_path{sopparms.getInputfile().toStdString()};
	if (in_path.empty()) {
		return;  // No file specified, do nothing.
	}
	if (!std::filesystem::exists(in_path)) {
		cookparms.sopAddError(SOP_MESSAGE, "Input file does not exist.");
		return;
	}

	std::vector<openvdb::FloatGrid::Ptr> output_grids;
	try {
		cookparms.sopAddMessage(SOP_MESSAGE, "Starting VQ-VDB decoding...");

		// Call the decompress method.
		sopcache->codec_->decompress(in_path, output_grids, sopparms.getBatchsize());

	} catch (const std::exception& e) {
		cookparms.sopAddError(SOP_MESSAGE, e.what());
		return;  // Stop execution on error
	}

	// If we are here, decoding was successful.
	GU_Detail* gdp = cookparms.gdh().gdpNC();
	gdp->clearAndDestroy();  // Clear any existing geometry.

	for (const auto& grid : output_grids) {
		GU_PrimVDB::buildFromGrid(*gdp, grid, nullptr, grid->getName().c_str());
	}
}
