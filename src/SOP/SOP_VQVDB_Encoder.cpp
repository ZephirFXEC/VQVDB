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
	if (codec_) {
		return true;
	}

	try {
		CodecConfig config;
		config.device = torch::cuda::is_available() ? CodecConfig::Device::CUDA : CodecConfig::Device::CPU;
		config.source = EmbeddedModel{};

		std::unique_ptr<IVQVAECodec> backend = IVQVAECodec::create(config);
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


void SOP_VQVDB_EncoderVerb::cook(const CookParms& cookparms) const {
	auto& sopparms = cookparms.parms<SOP_VQVDB_EncoderParms>();
	if (sopparms.getExecute() == 0) {
		return;
	}

	// Always reset the button state, even if we fail later.
	cookparms.getNode()->setInt("execute", 0, 0, 0);

	const auto sopcache = dynamic_cast<SOP_VQVDB_EncoderCache*>(cookparms.cache());
	if (!sopcache || !sopcache->initializeCodec()) {
		cookparms.sopAddError(SOP_MESSAGE, "Failed to initialize VQ-VDB codec backend.");
		return;
	}

	try {
		const std::filesystem::path out_path{sopparms.getOutputpath().toStdString()};
		const int batch_size = sopparms.getBatchsize();

		if (out_path.empty()) {
			cookparms.sopAddError(SOP_MESSAGE, "Output path must be specified.");
			return;
		}

		const GU_Detail* input_gdp = cookparms.inputGeo(0);
		if (!input_gdp) {
			cookparms.sopAddError(SOP_MESSAGE, "No input geometry connected.");
			return;
		}

		// --- Load Grid --- (Your existing logic is good)
		std::vector<openvdb::GridBase::Ptr> grids;
		if (const auto err = loadGrid(input_gdp, grids); err != UT_ERROR_NONE) {
			cookparms.sopAddError(SOP_MESSAGE, "Failed to load VDB grid from input.");
			return;
		}

		std::vector<openvdb::FloatGrid::Ptr> float_grids;
		for (const auto& grid : grids) {
			if (auto float_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid)) {
				float_grids.push_back(float_grid);
			} else {
				cookparms.sopAddError(SOP_MESSAGE, ("Skipped non-float grid: " + grid->getName()).c_str());
				return;
			}
		}

		// --- Run Encoder ---
		cookparms.sopAddMessage(SOP_MESSAGE, "Starting VQ-VDB encoding...");

		UT_Interrupt boss("Compressing...");
		sopcache->codec_->compress(float_grids, out_path, batch_size, &boss);

		cookparms.sopAddMessage(SOP_MESSAGE, ("Successfully saved to " + out_path.string()).c_str());

	} catch (const std::exception& e) {
		cookparms.sopAddError(SOP_MESSAGE, e.what());
	}
}
