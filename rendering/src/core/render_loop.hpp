#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

#include "core/depth_pyramid.hpp"

class Window;
struct CameraState;
struct CameraLimits;
struct InputState;
struct RendererState;
struct Timing;
struct UIState;

class RenderLoop {
   public:
	RenderLoop(Window& window, CameraState& cam, CameraLimits& limits, InputState& input, RendererState& renderer, Timing& timing, UIState& ui);

	void drawFrame();

   private:
	void updateTiming();
	void updateTitle();
	void resetVisibilityStats() noexcept;
	void clearVisibilityDebugState() noexcept;
	void updateVisibilityAndCache();
	void renderOcclusionDepthData(bool hasUploadedBlockData, const glm::mat4& viewProjection, int viewportY);

	Window& windowRef;
	CameraState& cameraRef;
	CameraLimits& limitsRef;
	InputState& inputRef;
	RendererState& rendererRef;
	Timing& timingRef;
	UIState& uiRef;
	std::vector<uint32_t> blockDebugStates;
	size_t lastDebugStateBlockCount{0};
	depth_pyramid::DepthPyramid previousDepthPyramid;
	depth_pyramid::AsyncDepthReadback depthReadback;
};
