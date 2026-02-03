#pragma once

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

	Window& windowRef;
	CameraState& cameraRef;
	CameraLimits& limitsRef;
	InputState& inputRef;
	RendererState& rendererRef;
	Timing& timingRef;
	UIState& uiRef;
};
