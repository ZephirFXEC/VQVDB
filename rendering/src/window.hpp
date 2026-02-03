#pragma once

#include <string>

struct GLFWwindow;

class Window {
   public:
	bool create(int width, int height, const std::string& title);
	void destroy();

	[[nodiscard]] bool shouldClose() const;
	void swapBuffers() const;
	void pollEvents() const;
	void setUserPointer(void* ptr);
	void setTitle(const std::string& title);
	GLFWwindow* raw() const { return handle; }

   private:
	GLFWwindow* handle{nullptr};
};
