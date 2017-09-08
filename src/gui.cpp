#include <map>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "gui.hpp"
#include "app.hpp"

#define NANOVG_GL2_IMPLEMENTATION
// #define NANOVG_GL3_IMPLEMENTATION
#include "../ext/nanovg/src/nanovg_gl.h"
// Hack to get framebuffer objects working on OpenGL 2 (we blindly assume the extension is supported)
#define NANOVG_FBO_VALID 1
#include "../ext/nanovg/src/nanovg_gl_utils.h"
#define BLENDISH_IMPLEMENTATION
#include "../ext/oui-blendish/blendish.h"
#define NANOSVG_IMPLEMENTATION
#include "../ext/nanosvg/src/nanosvg.h"

#ifdef ARCH_MAC
	// For CGAssociateMouseAndMouseCursorPosition
	#include <ApplicationServices/ApplicationServices.h>
#endif

#if GLFW_VERSION_MINOR == 0 && GLFW_VERSION_REVISION <= 4
#define OLD_GLFW_VERSION
#endif

namespace rack {

static GLFWwindow *window = NULL;
std::shared_ptr<Font> gGuiFont;
NVGcontext *gVg = NULL;
float gPixelRatio = 0.0;


void windowSizeCallback(GLFWwindow* window, int width, int height) {
	gScene->box.size = Vec(width, height);
}

void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods) {
	if (action == GLFW_PRESS) {
		// onMouseDown
		Widget *w = gScene->onMouseDown(gMousePos, button);

		if (button == GLFW_MOUSE_BUTTON_LEFT) {
			if (w) {
				// onDragStart
				w->onDragStart();
			}
			gDraggedWidget = w;

			if (w != gSelectedWidget) {
				if (gSelectedWidget) {
					w->onDeselect();
				}
				if (w) {
					w->onSelect();
				}
				gSelectedWidget = w;
			}
		}
	}
	else if (action == GLFW_RELEASE) {
		// onMouseUp
		Widget *w = gScene->onMouseUp(gMousePos, button);

		if (button == GLFW_MOUSE_BUTTON_LEFT) {
			if (gDraggedWidget) {
				// onDragDrop
				w->onDragDrop(gDraggedWidget);
			}
			// gDraggedWidget might have been set to null in the last event, recheck here
			if (gDraggedWidget) {
				// onDragEnd
				gDraggedWidget->onDragEnd();
			}
			gDraggedWidget = NULL;
			gDragHoveredWidget = NULL;
		}
	}
}

void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
	Vec mousePos = Vec(xpos, ypos).round();
	Vec mouseRel = mousePos.minus(gMousePos);

#ifdef ARCH_MAC
	// Workaround for Mac. We can't use GLFW_CURSOR_DISABLED because it's buggy, so implement it on our own.
	// This is not an ideal implementation. For example, if the user drags off the screen, the new mouse position will be clamped.
	int mouseMode = glfwGetInputMode(window, GLFW_CURSOR);
	if (mouseMode == GLFW_CURSOR_HIDDEN) {
		// CGSetLocalEventsSuppressionInterval(0.0);
		glfwSetCursorPos(window, gMousePos.x, gMousePos.y);
		CGAssociateMouseAndMouseCursorPosition(true);
		mousePos = gMousePos;
	}
	// Because sometimes the cursor turns into an arrow when its position is on the boundary of the window
	glfwSetCursor(window, NULL);
#endif

	gMousePos = mousePos;

	// onMouseMove
	Widget *hovered = gScene->onMouseMove(gMousePos, mouseRel);

	if (gDraggedWidget) {
		// onDragMove
		gDraggedWidget->onDragMove(mouseRel);

		if (hovered != gDragHoveredWidget) {
			if (gDragHoveredWidget) {
				gDragHoveredWidget->onDragLeave(gDraggedWidget);
			}
			if (hovered) {
				hovered->onDragEnter(gDraggedWidget);
			}
			gDragHoveredWidget = hovered;
		}
	}
	else {
		if (hovered != gHoveredWidget) {
			if (gHoveredWidget) {
				// onMouseLeave
				gHoveredWidget->onMouseLeave();
			}
			if (hovered) {
			// onMouseEnter
				hovered->onMouseEnter();
			}
			gHoveredWidget = hovered;
		}
	}
}

void cursorEnterCallback(GLFWwindow* window, int entered) {
	if (!entered) {
		if (gHoveredWidget) {
			gHoveredWidget->onMouseLeave();
		}
		gHoveredWidget = NULL;
	}
}

void scrollCallback(GLFWwindow *window, double x, double y) {
	Vec scrollRel = Vec(x, y);
	// onScroll
	gScene->onScroll(gMousePos, scrollRel.mult(-95));
}

void charCallback(GLFWwindow *window, unsigned int codepoint) {
	if (gSelectedWidget) {
		gSelectedWidget->onText(codepoint);
	}
}

static int lastWindowX, lastWindowY, lastWindowWidth, lastWindowHeight;

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS || action == GLFW_REPEAT) {
		if (key == GLFW_KEY_F11 || key == GLFW_KEY_ESCAPE) {
			/*
			// Toggle fullscreen
			GLFWmonitor *monitor = glfwGetWindowMonitor(window);
			if (monitor) {
				// Window mode
				glfwSetWindowMonitor(window, NULL, lastWindowX, lastWindowY, lastWindowWidth, lastWindowHeight, 0);
			}
			else {
				// Fullscreen
				glfwGetWindowPos(window, &lastWindowX, &lastWindowY);
				glfwGetWindowSize(window, &lastWindowWidth, &lastWindowHeight);
				monitor = glfwGetPrimaryMonitor();
				assert(monitor);
				const GLFWvidmode *mode = glfwGetVideoMode(monitor);
				glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
			}
			*/
		}
		else {
			if (gSelectedWidget) {
				gSelectedWidget->onKey(key);
			}
		}
	}
}

void errorCallback(int error, const char *description) {
	fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

void renderGui() {
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	int windowWidth, windowHeight;
	glfwGetWindowSize(window, &windowWidth, &windowHeight);
	gPixelRatio = (float)width / windowWidth;

	bool visible = glfwGetWindowAttrib(window, GLFW_VISIBLE) && !glfwGetWindowAttrib(window, GLFW_ICONIFIED);
	if (visible) {
		// Update and render
		glViewport(0, 0, width, height);
		glClearColor(0.0, 0.0, 0.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

		nvgBeginFrame(gVg, width, height, gPixelRatio);

		nvgSave(gVg);
		nvgReset(gVg);
		nvgScale(gVg, gPixelRatio, gPixelRatio);
		gScene->draw(gVg);
		nvgRestore(gVg);

		nvgEndFrame(gVg);
	}

	glfwSwapBuffers(window);
}

void guiInit() {
	int err;

	// Set up GLFW
	glfwSetErrorCallback(errorCallback);
	err = glfwInit();
	assert(err);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	// glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	// glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	// glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	// glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifndef OLD_GLFW_VERSION
	glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
#endif
	std::string title = gApplicationName + " " + gApplicationVersion;
	window = glfwCreateWindow(1000, 750, title.c_str(), NULL, NULL);
	assert(window);
	glfwMakeContextCurrent(window);

	glfwSwapInterval(1);

	glfwSetWindowSizeCallback(window, windowSizeCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);
	// glfwSetCursorPosCallback(window, cursorPosCallback);
	glfwSetCursorEnterCallback(window, cursorEnterCallback);
	glfwSetScrollCallback(window, scrollCallback);
	glfwSetCharCallback(window, charCallback);
	glfwSetKeyCallback(window, keyCallback);

	// Set up GLEW
	glewExperimental = GL_TRUE;
	err = glewInit();
	assert(err == GLEW_OK);

	// Check framebuffer support
	assert(GLEW_EXT_framebuffer_object);

	// GLEW generates GL error because it calls glGetString(GL_EXTENSIONS), we'll consume it here.
	glGetError();

#ifndef OLD_GLFW_VERSION
	glfwSetWindowSizeLimits(window, 640, 480, GLFW_DONT_CARE, GLFW_DONT_CARE);
#endif

	// Set up NanoVG
	gVg = nvgCreateGL2(NVG_ANTIALIAS);
	// gVg = nvgCreateGL3(NVG_ANTIALIAS);
	assert(gVg);

	// Set up Blendish
	gGuiFont = Font::load("res/DejaVuSans.ttf");
	bndSetFont(gGuiFont->handle);
	// bndSetIconImage(loadImage("res/icons.png"));
}

void guiDestroy() {
	gGuiFont.reset();
	nvgDeleteGL2(gVg);
	// nvgDeleteGL3(gVg);
	glfwDestroyWindow(window);
	glfwTerminate();
}

void guiRun() {
	assert(window);
	{
		int width, height;
		glfwGetWindowSize(window, &width, &height);
		windowSizeCallback(window, width, height);
	}
	gGuiFrame = 0;
	double lastTime = 0.0;
	while(!glfwWindowShouldClose(window)) {
		gGuiFrame++;
		glfwPollEvents();
		{
			double xpos, ypos;
			glfwGetCursorPos(window, &xpos, &ypos);
			cursorPosCallback(window, xpos, ypos);
		}
		gScene->step();

		renderGui();

		double currTime = glfwGetTime();
		// printf("%lf fps\n", 1.0/(currTime - lastTime));
		lastTime = currTime;
		(void) lastTime;
	}
}

bool guiIsKeyPressed(int key) {
	return glfwGetKey(window, key) == GLFW_PRESS;
}

void guiCursorLock() {
#ifdef ARCH_MAC
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
#else
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
#endif
}

void guiCursorUnlock() {
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}


////////////////////
// resources
////////////////////

Font::Font(const std::string &filename) {
	handle = nvgCreateFont(gVg, filename.c_str(), filename.c_str());
	if (handle >= 0) {
		fprintf(stderr, "Loaded font %s\n", filename.c_str());
	}
	else {
		fprintf(stderr, "Failed to load font %s\n", filename.c_str());
	}
}

Font::~Font() {
	// There is no NanoVG deleteFont() function yet, so do nothing
}

std::shared_ptr<Font> Font::load(const std::string &filename) {
	static std::map<std::string, std::weak_ptr<Font>> cache;
	auto sp = cache[filename].lock();
	if (!sp)
		cache[filename] = sp = std::make_shared<Font>(filename);
	return sp;
}

////////////////////
// Image
////////////////////

Image::Image(const std::string &filename) {
	handle = nvgCreateImage(gVg, filename.c_str(), NVG_IMAGE_REPEATX | NVG_IMAGE_REPEATY);
	if (handle > 0) {
		fprintf(stderr, "Loaded image %s\n", filename.c_str());
	}
	else {
		fprintf(stderr, "Failed to load image %s\n", filename.c_str());
	}
}

Image::~Image() {
	// TODO What if handle is invalid?
	nvgDeleteImage(gVg, handle);
}

std::shared_ptr<Image> Image::load(const std::string &filename) {
	static std::map<std::string, std::weak_ptr<Image>> cache;
	auto sp = cache[filename].lock();
	if (!sp)
		cache[filename] = sp = std::make_shared<Image>(filename);
	return sp;
}

////////////////////
// SVG
////////////////////

SVG::SVG(const std::string &filename) {
	handle = nsvgParseFromFile(filename.c_str(), "px", 96.0);
	if (handle) {
		fprintf(stderr, "Loaded SVG %s\n", filename.c_str());
	}
	else {
		fprintf(stderr, "Failed to load SVG %s\n", filename.c_str());
	}
}

SVG::~SVG() {
	nsvgDelete(handle);
}

std::shared_ptr<SVG> SVG::load(const std::string &filename) {
	static std::map<std::string, std::weak_ptr<SVG>> cache;
	auto sp = cache[filename].lock();
	if (!sp)
		cache[filename] = sp = std::make_shared<SVG>(filename);
	return sp;
}


} // namespace rack
