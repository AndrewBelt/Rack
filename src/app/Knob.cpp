#include "app.hpp"
#include "gui.hpp"
#include "engine.hpp"
// For GLFW_KEY_LEFT_CONTROL, etc.
#include <GLFW/glfw3.h>

namespace rack {

#define KNOB_SENSITIVITY 0.0015


void Knob::onDragStart(EventDragStart &e) {
	guiCursorLock();
	dragValue = value;
	randomizable = false;
}

void Knob::onDragMove(EventDragMove &e) {
	float sens = sensitivity;
	if (sens == 0.f) {
		sens = KNOB_SENSITIVITY;
	}
	// Drag slower if Mod
	float delta = sens * (maxValue - minValue) * -e.mouseRel.y;
	if (guiIsModPressed())
		delta /= 16.0;
	dragValue += delta;
	if (snap)
		setValue(roundf(dragValue));
	else
		setValue(dragValue);
}

void Knob::onDragEnd(EventDragEnd &e) {
	guiCursorUnlock();
	randomizable = true;
}

void Knob::onChange(EventChange &e) {
	if (!module)
		return;

	if (snap)
		engineSetParam(module, paramId, value);
	else
		engineSetParamSmooth(module, paramId, value);
}


} // namespace rack
