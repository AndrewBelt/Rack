#include "widget/Widget.hpp"
#include "widget/event.hpp"
#include "app.hpp"
#include <algorithm>


namespace rack {
namespace widget {


Widget::~Widget() {
	// You should only delete orphaned widgets
	assert(!parent);
	clearChildren();
}

void Widget::setPos(math::Vec pos) {
	box.pos = pos;
	// RepositionEvent
	RepositionEvent eReposition;
	onReposition(eReposition);
}

void Widget::setSize(math::Vec size) {
	box.size = size;
	// ResizeEvent
	ResizeEvent eResize;
	onResize(eResize);
}

void Widget::show() {
	if (visible)
		return;
	visible = true;
	// ShowEvent
	ShowEvent eShow;
	onShow(eShow);
}

void Widget::hide() {
	if (!visible)
		return;
	visible = false;
	// HideEvent
	HideEvent eHide;
	onHide(eHide);
}

void Widget::requestDelete() {
	requestedDelete = true;
}

math::Rect Widget::getChildrenBoundingBox() {
	math::Vec min = math::Vec(INFINITY, INFINITY);
	math::Vec max = math::Vec(-INFINITY, -INFINITY);
	for (Widget *child : children) {
		if (!child->visible)
			continue;
		min = min.min(child->box.getTopLeft());
		max = max.max(child->box.getBottomRight());
	}
	return math::Rect::fromMinMax(min, max);
}

math::Vec Widget::getRelativeOffset(math::Vec v, Widget *relative) {
	if (this == relative) {
		return v;
	}
	v = v.plus(box.pos);
	if (parent) {
		v = parent->getRelativeOffset(v, relative);
	}
	return v;
}

math::Rect Widget::getViewport(math::Rect r) {
	math::Rect bound;
	if (parent) {
		bound = parent->getViewport(box);
	}
	else {
		bound = box;
	}
	bound.pos = bound.pos.minus(box.pos);
	return r.clamp(bound);
}

void Widget::addChild(Widget *child) {
	assert(child);
	assert(!child->parent);
	child->parent = this;
	children.push_back(child);
	// AddEvent
	AddEvent eAdd;
	child->onAdd(eAdd);
}

void Widget::removeChild(Widget *child) {
	assert(child);
	// Make sure `this` is the child's parent
	assert(child->parent == this);
	// RemoveEvent
	RemoveEvent eRemove;
	child->onRemove(eRemove);
	// Prepare to remove widget from the event state
	APP->event->finalizeWidget(child);
	// Delete child from children list
	auto it = std::find(children.begin(), children.end(), child);
	assert(it != children.end());
	children.erase(it);
	// Revoke child's parent
	child->parent = NULL;
}

void Widget::clearChildren() {
	for (Widget *child : children) {
		// RemoveEvent
		RemoveEvent eRemove;
		child->onRemove(eRemove);
		APP->event->finalizeWidget(child);
		child->parent = NULL;
		delete child;
	}
	children.clear();
}

void Widget::step() {
	for (auto it = children.begin(); it != children.end();) {
		Widget *child = *it;
		// Delete children if a delete is requested
		if (child->requestedDelete) {
			// RemoveEvent
			RemoveEvent eRemove;
			child->onRemove(eRemove);
			APP->event->finalizeWidget(child);
			it = children.erase(it);
			child->parent = NULL;
			delete child;
			continue;
		}

		child->step();
		it++;
	}
}

void Widget::draw(const DrawArgs &args) {
	// Iterate children
	for (Widget *child : children) {
		// Don't draw if invisible
		if (!child->visible)
			continue;
		// Don't draw if child is outside clip box
		if (!args.clipBox.isIntersecting(child->box))
			continue;

		DrawArgs childCtx = args;
		// Intersect child clip box with self
		childCtx.clipBox = childCtx.clipBox.intersect(child->box);
		childCtx.clipBox.pos = childCtx.clipBox.pos.minus(child->box.pos);

		nvgSave(args.vg);
		nvgTranslate(args.vg, child->box.pos.x, child->box.pos.y);

		child->draw(childCtx);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
		// Call deprecated draw function, which does nothing by default
		child->draw(args.vg);
#pragma GCC diagnostic pop

		nvgRestore(args.vg);
	}
}


} // namespace widget
} // namespace rack