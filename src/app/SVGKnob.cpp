#include "app.hpp"


namespace rack {


SVGKnob::SVGKnob() {
	shadow = new CircularShadow();
	addChild(shadow);

	tw = new TransformWidget();
	addChild(tw);

	sw = new SVGWidget();
	tw->addChild(sw);
}

void SVGKnob::setSVG(std::shared_ptr<SVG> svg) {
	sw->setSVG(svg);
	tw->box.size = sw->box.size;
	box.size = sw->box.size;
	shadow->box.size = sw->box.size;
	shadow->box.pos = Vec(0, sw->box.size.y * 0.1);
	// shadow->box = shadow->box.grow(Vec(2, 2));
}

void SVGKnob::step() {
	// Re-transform TransformWidget if dirty
	if (dirty) {
		float angle = 0.0;
		if (isfinite(minValue) && isfinite(maxValue))
			angle = rescale(value, minValue, maxValue, minAngle, maxAngle);
		tw->identity();
		// Rotate SVG
		Vec center = sw->box.getCenter();
		tw->translate(center);
		tw->rotate(angle);
		tw->translate(center.neg());
	}
	FramebufferWidget::step();
}

void SVGKnob::onChange(EventChange &e) {
	dirty = true;
	Knob::onChange(e);
}


} // namespace rack
