
#include "Cmp.hpp"

void Cmp::onReset() {
	_thresholdState = LOW;
	_windowState = LOW;
}

void Cmp::step() {
	if (!(
		outputs[GREATER_OUTPUT].active ||
		outputs[LESS_OUTPUT].active ||
		outputs[EQUAL_OUTPUT].active ||
		outputs[NOT_EQUAL_OUTPUT].active
	)) {
		return;
	}

	float a = params[A_PARAM].value * 10.0f;
	if (inputs[A_INPUT].active) {
		a = clamp(a + inputs[A_INPUT].value, -12.0f, 12.0f);
	}

	float b = params[B_PARAM].value * 10.0f;
	if (inputs[B_INPUT].active) {
		b = clamp(b + inputs[B_INPUT].value, -12.0f, 12.0f);
	}

	float window = params[WINDOW_PARAM].value;
	if (inputs[WINDOW_INPUT].active) {
		window *= clamp(inputs[WINDOW_INPUT].value / 10.0f, 0.0f, 1.0f);
	}
	window *= 10.0f;

	float high = 10.0f;
	float low = 0.0f;
	if (params[OUTPUT_PARAM].value > 0.5f) {
		high = 5.0f;
		low = -5.0f;
	}

	int lag = -1;
	stepChannel(
		a >= b,
		high,
		low,
		_thresholdState,
		_thresholdLag,
		lag,
		outputs[GREATER_OUTPUT],
		outputs[LESS_OUTPUT]
	);
	stepChannel(
		fabsf(a - b) <= window,
		high,
		low,
		_windowState,
		_windowLag,
		lag,
		outputs[EQUAL_OUTPUT],
		outputs[NOT_EQUAL_OUTPUT]
	);
}

void Cmp::stepChannel(
	bool high,
	float highValue,
	float lowValue,
	State& state,
	int& channelLag,
	int& lag,
	Output& highOutput,
	Output& lowOutput
) {
	switch (state) {
		case LOW: {
			if (high) {
				if (lag < 0) {
					lag = lagInSamples();
				}
				if (lag < 1) {
					state = HIGH;
				}
				else {
					state = LAG_HIGH;
					channelLag = lag;
				}
			}
			break;
		}
		case HIGH: {
			if (!high) {
				if (lag < 0) {
					lag = lagInSamples();
				}
				if (lag < 1) {
					state = LOW;
				}
				else {
					state = LAG_LOW;
					channelLag = lag;
				}
			}
			break;
		}
		case LAG_LOW: {
			if (!high) {
				--channelLag;
				if(channelLag == 0) {
					state = LOW;
				}
			}
			else {
				state = HIGH;
			}
			break;
		}
		case LAG_HIGH: {
			if (high) {
				--channelLag;
				if(channelLag == 0) {
					state = HIGH;
				}
			}
			else {
				state = LOW;
			}
			break;
		}
	};

	switch (state) {
		case LOW:
		case LAG_HIGH: {
			highOutput.value = lowValue;
			lowOutput.value = highValue;
			break;
		}
		case HIGH:
		case LAG_LOW: {
			highOutput.value = highValue;
			lowOutput.value = lowValue;
			break;
		}
	}
}

int Cmp::lagInSamples() {
	float lag = params[LAG_PARAM].value;
	if (inputs[LAG_INPUT].active) {
		lag *= clamp(inputs[LAG_INPUT].value / 10.0f, 0.0f, 1.0f);
	}
	return lag * lag * engineGetSampleRate();
}

struct CmpWidget : ModuleWidget {
	static constexpr int hp = 6;

	CmpWidget(Cmp* module) : ModuleWidget(module) {
		box.size = Vec(RACK_GRID_WIDTH * hp, RACK_GRID_HEIGHT);

		{
			SVGPanel *panel = new SVGPanel();
			panel->box.size = box.size;
			panel->setBackground(SVG::load(assetPlugin(plugin, "res/Cmp.svg")));
			addChild(panel);
		}

		addChild(Widget::create<ScrewSilver>(Vec(0, 0)));
		addChild(Widget::create<ScrewSilver>(Vec(box.size.x - 15, 365)));

		// generated by svg_widgets.rb
		auto aParamPosition = Vec(8.0, 46.0);
		auto bParamPosition = Vec(53.0, 46.0);
		auto windowParamPosition = Vec(8.0, 151.0);
		auto lagParamPosition = Vec(53.0, 151.0);
		auto outputParamPosition = Vec(25.5, 251.0);

		auto aInputPosition = Vec(10.5, 87.0);
		auto bInputPosition = Vec(55.5, 87.0);
		auto windowInputPosition = Vec(10.5, 192.0);
		auto lagInputPosition = Vec(55.5, 192.0);

		auto greaterOutputPosition = Vec(16.0, 283.0);
		auto lessOutputPosition = Vec(50.0, 283.0);
		auto equalOutputPosition = Vec(16.0, 319.0);
		auto notEqualOutputPosition = Vec(50.0, 319.0);
		// end generated by svg_widgets.rb

		addParam(ParamWidget::create<Knob29>(aParamPosition, module, Cmp::A_PARAM, -1.0, 1.0, 0.0));
		addParam(ParamWidget::create<Knob29>(bParamPosition, module, Cmp::B_PARAM, -1.0, 1.0, 0.0));
		addParam(ParamWidget::create<Knob29>(windowParamPosition, module, Cmp::WINDOW_PARAM, 0.0, 1.0, 0.5));
		addParam(ParamWidget::create<Knob29>(lagParamPosition, module, Cmp::LAG_PARAM, 0.0, 1.0, 0.1));
		{
			auto w = ParamWidget::create<Knob16>(outputParamPosition, module, Cmp::OUTPUT_PARAM, 0.0, 1.0, 0.0);
			auto k = dynamic_cast<SVGKnob*>(w);
			k->snap = true;
			k->minAngle = 3.0f * (M_PI / 8.0f);
			k->maxAngle = 5.0f * (M_PI / 8.0f);
			k->speed = 3.0;
			addParam(w);
		}

		addInput(Port::create<Port24>(aInputPosition, Port::INPUT, module, Cmp::A_INPUT));
		addInput(Port::create<Port24>(bInputPosition, Port::INPUT, module, Cmp::B_INPUT));
		addInput(Port::create<Port24>(windowInputPosition, Port::INPUT, module, Cmp::WINDOW_INPUT));
		addInput(Port::create<Port24>(lagInputPosition, Port::INPUT, module, Cmp::LAG_INPUT));

		addOutput(Port::create<Port24>(greaterOutputPosition, Port::OUTPUT, module, Cmp::GREATER_OUTPUT));
		addOutput(Port::create<Port24>(lessOutputPosition, Port::OUTPUT, module, Cmp::LESS_OUTPUT));
		addOutput(Port::create<Port24>(equalOutputPosition, Port::OUTPUT, module, Cmp::EQUAL_OUTPUT));
		addOutput(Port::create<Port24>(notEqualOutputPosition, Port::OUTPUT, module, Cmp::NOT_EQUAL_OUTPUT));
	}
};

RACK_PLUGIN_MODEL_INIT(Bogaudio, Cmp) {
   Model *modelCmp = createModel<Cmp, CmpWidget>("Bogaudio-Cmp", "CMP", "comparator", LOGIC_TAG);
   return modelCmp;
}