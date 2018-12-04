#include "HiddenNode.h"

void HiddenNode::set_bias(double bias_input){
	bias = bias_input;
}

void HiddenNode::set_output(double output_input){
	output = output_input;
}

void HiddenNode::set_weights(double* weights_input){
	for (int i = 0; i < NUMBER_OF_INPUT_CELLS; ++i)
	{
		weights[i] = weights_input[i];
	}
}

double HiddenNode::get_bias(){
	return bias;
}

double HiddenNode::get_output(){
	return output;
}

double* HiddenNode::get_weights(){
	return weights;
}