#include "HiddenNode.h"

void HiddenNode::set_bias(double bias_input){
	bias = bias_input;
}

void HiddenNode::set_output(double output_input){
	output = output_input;
}

void HiddenNode::set_weight(double weight_input, int index){
		weights[index] = weight_input;
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