#include "OutputNode.h"

void OutputNode::set_bias(double bias_input){
	bias = bias_input;
}

void OutputNode::set_output(double output_input){
	output = output_input;
}

void OutputNode::set_weights(double* weights_input){
	for (int i = 0; i < NUMBER_OF_HIDDEN_CELLS; ++i)
	{
		weights[i] = weights_input[i];
	}
}

double OutputNode::get_bias(){
	return bias;
}

double OutputNode::get_output(){
	return output;
}

double* OutputNode::get_weights(){
	return weights;
}