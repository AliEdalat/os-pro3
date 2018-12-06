#include "OutputNode.h"

void OutputNode::set_bias(double bias_input){
	bias = bias_input;
}

void OutputNode::set_output(double output_input){
	output = output_input;
}

void OutputNode::set_weight(double weight_input, int index){
		weights[index] = weight_input;
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