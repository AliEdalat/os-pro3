#ifndef HIDDEN_NODE
#define HIDDEN_NODE

#define NUMBER_OF_INPUT_CELLS 784

class HiddenNode
{
public:
	HiddenNode(){}
	void set_bias(double bias_input);
	void set_output(double output_input);
	void set_weight(double weight_input, int index);
	double get_bias();
	double get_output();
	double* get_weights();

private:
	double weights[NUMBER_OF_INPUT_CELLS];
    double bias;
    double output;
};

#endif