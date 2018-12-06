#ifndef OUTPUT_NODE
#define OUTPUT_NODE

#define NUMBER_OF_HIDDEN_CELLS 256

class OutputNode
{
public:
	OutputNode(){}
	void set_bias(double bias_input);
	void set_output(double output_input);
	void set_weight(double weight_input, int index);
	double get_bias();
	double get_output();
	double* get_weights();

private:
	double weights[NUMBER_OF_HIDDEN_CELLS];
    double bias;
    double output;
};

#endif