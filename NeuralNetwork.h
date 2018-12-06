#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <pthread.h> 
#include <semaphore.h>

#include <string>
#include <vector>

#include "HiddenNode.h"
#include "OutputNode.h"

#define MNIST_MAX_TESTING_IMAGES 10000
#define MNIST_IMG_WIDTH 28
#define MNIST_IMG_HEIGHT 28
#define NUMBER_OF_HIDDEN_CELLS 256
#define NUMBER_OF_OUTPUT_CELLS 10
#define HIDDEN_WEIGHTS_FILE "net_params/hidden_weights.txt"
#define HIDDEN_BIASES_FILE "net_params/hidden_biases.txt"
#define OUTPUT_WEIGHTS_FILE "net_params/out_weights.txt"
#define OUTPUT_BIASES_FILE "net_params/out_biases.txt"

typedef uint8_t mnist_label;

struct mnist_image{
    uint8_t pixel[28*28];
};

struct read_image_thread_info
{
	sem_t* hidden;
	FILE* imageFile;
	FILE* labelFile;
	int imgCount;
	mnist_image img;
    mnist_label lbl;
};

struct hidden_computer
{
	sem_t* hidden;
	sem_t* inc;
	std::vector<HiddenNode*>* hidden_nodes;
	int start;
	int len;
	int* max;
	bool* min;
	struct mnist_image* inputs;
	
};

class NeuralNetwork
{
public:
	NeuralNetwork();
	void set_image_and_labels(std::string image, std::string label);
	void run();


private:
	sem_t full;
	sem_t hidden[MNIST_MAX_TESTING_IMAGES];
	sem_t inc[MNIST_MAX_TESTING_IMAGES];
	FILE* imageFile;
	FILE* labelFile;
	std::vector<HiddenNode*> hidden_nodes;
	std::vector<OutputNode*> output_nodes;

	void allocate_hidden_parameters();
	void allocate_output_parameters();
	static void* draw_image(void *arg);
	static void* calc_hidden(void *arg);
	
};

#endif