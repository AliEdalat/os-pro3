#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <pthread.h> 
#include <semaphore.h>

#include <string>
#include <vector>

#include "HiddenNode.h"
#include "OutputNode.h"
#include "mnist_image.h"

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

struct image_info
{
	struct mnist_image image;
	mnist_label label;
};

struct read_image_thread_info
{
	sem_t* full;
	sem_t* hidden;
	sem_t* guard_other;
	FILE* imageFile;
	FILE* labelFile;
	struct image_info* slice;
	bool* read_new_slice;
	int imgCount;
	int hidden_threads_number;
	mnist_image img;
    mnist_label lbl;
};

struct hidden_computer
{
	sem_t* hidden;
	sem_t* inc;
	sem_t* output;
	sem_t* output_guard;
	sem_t* full;
	std::vector<HiddenNode*>* hidden_nodes;
	int start;
	int len;
	int hidden_threads_number;
	struct mnist_image* inputs;
	
};

struct output_computer
{
	sem_t* output;
	sem_t* inc_output;
	sem_t* prediction;
	sem_t* predict_guard;
	sem_t* guard_other;
	int key;
	std::vector<HiddenNode*>* hidden_nodes;
	std::vector<OutputNode*>* output_nodes;
	
};

struct prediction_computer
{
	sem_t* prediction;
	sem_t* full;
	std::vector<OutputNode*>* output_nodes;
	int* errCount;
	mnist_label label;
	int imgCount;
};

class NeuralNetwork
{
public:
	NeuralNetwork();
	void set_image_and_labels(std::string image, std::string label);
	void run(int hidden_threads_number = 8);


private:
	sem_t full;
	sem_t hidden;
	sem_t inc;
	sem_t output;
	sem_t inc_output;
	sem_t prediction;
	sem_t output_guard;
	sem_t predict_guard;
	sem_t guard_other;
	FILE* imageFile;
	FILE* labelFile;
	std::string image;
	std::string label;
	std::vector<HiddenNode*> hidden_nodes;
	std::vector<OutputNode*> output_nodes;

	void allocate_hidden_parameters();
	void allocate_output_parameters();

	static void* draw_image(void *arg);
	static void* calculate_hidden(void *arg);
	static void* calculate_output(void *arg);
	static void* show_prediction_result(void *arg);
	
};

#endif