#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <pthread.h> 
#include <semaphore.h>

#include <string> 

#define MNIST_MAX_TESTING_IMAGES 10000
#define MNIST_IMG_WIDTH 28
#define MNIST_IMG_HEIGHT 28

class NeuralNetwork
{
public:
	NeuralNetwork(){}
	void set_image_and_labels(std::string image, std::string label);
	void run();


private:
	sem_t full;
	FILE* imageFile;
	FILE* labelFile;
	
};

#endif