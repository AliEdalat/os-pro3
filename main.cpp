#include "NeuralNetwork.h"

using namespace std;

int main(int argc, char const *argv[])
{
	NeuralNetwork n;
	n.set_image_and_labels("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
	n.run();
	return 0;
}