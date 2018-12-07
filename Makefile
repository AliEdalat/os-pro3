HiddenNode: HiddenNode.cpp HiddenNode.h
	g++ -std=c++11 -c HiddenNode.cpp -o HiddenNode
OutputNode: OutputNode.cpp OutputNode.h
	g++ -std=c++11 -c OutputNode.cpp -o OutputNode
NeuralNetwork: utilize NeuralNetwork.cpp NeuralNetwork.h
	g++ -std=c++11 -c NeuralNetwork.cpp -o NeuralNetwork
utilize: utilize.cpp utilize.h
	g++ -std=c++11 -c utilize.cpp -o utilize
.: HiddenNode OutputNode NeuralNetwork main.cpp
	g++ -std=c++11 -pthread utilize HiddenNode OutputNode NeuralNetwork main.cpp

clean:
	rm OutputNode HiddenNode NeuralNetwork