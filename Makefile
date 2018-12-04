HiddenNode: HiddenNode.cpp HiddenNode.h
	g++ -c HiddenNode.cpp -o HiddenNode
OutputNode: OutputNode.cpp OutputNode.h
	g++ -c OutputNode.cpp -o OutputNode
NeuralNetwork: NeuralNetwork.cpp NeuralNetwork.h
	g++ -c NeuralNetwork.cpp -o NeuralNetwork
.: OutputNode HiddenNode NeuralNetwork main.cpp
	g++ HiddenNode OutputNode NeuralNetwork main.cpp

clean:
	rm OutputNode HiddenNode