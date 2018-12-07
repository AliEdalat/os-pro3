#include "NeuralNetwork.h"
#include <string.h>
#include <math.h>
#include <time.h>
#include <sstream>
#include <fstream>
#include <string>
#include <iostream>
#include "utilize.h"

struct MNIST_ImageFileHeader{
    uint32_t magicNumber;
    uint32_t maxImages;
    uint32_t imgWidth;
    uint32_t imgHeight;
};

struct MNIST_LabelFileHeader{
    uint32_t magicNumber;
    uint32_t maxImages;
};

uint32_t flipBytes(uint32_t n){

    uint32_t b0,b1,b2,b3;

    b0 = (n & 0x000000ff) <<  24u;
    b1 = (n & 0x0000ff00) <<   8u;
    b2 = (n & 0x00ff0000) >>   8u;
    b3 = (n & 0xff000000) >>  24u;

    return (b0 | b1 | b2 | b3);

}

void readImageFileHeader(FILE *imageFile, MNIST_ImageFileHeader *ifh){

    ifh->magicNumber =0;
    ifh->maxImages   =0;
    ifh->imgWidth    =0;
    ifh->imgHeight   =0;

    fread(&ifh->magicNumber, 4, 1, imageFile);
    ifh->magicNumber = flipBytes(ifh->magicNumber);

    fread(&ifh->maxImages, 4, 1, imageFile);
    ifh->maxImages = flipBytes(ifh->maxImages);

    fread(&ifh->imgWidth, 4, 1, imageFile);
    ifh->imgWidth = flipBytes(ifh->imgWidth);

    fread(&ifh->imgHeight, 4, 1, imageFile);
    ifh->imgHeight = flipBytes(ifh->imgHeight);
}

void readLabelFileHeader(FILE *imageFile, MNIST_LabelFileHeader *lfh){

    lfh->magicNumber =0;
    lfh->maxImages   =0;

    fread(&lfh->magicNumber, 4, 1, imageFile);
    lfh->magicNumber = flipBytes(lfh->magicNumber);

    fread(&lfh->maxImages, 4, 1, imageFile);
    lfh->maxImages = flipBytes(lfh->maxImages);

}

FILE *openMNISTImageFile(const char* fileName){

    FILE *imageFile;
    imageFile = fopen (fileName, "rb");
    if (imageFile == NULL) {
        printf("Abort! Could not fine MNIST IMAGE file: %s\n",fileName);
        exit(0);
    }

    MNIST_ImageFileHeader imageFileHeader;
    readImageFileHeader(imageFile, &imageFileHeader);

    return imageFile;
}

FILE *openMNISTLabelFile(const char* fileName){

    FILE *labelFile;
    labelFile = fopen (fileName, "rb");
    if (labelFile == NULL) {
        printf("Abort! Could not find MNIST LABEL file: %s\n",fileName);
        exit(0);
    }

    MNIST_LabelFileHeader labelFileHeader;
    readLabelFileHeader(labelFile, &labelFileHeader);

    return labelFile;
}


mnist_image get_image(FILE *imageFile){

    mnist_image img;
    size_t result;
    result = fread(&img, sizeof(img), 1, imageFile);
    if (result!=1) {
        printf("\nError when reading IMAGE file! Abort!\n");
        exit(1);
    }

    return img;
}

mnist_label get_label(FILE *labelFile){

    mnist_label lbl;
    size_t result;
    result = fread(&lbl, sizeof(lbl), 1, labelFile);
    if (result!=1) {
        printf("\nError when reading LABEL file! Abort!\n");
        exit(1);
    }

    return lbl;
}

NeuralNetwork::NeuralNetwork()
: hidden_nodes(NUMBER_OF_HIDDEN_CELLS)
, output_nodes(NUMBER_OF_OUTPUT_CELLS)
{
	for (int i = 0; i < NUMBER_OF_HIDDEN_CELLS; ++i)
	{
		hidden_nodes[i] = new HiddenNode();
	}

	for (int i = 0; i < NUMBER_OF_OUTPUT_CELLS; ++i)
	{
		output_nodes[i] = new OutputNode();
	}
	sem_init(&full, 0, 1);
	sem_init(&output_guard, 0, 1);
	sem_init(&predict_guard, 0, 1);
	sem_init(&hidden, 0, 0);
	sem_init(&inc, 0, 0);
	sem_init(&output, 0, 0);
	sem_init(&inc_output, 0, 0);
	sem_init(&prediction, 0, 0);
}

void NeuralNetwork::set_image_and_labels(std::string image, std::string label){
	this->image = image;
	this->label = label;
}

void* NeuralNetwork::calculate_hidden(void *arg){
	struct hidden_computer* info = (struct hidden_computer*)(arg);
	// std::cout << "cal_hidden" << std::endl;
	sem_wait(info->hidden);
	for (int i = info->start; i < info->start+info->len; ++i)
	{
		double temp = 0;
        for (int z = 0; z < NUMBER_OF_INPUT_CELLS; z++) {
            temp += (info->inputs)->pixel[z] * (*(info->hidden_nodes))[i]->get_weights()[z];
        }
        temp += (*(info->hidden_nodes))[i]->get_bias();
        (*(info->hidden_nodes))[i]->set_output((temp >= 0) ?  temp : 0);
	}
	sem_post(info->inc);
	sem_wait(info->output_guard);
	int size;
	sem_getvalue(info->inc, &size);
	// std::cout << "hidden" << std::endl;
	if (size == info->hidden_threads_number)
	{
		// for (int i = 0; i < 8; ++i)
		// {
		// 	sem_trywait(info->inc);
		// }
		for (int i = 0; i < 10; ++i)
		{
			sem_post(info->output);
		}
	}
	sem_post(info->output_guard);
	pthread_exit(NULL);
}

void* NeuralNetwork::calculate_output(void *arg){
	struct output_computer* info = (struct output_computer*)(arg);
	// std::cout << "cal_output" << std::endl;
	sem_wait(info->output);
	double temp = 0;
	for (int i = 0; i < NUMBER_OF_HIDDEN_CELLS; ++i)
	{
        temp += (*(info->hidden_nodes))[i]->get_output() * (*(info->output_nodes))[info->key]->get_weights()[i];
	}
	temp += 1/(1+ exp(-1* temp));
	(*(info->output_nodes))[info->key]->set_output(temp);
	sem_post(info->inc_output);
	sem_wait(info->predict_guard);
	int size;
	sem_getvalue(info->inc_output, &size);
	// std::cout << "output" << std::endl;
	if (size == 10)
	{
		// for (int i = 0; i < 10; ++i)
		// {
		// 	sem_trywait(info->inc_output);
		// }
		// std::cout << "prediction flag" << std::endl;
		sem_post(info->prediction);
	}
	sem_post(info->predict_guard);
	pthread_exit(NULL);	
}

void* NeuralNetwork::show_prediction_result(void *arg){
	struct prediction_computer* info = (struct prediction_computer*)(arg);
	//std::cout << "show_prediction_result" << std::endl;
	sem_wait(info->prediction);
	double max_num = 0;
    int max_index = 0;
    for (int i=0; i<NUMBER_OF_OUTPUT_CELLS; i++){

        if ((*(info->output_nodes))[i]->get_output() > max_num){
            max_num = (*(info->output_nodes))[i]->get_output();
            max_index = i;
        }
    }
    if (max_index != info->label) (*(info->errCount))++;
    printf("\n      Prediction: %d   Actual: %d ",max_index, info->label);
    displayProgress(info->imgCount, *(info->errCount), 5, 66);
    sem_post(info->full);
    // std::cout << "show_prediction_result_completed" << std::endl;
    pthread_exit(NULL);
}

void* NeuralNetwork::draw_image(void *arg){
	//std::cout << "draw entry!!" << std::endl;
	struct read_image_thread_info* info = (struct read_image_thread_info*)(arg);
    sem_wait(info->full);
    // display progress
    display_loading_progress_testing(info->imgCount,5,5);

    // Reading next image and corresponding label
    info->img = get_image(info->imageFile);
    info->lbl = get_label(info->labelFile);

    display_image(&info->img, 8,6);
    //printf("\n      Actual: %d\n", info->lbl);
    for (int i = 0; i < info->hidden_threads_number; ++i)
	{
		sem_post(info->hidden);
	}
    pthread_exit(NULL);
}

void NeuralNetwork::run(int hidden_threads_number){
	int errCount = 0;
	time_t startTime = time(NULL);
	clear_screen();
    printf("    MNIST-NN: a simple 2-layer neural network processing the MNIST handwriting images\n");
	display_image_frame(7,5);
	allocate_hidden_parameters();
	allocate_output_parameters();
	imageFile = openMNISTImageFile(image.c_str());
	labelFile = openMNISTLabelFile(label.c_str());
	pthread_t threads [12 + hidden_threads_number];
	int counter = 0;
	int start = 0;
	int size;
	struct read_image_thread_info read_info;
	struct prediction_computer prediction_info;
	struct hidden_computer hiddens_inf[hidden_threads_number];
	struct output_computer outputs_inf[10];
	for (int imgCount=0; imgCount < MNIST_MAX_TESTING_IMAGES; imgCount++){
		//sem_wait(&full);
		start = 0;
		sem_init(&hidden, 0, 0);
		sem_init(&inc, 0, 0);
		sem_init(&output, 0, 0);
		sem_init(&inc_output, 0, 0);
		sem_init(&prediction, 0, 0);
		sem_init(&output_guard, 0, 1);
		sem_init(&predict_guard, 0, 1);
		read_info.imageFile = imageFile;
		read_info.labelFile = labelFile;
		read_info.imgCount = imgCount;
		read_info.hidden = &hidden;
		read_info.full = &full;
		read_info.hidden_threads_number = hidden_threads_number;
		counter++;
		pthread_create(&threads[0],NULL,draw_image,&read_info);
		pthread_join(threads[0],NULL);
		for (int i = 0; i < hidden_threads_number; ++i)
		{
			hiddens_inf[i].start = start;
			start += (256/hidden_threads_number);
			hiddens_inf[i].len = (256/hidden_threads_number);
			hiddens_inf[i].hidden = &hidden;
			hiddens_inf[i].inc = &inc;
			hiddens_inf[i].output_guard = &output_guard;
			hiddens_inf[i].output = &output;
			hiddens_inf[i].hidden_nodes = &hidden_nodes;
			hiddens_inf[i].inputs = &read_info.img;
			hiddens_inf[i].hidden_threads_number = hidden_threads_number;
			pthread_create(&threads[i+1],NULL,calculate_hidden,&hiddens_inf[i]);
		}
		// for (int i = 0; i < 8; ++i)
		// {
		// 	pthread_join(threads[i+1],NULL);
		// }
		for (int i = 0; i < 10; ++i)
		{
			outputs_inf[i].key = i;
			outputs_inf[i].hidden_nodes = &hidden_nodes;
			outputs_inf[i].output_nodes = &output_nodes;
			outputs_inf[i].inc_output = &inc_output;
			outputs_inf[i].prediction = &prediction;
			outputs_inf[i].predict_guard = &predict_guard;
			outputs_inf[i].output = &output;
			pthread_create(&threads[i + hidden_threads_number + 1],NULL,calculate_output,&outputs_inf[i]);
		}
		// for (int i = 0; i < 10; ++i)
		// {
		// 	pthread_join(threads[i+9],NULL);
		// }
		prediction_info.label = read_info.lbl;
		prediction_info.imgCount = imgCount;
		prediction_info.errCount = &errCount;
		prediction_info.output_nodes = &output_nodes;
		prediction_info.prediction = &prediction;
		prediction_info.full = &full;
		pthread_create(&threads[11 + hidden_threads_number],NULL,show_prediction_result,&prediction_info);
		//pthread_join(threads[19],NULL);
		// while(1){
		// 	;
		// }
		for (int i = 0; i < (12 + hidden_threads_number); ++i)
		{
			pthread_join(threads[i], NULL);
		}
		//sem_post(&full);
    }
    // std::cout << counter << std::endl;
    fclose(imageFile);
    fclose(labelFile);
    locateCursor(38, 5);
    time_t endTime = time(NULL);
    double executionTime = difftime(endTime, startTime);
    printf("\n    DONE! Total execution time: %.1f sec\n\n",executionTime);
}

void NeuralNetwork::allocate_hidden_parameters(){
    int idx = 0;
    int bidx = 0;
    std::ifstream weights(HIDDEN_WEIGHTS_FILE);
    for(std::string line; getline(weights, line); )   //read stream line by line
    {
        std::stringstream in(line);
        for (int i = 0; i < 28*28; ++i){
            in >> hidden_nodes[idx]->get_weights()[i];
      }
      idx++;
    }
    weights.close();

    std::ifstream biases(OUTPUT_BIASES_FILE);
    double bios_temp;
    for(std::string line; getline(biases, line); )   //read stream line by line
    {
        std::stringstream in(line);
        in >> bios_temp;
        hidden_nodes[bidx]->set_bias(bios_temp);
        bidx++;
    }
    biases.close();

}

void NeuralNetwork::allocate_output_parameters(){
    int idx = 0;
    int bidx = 0;
    std::ifstream weights(OUTPUT_WEIGHTS_FILE); //"layersinfo.txt"
    for(std::string line; getline(weights, line); )   //read stream line by line
    {
        std::stringstream in(line);
        for (int i = 0; i < 256; ++i){
            in >> output_nodes[idx]->get_weights()[i];
      }
      idx++;
    }
    weights.close();

    std::ifstream biases(OUTPUT_BIASES_FILE);
    double bios_temp;
    for(std::string line; getline(biases, line); )   //read stream line by line
    {
        std::stringstream in(line);
        in >> bios_temp;
        output_nodes[bidx]->set_bias(bios_temp);
        bidx++;
    }
    biases.close();
}