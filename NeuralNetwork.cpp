#include "NeuralNetwork.h"
#include <string.h>
#include <math.h>
#include <sstream> //this header file is needed when using stringstream
#include <fstream>
#include <string>
#include <iostream>

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

void locateCursor(const int row, const int col){
    printf("%c[%d;%dH",27,row,col);
}

void clear_screen(){
    printf("\e[1;1H\e[2J");
}

void display_image(mnist_image *img, int row, int col){

    char imgStr[(MNIST_IMG_HEIGHT * MNIST_IMG_WIDTH)+((col+1)*MNIST_IMG_HEIGHT)+1];
    strcpy(imgStr, "");

    for (int y=0; y<MNIST_IMG_HEIGHT; y++){

        for (int o=0; o<col-2; o++) strcat(imgStr," ");
        strcat(imgStr,"|");

        for (int x=0; x<MNIST_IMG_WIDTH; x++){
            strcat(imgStr, img->pixel[y*MNIST_IMG_HEIGHT+x] ? "X" : "." );
        }
        strcat(imgStr,"\n");
    }

    if (col!=0 && row!=0) locateCursor(row, 0);
    printf("%s",imgStr);
}

void display_image_frame(int row, int col){

    if (col!=0 && row!=0) locateCursor(row, col);

    printf("------------------------------\n");

    for (int i=0; i<MNIST_IMG_HEIGHT; i++){
        for (int o=0; o<col-1; o++) printf(" ");
        printf("|                            |\n");
    }

    for (int o=0; o<col-1; o++) printf(" ");
    printf("------------------------------");

}

void display_loading_progress_testing(int imgCount, int y, int x){

    float progress = (float)(imgCount+1)/(float)(MNIST_MAX_TESTING_IMAGES)*100;

    if (x!=0 && y!=0) locateCursor(y, x);

    printf("Testing image No. %5d of %5d images [%d%%]\n                                  ",(imgCount+1),MNIST_MAX_TESTING_IMAGES,(int)progress);

}

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

void displayProgress(int imgCount, int errCount, int y, int x){

    double successRate = 1 - ((double)errCount/(double)(imgCount+1));

    if (x!=0 && y!=0) locateCursor(y, x);

    printf("Result: Correct=%5d  Incorrect=%5d  Success-Rate= %5.2f%% \n",imgCount+1-errCount, errCount, successRate*100);


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
	//sem_init(&hidden, 0, 0);
}

void NeuralNetwork::set_image_and_labels(std::string image, std::string label){
	imageFile = openMNISTImageFile(image.c_str());
	labelFile = openMNISTLabelFile(label.c_str());
}

void* NeuralNetwork::calculate_hidden(void *arg){
	struct hidden_computer* info = (struct hidden_computer*)(arg);
	//std::cout << "cal_hidden" << std::endl;
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
	int size;
	sem_getvalue(info->inc, &size);
	//std::cout << "hidden" << std::endl;
	if (size == 8)
	{
		//std::cout << "output flag" << std::endl;
		sem_init(info->output, 0, 10);
	}
	pthread_exit(NULL);
}

void* NeuralNetwork::calculate_output(void *arg){
	struct output_computer* info = (struct output_computer*)(arg);
	//std::cout << "cal_output" << std::endl;
	sem_wait(info->output);
	double temp = 0;
	for (int i = 0; i < NUMBER_OF_HIDDEN_CELLS; ++i)
	{
        temp += (*(info->hidden_nodes))[i]->get_output() * (*(info->output_nodes))[info->key]->get_weights()[i];
	}
	temp += 1/(1+ exp(-1* temp));
	(*(info->output_nodes))[info->key]->set_output(temp);
	sem_post(info->inc_output);
	int size;
	sem_getvalue(info->inc_output, &size);
	//std::cout << "output" << std::endl;
	if (size == 10)
	{
		//std::cout << "prediction flag" << std::endl;
		sem_init(info->prediction, 0, 1);
	}
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
    sem_post(info->prediction);
    //std::cout << "show_prediction_result_completed" << std::endl;
}

void* NeuralNetwork::draw_image(void *arg){
	//std::cout << "draw entry!!" << std::endl;
	struct read_image_thread_info* info = (struct read_image_thread_info*)(arg);
    // display progress
    display_loading_progress_testing(info->imgCount,5,5);

    // Reading next image and corresponding label
    info->img = get_image(info->imageFile);
    info->lbl = get_label(info->labelFile);

    display_image(&info->img, 8,6);
    //printf("\n      Actual: %d\n", info->lbl);
    sem_init(info->hidden, 0, 8);
    pthread_exit(NULL);
}

void NeuralNetwork::run(){
	int errCount = 0;
	//display_image_frame(7,5);
	allocate_hidden_parameters();
	allocate_output_parameters();
	pthread_t threads [20];
	int counter = 0;
	int start = 0;
	int size;
	struct read_image_thread_info read_info;
	struct prediction_computer prediction_info;
	struct hidden_computer hiddens_inf[8];
	struct output_computer outputs_inf[10];
	for (int imgCount=0; imgCount<MNIST_MAX_TESTING_IMAGES; imgCount++){
		sem_wait(&full);
		start = 0;
		sem_init(&hidden, 0, 0);
		sem_init(&inc, 0, 0);
		sem_init(&output, 0, 0);
		sem_init(&inc_output, 0, 0);
		sem_init(&prediction, 0, 0);
		read_info.imageFile = imageFile;
		read_info.labelFile = labelFile;
		read_info.imgCount = imgCount;
		read_info.hidden = &hidden;
		counter++;
		pthread_create(&threads[0],NULL,draw_image,&read_info);
		pthread_join(threads[0],NULL);
		for (int i = 0; i < 8; ++i)
		{
			hiddens_inf[i].start = start;
			start += 32;
			hiddens_inf[i].len = 32;
			hiddens_inf[i].hidden = &hidden;
			hiddens_inf[i].inc = &inc;
			hiddens_inf[i].output = &output;
			hiddens_inf[i].hidden_nodes = &hidden_nodes;
			hiddens_inf[i].inputs = &read_info.img;
			pthread_create(&threads[i+1],NULL,calculate_hidden,&hiddens_inf[i]);
		}
		for (int i = 0; i < 8; ++i)
		{
			pthread_join(threads[i+1],NULL);
		}
		for (int i = 0; i < 10; ++i)
		{
			outputs_inf[i].key = i;
			outputs_inf[i].hidden_nodes = &hidden_nodes;
			outputs_inf[i].output_nodes = &output_nodes;
			outputs_inf[i].inc_output = &inc_output;
			outputs_inf[i].prediction = &prediction;
			outputs_inf[i].output = &output;
			pthread_create(&threads[i+9],NULL,calculate_output,&outputs_inf[i]);
		}
		for (int i = 0; i < 10; ++i)
		{
			pthread_join(threads[i+9],NULL);
		}
		prediction_info.label = read_info.lbl;
		prediction_info.imgCount = imgCount;
		prediction_info.errCount = &errCount;
		prediction_info.output_nodes = &output_nodes;
		prediction_info.prediction = &prediction;
		pthread_create(&threads[19],NULL,show_prediction_result,&prediction_info);
		pthread_join(threads[19],NULL);
		sem_post(&full);
    }
    // std::cout << counter << std::endl;
    fclose(imageFile);
    fclose(labelFile);
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