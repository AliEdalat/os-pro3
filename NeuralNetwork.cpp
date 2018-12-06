#include "NeuralNetwork.h"
#include <string.h>
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

void* NeuralNetwork::calc_hidden(void *arg){
	struct hidden_computer* info = (struct hidden_computer*)(arg);
	sem_wait(info->hidden);
	std::cout << info->start << std::endl;
	for (int i = info->start; i < info->start+info->len; ++i)
	{
		double temp = 0;
        for (int z = 0; z < NUMBER_OF_INPUT_CELLS; z++) {
            temp += (info->inputs)->pixel[z] * (*(info->hidden_nodes))[i]->get_weights()[z];
        }
        temp += (*(info->hidden_nodes))[i]->get_bias();
        (*(info->hidden_nodes))[i]->set_output((temp >= 0) ?  temp : 0);
	}
	int size;
	sem_getvalue(info->hidden, &size);
	// if (size == 0)
	// {
	// 	*(info->min) = true;
	// }else if (size == 8 && *(info->min) == true){
	// 	*(info->max) = true;
	// }
	//sem_wait(info->inc);
	(*(info->max))++;
	std::cout << (*(info->max)) << std::endl;
	//sem_post(info->inc);
	std::cout << "jjdjdjjdjddjjdjd   >>>>>>> " << size << std::endl; 
	sem_post(info->hidden);
	pthread_exit(NULL);
}

void* NeuralNetwork::draw_image(void *arg){
	std::cout << "draw entry!!" << std::endl;
	struct read_image_thread_info* info = (struct read_image_thread_info*)(arg);
    // display progress
    //display_loading_progress_testing(info->imgCount,5,5);

    // Reading next image and corresponding label
    info->img = get_image(info->imageFile);
    info->lbl = get_label(info->labelFile);

    //display_image(&img, 8,6);
    printf("\n      Actual: %d\n", info->lbl);
    sem_init(info->hidden, 0, 8);
    pthread_exit(NULL);
}

void NeuralNetwork::run(){
	int errCount = 0;
	display_image_frame(7,5);
	allocate_hidden_parameters();
	allocate_output_parameters();
	pthread_t threads [20];
	int counter = 0;
	int start = 0;
	int size;
	struct read_image_thread_info temp;
	struct hidden_computer hiddens_inf[8];
	int max [MNIST_MAX_TESTING_IMAGES] = {0};
	for (int imgCount=0; imgCount<MNIST_MAX_TESTING_IMAGES; imgCount++){
		sem_wait(&full);
		start = 0;
		//int max = 0;
		sem_init(&hidden[imgCount], 0, 0);
		sem_init(&inc[imgCount], 0, 1);
		temp.hidden = &hidden[imgCount];
		temp.imageFile = imageFile;
		temp.labelFile = labelFile;
		counter++;
		pthread_create(&threads[0],NULL,draw_image,&temp);
		for (int i = 0; i < 8; ++i)
		{
			hiddens_inf[i].start = start;
			start += 32;
			hiddens_inf[i].len = 32;
			hiddens_inf[i].hidden = &hidden[imgCount];
			hiddens_inf[i].hidden_nodes = &hidden_nodes;
			hiddens_inf[i].inputs = &temp.img;
			hiddens_inf[i].max = &max[imgCount];
			hiddens_inf[i].inc = &inc[imgCount];
			pthread_create(&threads[i+1],NULL,calc_hidden,&hiddens_inf[i]);
		}
		
		sem_getvalue(&hidden[imgCount], &size);
		std::cout << "jjdjdjjdjddjjdjd   >>>>>>> " << size << std::endl; 
		while (max[imgCount] != 8)
		{
			//std::cout << max[imgCount] << std::endl;
			;
		}
		sem_post(&full);
    }
    std::cout << counter << std::endl;
    for (int i = 0; i < 9; ++i)
    {
    	pthread_join(threads[i],NULL);
    }
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