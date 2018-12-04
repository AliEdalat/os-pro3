#include "NeuralNetwork.h"
#include <string.h>

struct MNIST_Image{
    uint8_t pixel[28*28];
};

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

typedef uint8_t MNIST_Label;

void locateCursor(const int row, const int col){
    printf("%c[%d;%dH",27,row,col);
}

void clear_screen(){
    printf("\e[1;1H\e[2J");
}

void display_image(MNIST_Image *img, int row, int col){

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


MNIST_Image get_image(FILE *imageFile){

    MNIST_Image img;
    size_t result;
    result = fread(&img, sizeof(img), 1, imageFile);
    if (result!=1) {
        printf("\nError when reading IMAGE file! Abort!\n");
        exit(1);
    }

    return img;
}

MNIST_Label get_label(FILE *labelFile){

    MNIST_Label lbl;
    size_t result;
    result = fread(&lbl, sizeof(lbl), 1, labelFile);
    if (result!=1) {
        printf("\nError when reading LABEL file! Abort!\n");
        exit(1);
    }

    return lbl;
}

void NeuralNetwork::set_image_and_labels(std::string image, std::string label){
	imageFile = openMNISTImageFile(image.c_str());
	labelFile = openMNISTLabelFile(label.c_str());
}

void NeuralNetwork::run(){
	int errCount = 0;
	display_image_frame(7,5);
	for (int imgCount=0; imgCount<MNIST_MAX_TESTING_IMAGES; imgCount++){
        // display progress
        display_loading_progress_testing(imgCount,5,5);

        // Reading next image and corresponding label
        MNIST_Image img = get_image(imageFile);
        MNIST_Label lbl = get_label(labelFile);

        display_image(&img, 8,6);
        printf("\n      Actual: %d \n", lbl);
    }
}

// void allocate_hidden_parameters(){
//     int idx = 0;
//     int bidx = 0;
//     ifstream weights(HIDDEN_WEIGHTS_FILE);
//     for(string line; getline(weights, line); )   //read stream line by line
//     {
//         stringstream in(line);
//         for (int i = 0; i < 28*28; ++i){
//             in >> hidden_nodes[idx].weights[i];
//       }
//       idx++;
//     }
//     weights.close();

//     ifstream biases(OUTPUT_BIASES_FILE);
//     for(string line; getline(biases, line); )   //read stream line by line
//     {
//         stringstream in(line);
//         in >> hidden_nodes[bidx].bias;
//         bidx++;
//     }
//     biases.close();

// }

// void allocate_output_parameters(){
//     int idx = 0;
//     int bidx = 0;
//     ifstream weights(OUTPUT_WEIGHTS_FILE); //"layersinfo.txt"
//     for(string line; getline(weights, line); )   //read stream line by line
//     {
//         stringstream in(line);
//         for (int i = 0; i < 256; ++i){
//             in >> output_nodes[idx].weights[i];
//       }
//       idx++;
//     }
//     weights.close();

//     ifstream biases(OUTPUT_BIASES_FILE);
//     for(string line; getline(biases, line); )   //read stream line by line
//     {
//         stringstream in(line);
//         in >> output_nodes[bidx].bias;
//         bidx++;
//     }
//     biases.close();

// }