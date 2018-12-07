#ifndef UTILIZE
#define UTILIZE

#include <iostream>
#include <string.h>
#include "mnist_image.h"

#define MNIST_MAX_TESTING_IMAGES 10000
#define MNIST_IMG_WIDTH 28
#define MNIST_IMG_HEIGHT 28

void locateCursor(const int row, const int col);
void clear_screen();
void display_image(mnist_image *img, int row, int col);
void display_image_frame(int row, int col);
void display_loading_progress_testing(int imgCount, int y, int x);
void displayProgress(int imgCount, int errCount, int y, int x);

#endif