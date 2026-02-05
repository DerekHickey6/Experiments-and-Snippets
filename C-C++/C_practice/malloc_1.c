#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Defines

int main(int argc, char *argv[]){

    int number = 42;
    int *number_mem = malloc(sizeof(int));

    *number_mem = number;
    printf("%d", *number_mem);
    free(number_mem);
    return 0;
}