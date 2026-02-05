#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]){
    
    int memory;

    printf("How much memory do you need?");
    scanf("%d", &memory);
    int *array = malloc(memory*sizeof(int));

    for(int i= 0; i < memory; i++){
        array[i] = i+1;
        printf("%d, ", array[i]);
    }
    free(array);

    return 0;
}