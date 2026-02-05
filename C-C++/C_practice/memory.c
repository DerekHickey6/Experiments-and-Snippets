#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

int main(){

    int size;
    printf("How many int elements do you need?: ");
    scanf("%d", &size);
    printf("%d", size);

    int *arr = malloc(size * sizeof(int));
    for (int i = 0; i < size; i++){
        scanf("%d", &arr[i]);
    }
    free(arr);
    return 0;
}