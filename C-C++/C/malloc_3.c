#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main(int argc, char *argv[]){
    int rows, columns;
    printf("How many rows? ");
    scanf("%d", &rows);
    printf("How many columns? ");
    scanf("%d", &columns);

    // Allocates memory - rows first, then columns
    int **matrix = malloc(rows * sizeof(int *));
    for(int i = 0; i < rows; i++){
        matrix[i] = malloc(columns * sizeof(int));
    }
    // fills 2-D array
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            matrix[i][j] = i+j;
        }
    }
    // prints array
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            printf("%d, ", matrix[i][j]);
        }
        printf("\n");
    }
    // frees memory in reverse order, starting with columns, then rows
    for(int i = 0; i < rows; i++){
        free(matrix[i]);
    }
    free(matrix);
}