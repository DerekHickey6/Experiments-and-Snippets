#include <stdio.h>
#include <stdlib.h>

#define MAX_SIZE 10

void rotate(int num_rotations, int arr[], int size);

 int main(int argc, char *argv[]){

        if (argc > 2) {
        printf("Usage: %s <number>\n", argv[0]);
        return 1;
        }

         int i = 0;
         char *sep = "";
         
         int arr_1[MAX_SIZE] = {1,2,3,4,5,6,7,8,9,10};

         // This for statement prints the content of arr_1 to console
         for (i = 0; i < MAX_SIZE; i++) {
             printf("%s%d", sep, arr_1[i]);
             sep = ", ";
         }
         printf("\n");

         // Rotates by argument value
         rotate(atoi(argv[1]), arr_1, MAX_SIZE);

         sep = "";
         for (i = 0; i < MAX_SIZE; i++) {
             printf("%s%d", sep, arr_1[i]);
             sep = ", ";
         }
         printf("\n");
         
         // Rotates back to original

         rotate(-atoi(argv[1]), arr_1, MAX_SIZE);

          sep = "";
          for (i = 0; i < MAX_SIZE; i++) {
              printf("%s%d", sep, arr_1[i]);
              sep = ", ";
          }
          printf("\n");
          
}

/*
  * Design a function rotate(int num, .........) that take a number as a parameter (int num) and any other parameters as required
  * When called, the function will modify arr_1 by moving each of its values to an index lowered by num. If the new index becomes negative, then MAX_SIZE is added to get the correct index
  * For example, if num = 2, then a number at index 4 is moved to index 2 (4 - num) and a number at index 1 is moved to index 9(1 - num + MAX_SIZE)
  * Another example, if at the start, arr_1 is {2,4,6,8,1,5},
  * calling arr_1 with num = 2 should change arr_1 to {6,8,1,5,2,4}
  */

// Rotates array by num_rotations
void rotate(int num_rotations, int arr[], int size){
    int temp = 0;

    // if rotations is negative, shifts array to the right (changes shift direction)
    if(num_rotations < 0){
        num_rotations = size + num_rotations;
    }
    // Rotates array by num_rotations
    for(int i = 0; i < num_rotations; i++){
        temp = arr[0];
        for(int j = 0; j < size; j++){
            arr[j] = arr[j+1];
        }
        arr[size-1] = temp;
    }
}