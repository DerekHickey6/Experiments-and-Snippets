#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SIZE 200

int main(int argc, char *argv[]){

    char source[MAX_SIZE] = "The house is flying to the moon";
    char destination[MAX_SIZE];
    char temp[2];

    destination[0] = '\0';
    temp[1] ='\0';

    for(int i = 0; source[i] != '\0'; i++){
        int exists = 0;
        temp[0] = source[i];
        for(int j = 0; destination[j] != '\0'; j++){
            if(source[i] == destination[j]){
                exists = 1;
                break;
            }
        }
        if(!exists){
            temp[0] = source[i];
            strcat(destination, temp);
        }
    }
    printf("%s", destination);
}