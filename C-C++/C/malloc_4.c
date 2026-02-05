#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* duplicateString(const char *str);

int main(int argc, char *argv[]){
    
    char *input = malloc(100*sizeof(char));
    char *dup;
    printf("What word/phrase would you like to duplicate? ");
    fgets(input, 100, stdin);
    
    dup = duplicateString(input);
    // for(int i = 0; i < strlen(dup); i++){
    //     printf("%s", dup)
    // }
    printf("%s", dup);
    free(dup);
    free(input);
    return 0;
}

char* duplicateString(const char *str){
    char *duplicate = malloc(sizeof(char)*(strlen(str)+1));

    for(int i = 0; i <= strlen(str); i++){
        duplicate[i] = str[i];
    }
    
    return duplicate;
}