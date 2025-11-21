#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(){
    char string[100] = "the dog was in the house today";
    char rev_string[30] = "";
    char rev_word[10] = "";
    int word_count = 0;

    char *token = strtok(string, " ");
    if(token == NULL) exit(1);
    while(token != NULL){
        int len = strlen(token);
        
        for(int i = 0; i < len; i++){
            rev_word[i] = token[len - i - 1];
        }
        rev_word[len] = 0;

        strcat(rev_string, rev_word);
        strcat(rev_string, " ");
        word_count++;
        token = strtok(NULL, " ");
    }
    
    printf("%s", rev_string);
}