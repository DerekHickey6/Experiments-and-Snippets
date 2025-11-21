#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]){
    char line[200];
    char rev_word[20];
    char rev_line[200];
    

    while(fgets(line, sizeof(line), stdin)){
        
        rev_line[0] = '\0';
        rev_word[0] = '\0';
        char *token = strtok(line, " ,.");
        while(token != NULL){

            int len = strlen(token);
            
            for(int i = 0; i < len; i++){
                rev_word[i] = token[len-i-1];
            }
            rev_word[len] = '\0';
            rev_word[strcspn(rev_word, "\r\n")] = 0;
            strcat(rev_line, rev_word);
            strcat(rev_line, " ");

            token = strtok(NULL, " ,.");
        }
    }
        
        printf("%s", rev_line);
}