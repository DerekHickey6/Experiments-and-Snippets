#include <stdio.h>
#include <stdbool.h>
#include <string.h>

typedef struct {
    int width;
    int length;
} rectangle;

typedef struct {
    int x;
    int y;
} position;

typedef struct {
    char owner[30];
    rectangle rectangle;
    position position;

} buildingPlan;

int main() {
    rectangle myrectange = {5, 10};

    printf("length is equal to %d\n", myrectange.length);

    myrectange.length = 15;
    printf("length is equal to %d\n", myrectange.length);

    buildingPlan myHouse = {"Derek Hickey", {40, 60}, {1200, 2400}};

    printf("The house at %d %d (size %d x %d) is owned by %s\n", 
            myHouse.position.x,
            myHouse.position.y,
            myHouse.rectangle.width,
            myHouse.rectangle.length,
            myHouse.owner
    );

    return 0;
}

