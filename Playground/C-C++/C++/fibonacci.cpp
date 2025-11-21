#include <iostream>
#include <cstdio>
#include <cmath>
#include <iomanip>
using namespace std;

int fibonacci(int n); 		//Calculates value of fibonacci term
int numDigits(int x);		//Defines number of digits in a integer (currently up to 15 digits)
void printStar(int x);      //Prints the '*' in the number of certain fibonacci sequences

/* 
This is a program that will print out the Fibonacci Sequence terms in ascending order,
the sequence will stop once the term exceeds 9 digits. There is an option included (commented out)
to choose the desired digit amount for the final term from user input.
The program will then print the sum of the terms printed. 
*/

int main()
{	
	int i = 1;
	int sum = 0;
	int desiredDigits = 9;
	
	//  --- if included, this will allow user to select desired digits in final term ---
	//cout << "Welcome to fibbonacci, how many digits do you want in final term? (Below 11)";
    //cin >> desiredDigits;      

	printStar(5);				//prints '*' related to the fibonacci value passed in
    printStar(6);
    printStar(7);

	do{
		cout << "Term " << setw(3) <<  i << ":" << setw(10) << fibonacci(i) << endl;//Prints term # and fibonacci value in ascending order
		sum = sum + fibonacci(i);													//Calculates sum of all digits
		i++;
	} while (numDigits(fibonacci(i)) <= desiredDigits); 							//Compares number of digits in Fibonacci value to digit limit for a stopping point
	
	printStar(8);
	cout << "Sum of the " << i-1 << " terms:" << setw(11) << sum << endl;
	printStar(9);
}

int fibonacci(int n)		//Calculates value of fibonacci term
{
	long int sum;

	if (n <= 2) {
		return 1;
	} else if (n > 2) {
	return fibonacci(n-1) + fibonacci(n-2);
	}
	return 0;
}
int numDigits(int x)		//Defines number of digits in a integer (currently up to 15 digits)
{
	int d = 1;
	while (d <= 15){
		if (x < pow(10,d)){
			return d;
		}
		d++;
	}
	return d;
	
}
void printStar(int x)		//Prints the '*' in the number of certain fibonacci sequences
{
	for (int i = 0; i < (fibonacci(x)); i++)
    {
        cout << "*";
    }
    cout << endl;
}
/*

ALTERNATE FUNCTION for 'numDigits'
if (x < 10){
		digits = 1;
	} else if (x < 100){
		digits = 2;
	} else if (x < 1000){
		digits = 3;
	} else if (x < 10000){
		digits = 4;
	} else if (x < 100000){
		digits = 5;
	} else if (x < 1000000){
		digits = 6;
	} else if (x < 10000000){
		digits = 7;
	} else if (x < 100000000){
		digits = 8;
	} else if (x < 1000000000){
		digits = 9;
	}*/