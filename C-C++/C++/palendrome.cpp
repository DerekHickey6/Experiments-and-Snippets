#include <iostream>
#include <cstring>
#include <iomanip>

using namespace std;

/* 	This is a program that takes a number from use input and tests to see if its a palindrome, if it is program stops.
	if input number is not palindrome, program will reverse the number and add it to the original number and produce
	a sum, it will then check to see if the sum is a palindrome (each 'step' is printed showing the number + reverse = sum).
	if the sum equals a palindrome, the program will display how many steps it took to get to a palindrome.
	if the sum is not a palindrome after 40 steps, program will display "NOT PAL (40 steps)" and exit program.
*/

string reverseNumber(unsigned long long n); 									
unsigned long long reverseInt (unsigned long long x);							
void updateNumbers(unsigned long long &num, unsigned long long &rev);			
void printPALsteps(unsigned long long num, unsigned long long rev, int counter);
void printPALcalc(unsigned long long num, unsigned long long rev, int counter);	

int main()
{
	unsigned long long num, rev, sum;				// initialize variables
	int count = 0;									// counter for number of 'steps'
	cout << "Number :> ";										
	cin >> num;										// prompt use to input a value
	rev = reverseInt(num);							// reverses input value and converts to an unsigned long long 

	while (num != rev && count < 40){
		count ++;
		printPALcalc(num, rev, count); 				// prints palindrome calculations for each iteration
		if (count < 40) {
			updateNumbers(num, rev);					// updates num & rev to new values based off sum of previous calculation
		}
	}
	printPALsteps(num, rev, count); 				// prints palindrome status and total calculations (steps) done
}


// Functions
string reverseNumber(unsigned long long n)			// ** i would normally combine reverseNumber + reverseInt into 1 function, but i was unsure if i am allowed as reverseNumber was in original supplied code
{
	string rev, x = to_string(n);
	for(int i = x.length()-1; i >= 0; i--){
		rev += x[i];
	}
	return rev;
}
unsigned long long reverseInt (unsigned long long x)
{
	return stoull(reverseNumber(x));
}
void updateNumbers(unsigned long long &num, unsigned long long &rev){
	num += rev;
	rev = reverseInt(num);
}
void printPALsteps(unsigned long long num, unsigned long long rev, int counter){
	if (num == rev){
		cout << "PAL (" << counter << " steps)" << endl;
	} else {
		cout << "NOT PAL (" << counter << " steps)" << endl; 
	}
}
void printPALcalc(unsigned long long num, unsigned long long rev, int counter){
	cout << setw(2) << counter << ". ";	
	cout << num << " + " << rev << " = " << num + rev << endl;
}