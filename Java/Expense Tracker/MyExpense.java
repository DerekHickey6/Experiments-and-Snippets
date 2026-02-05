//**************************************************************************************************************************************************************//
//  MyExpense.java                                                                                                                                              //
//                                                                                                                                                              //
//  Derek Hickey T00777678                                                                                                                                      //
//  May 19, 2025                                                                                                                                                //
//  COMP 1231 Assignment 3 Question 1                                                                                                                           //
//  This class represents a program that manages monthly expenses. It allows the user to input expenses, displays them in a formatted table,                    //
//  calculates the total expense, and saves the data to a file. It handles exceptions for file operations and user input.                                       //                                        
//**************************************************************************************************************************************************************//

import java.io.*;
import java.util.*;


    
public class MyExpense {
    public static void main(String[] args) {
        // Create an array of Item objects to store expenses
        Item[] expenses = new Item[5];
        int count = 0;
        Scanner scanner = new Scanner(System.in);
        
        // Read existing expenses from the file and populates the array while handling exceptions
        try{
            BufferedReader reader = new BufferedReader(new FileReader("expense.txt"));
            String line;
            while((line = reader.readLine()) != null && count < expenses.length) {
                String[] parts = line.trim().split("\t");
                if(parts.length == 2) {
                    double amount = Double.parseDouble(parts[1]);
                    expenses[count++] = new Item(parts[0], amount);
                }
            }
            reader.close();
        } catch (FileNotFoundException e) {                                         // checks if the file exists
            System.out.println("File not found: expense.txt");
            e.printStackTrace();
            scanner.close();
            return; 
        } catch (IOException e) {                                                   // checks for error while reading the file            
            System.out.println("An error occurred while trying to read the file. ");
            e.printStackTrace();
            scanner.close();
            return;
        } 
        
        // Display existing expenses
        displayExpenses(expenses, count);

        // Prompt user for new expenses until they enter '!' to exit
        while (true) {
            System.out.print("Enter item description, followed by ONE [TAB] key and then the amount: (or type '!' to exit): ");
            String input = scanner.nextLine();
            
            // Check if the user wants to exit
            if (input.equals("!")) {
                break;
            }

            // Split the input into description and amount
            String[] parts = input.split("\t");
            if (parts.length != 2) {
                System.out.println("Invalid input format.");
                continue;
            }

            // checks if input is valid and adds it to the array while handling exceptions
            try {
                double amount = Double.parseDouble(parts[1]);
                if (count < expenses.length) {
                    expenses[count++] = new Item(parts[0], amount);
                    System.out.println("Items added: " + parts[0] + " Amount = " + amount);
                } else {
                    System.out.println("Array is full! Cannot add more items.");
                }
                
            } catch (NumberFormatException e) {
                System.out.println("Invalid number! Record skipped.");
            } catch (ArrayIndexOutOfBoundsException e) {
                System.out.println("Array out of bounds! Record skipped.");
            }
        }

        // Display all expenses after user input
        System.out.println(" Updated list of monthly expenses:");
        displayExpenses(expenses, count);

        // Calculate and display the total expense
        double total = 0.0;
        for (int i = 0; i < count; i++) {
            total += expenses[i].getAmount();
        }
        System.out.println("----------------------------------------");
        System.out.printf("Total Expense:\t\t %.2f\n", total);

        // Write the updated expenses back to the file
        try { 
            BufferedWriter writer = new BufferedWriter(new FileWriter("expense.txt"));
            for (int i = 0; i < count; i++) {
                writer.write(expenses[i].getDescription() + "\t" + expenses[i].getAmount() + "\n");
            }
            writer.close();
        } catch (IOException e) {
            System.out.println("An error occurred while writing to the file.\n");
        }
        scanner.close();

    }   

    // Method to display the expenses in a formatted table
    private static void displayExpenses(Item[] items, int count) {
        System.out.println("----------------------------------------");
        System.out.println("Description              Amount");
        System.out.println("----------------------------------------");
        for (int i = 0; i < count; i++) {
            System.out.printf("%-20s %10.2f\n", items[i].getDescription(), items[i].getAmount());
        }
        
    }
    
}