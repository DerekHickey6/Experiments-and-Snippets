//**********************************************************************************************************//
//  Item.java                                                                                               //
//                                                                                                          //
//  Derek Hickey T00777678                                                                                  //
//  May 19, 2025                                                                                            //
//  COMP 1231 Assignment 3 Question 1                                                                       //
//  This class represents an item with a description and amount.                                            //
//  It provides constructors to create an item, and accessor methods to retrieve the item's properties.     //                                                                         
//**********************************************************************************************************//

public class Item {
    //instance variables
    private String description;
    private double amount;

    // default constructor
    public Item() {
        this.description = "";
        this.amount = 0.0;
    }
    // parameterized constructor
    public Item(String description, double amount) {
        this.description = description;
        this.amount = amount;
    }

    //accessor method for description
    public String getDescription() {
        return description;
    }

    //accessor method for amount
    public double getAmount() {
        return amount;
    }
}
