package Assignment2;

//**************************************************************************************************************************************************************//
//  PasswordLockable.java                                                                                                                                       //
//                                                                                                                                                              //
//  Derek Hickey T00777678                                                                                                                                      //
//  May 7, 2025                                                                                                                                                 //
//  COMP 1231 Assignment 2 Question 1                                                                                                                           //
//  This interface defines the methods for locking and unlocking a record with a password. It is used to secure sensitive information in the Book class.        //
//**************************************************************************************************************************************************************//

public interface PasswordLockable {
    
    // methods to be implemented by classes that implement this interface
    void setPassword(String password);
    String lock(String password);
    String unlock(String password);
    boolean isLocked();
}
