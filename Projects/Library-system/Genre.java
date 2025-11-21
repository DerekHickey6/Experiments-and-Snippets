//**************************************************************************************************************************************************************//
//  Genre.java                                                                                                                                                  //
//                                                                                                                                                              //                                                                                                                                                                                                                                                                                         //
//  Derek Hickey T00777678                                                                                                                                      //
//  May 7, 2025                                                                                                                                                 //
//  COMP 1231 Assignment 2 Question 1                                                                                                                           //
//  This interface defines the genre codes for books                                                                                                            //
//**************************************************************************************************************************************************************//

public interface Genre {
    // Constants for genre codes
    int UNCLASSIFIED = 0;
    int ADVENTURE = 1;
    int MYSTERY = 2;
    int FANTASY = 3;

    // Methods to be implemented by classes that implement this interface
    void setGenreCode(int genreCode);
    int getGenreCode();
}
