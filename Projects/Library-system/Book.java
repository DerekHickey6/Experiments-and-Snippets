//**************************************************************************************************************************************************************//
//  Book.java                                                                                                                                                 //
//                                                                                                                                                              //
//  Derek Hickey T00777678                                                                                                                                      //
//  May 7, 2025                                                                                                                                                 //
//  COMP 1231 Assignment 2 Question 1                                                                                                                           //
//  This class represents a book with a code, title, price, genre code, and password. It implements the Genre, PasswordLockable & Comparable<> interfaces.      //
//  It provides methods to set and get the book's attributes, lock and unlock the record, and compare books based on price.                                     //
//**************************************************************************************************************************************************************//

public class Book implements Genre, PasswordLockable, Comparable<Book> {
    // instance variables
    private String bookCode;
    private String title;
    private double price;
    private int genreCode;
    private String password;
    private boolean isLocked;

    // default constructor
    public Book() {
        this.bookCode = "Unknown";
        this.title = "Unknown";
        this.price = 0.0;
        this.genreCode = 0; 
        this.password = "default"; 
        this.isLocked = false; 
    }

    // parameterized constructor
    public Book(String bookCode, String title, double price, int genreCode) {
        this.bookCode = bookCode;
        this.title = title;
        this.price = price;
        this.genreCode = genreCode;
        this.password = "default"; // Default password
        this.isLocked = false; // Initially not locked
    }

    // Accessor method for book code
    public String getBookCode() {
        return bookCode;
    }
    // Mutator method for book code
    public void setBookCode(String bookCode) {
        if(!isLocked) {
            this.bookCode = bookCode;
        } else {
            System.out.println("…record is locked …no update performed.");
        }
    }
    // Accessor method for title
    public String getTitle() {
        if(title == null || title.isEmpty()) {
            return "Unknown title";
        }
        return title;
    }
    // Mutator method for title
    public void setTitle(String title) {
        if(!isLocked) {
            this.title = title;
        } else {
            System.out.println("…record is locked …no update performed.");
        }
    }
    // Accessor method for price
    public double getPrice() {
        if(price < 0) {
            return 0.0;
        }
        return price;
    }
    // Mutator method for price
    public void setPrice(double price) {
        if(!isLocked) {
            this.price = price;
        } else {
            System.out.println("…record is locked …no update performed.");
        }
    }

    // Accessor method for password
    public String getPassword() {
        if(password == null || password.isEmpty()) {
            return "No password set";
        }
        return password;
    }

    // Accessor method for genre code
    @Override
    public int getGenreCode() {
        return genreCode;
    }

    // checks if the book record is locked
    @Override   
    public boolean isLocked() {
        return isLocked;
    }

    // compare books based on price
    @Override
    public int compareTo(Book other) {
        if (this.price < other.price) {
            return -1;
        } else if (this.price > other.price) {
            return 1;
        } else {
            return 0;
        }
    }
    
    // Mutator method for genre code
    @Override  
    public void setGenreCode(int genreCode){
        if(!isLocked) {
            this.genreCode = genreCode;
        } else {
            System.out.println("…record is locked …no update performed.");
        }
    }

    // Mutator method for password
    @Override
    public void setPassword(String password) {
        this.password = password;
    }

    // Mutator method for locking the book record
    @Override
    public String lock(String password) {
        if (this.password.equals(password)) {
            isLocked = true;
            return "Book record is now locked.";
        } else {
            return "Incorrect password. Book remains unlocked.";
        }
    }

    // Mutator method for unlocking the book record
    @Override
    public String unlock(String password) {
        if (this.password.equals(password)) {
            isLocked = false;
            return "Book record is now unlocked.";
        } else {
            return "Incorrect password. Book remains locked.";
        }
    }

    // toString method to display book details
    @Override
    public String toString() {
        return "Book Code:" + bookCode + "\tTitle: " + getTitle() + "\tPrice: $" + price + "\tGenre: " + genreCodeToString(genreCode);
    }

    // method to convert genre code to string
    public String genreCodeToString(int genre) {
        switch (genre) {
            case 0:
                return "Unclassified";
            case 1:
                return "Adventure";
            case 2:
                return "Mystery";
            case 3:
                return "Fantasy";
            default:
                return "Unknown Genre";
        }
    }
}
