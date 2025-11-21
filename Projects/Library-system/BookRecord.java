//**************************************************************************************************************************************************************//
//  BookRecord.java                                                                                                                                             //
//                                                                                                                                                              //
//  Derek Hickey T00777678                                                                                                                                      //
//  COMP 1231 Assignment 2 Question 1                                                                                                                           //
//  This application creates 5 book records and is designed to be a user-friendly menu to allow the user to set a password, lock/unlock records,                //
//  change details, display records, and sort by price. It uses the Book class.                                                                                 //
//**************************************************************************************************************************************************************//

import java.util.Scanner;

public class BookRecord {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        // Create an array of 5 Book objects (records)
        Book[] books = new Book[5];
        books[0] = new Book("R101", "The Hobbit", 10.99, Genre.ADVENTURE);
        books[1] = new Book("T342", "Indian Jones", 15.99, Genre.MYSTERY);
        books[2] = new Book("P290", "Harry Potter", 20.99, Genre.FANTASY);
        books[3] = new Book("B993", "The Great Gatsby", 12.99, Genre.ADVENTURE);
        books[4] = new Book("G789", "", 9.99, Genre.MYSTERY);

        System.out.println("Creating 5 book records:");
        for (Book book : books) {
            System.out.println(book.toString());
        }
        
        // Display the menu to the user
        menu(books);
        scanner.close();
    }

    public static void menu(Book[] books) {
        Scanner scanner = new Scanner(System.in);
        // Main menu for user interaction
        System.out.println("\nWhat would you like to do?");
        System.out.println("1. Set password for book records");
        System.out.println("2. Lock book records");
        System.out.println("3. Unlock book records");
        System.out.println("4. Change details of a book record");
        System.out.println("5. Display book records");
        System.out.println("6. List books by price");
        System.out.println("7. Exit");

        int choice = scanner.nextInt();
        
        // Handles user choice
        switch (choice) {
            // checks to see if there is a password set, if not, it sets new password. if there is, checks current password and before setting new password
            case 1:
                if (books[0].getPassword().equals("default")) {
                    System.out.println("There is no password set. Please set a new password.");
                    String newPassword = scanner.next();
                    for(Book book : books) {
                        book.setPassword(newPassword);
                    }
                    System.out.println("Password updated successfully.");
                } else {
                    System.out.println("Please enter current password:");
                    String currentPassword = scanner.next();
                    if(currentPassword.equals(books[0].getPassword())){
                        System.out.println("Enter new password:");
                        String newPassword = scanner.next();
                        for(Book book : books) {
                            book.setPassword(newPassword);
                        }
                        System.out.println("Password updated successfully.");
                    } else {
                        System.out.println("Incorrect password. Password not updated.");
                    }
                }   menu(books);
                break;
            // checks current password and locks records if correct. if incorrect, notifies user and does not lock records   
            case 2:
                {
                    System.out.println("Enter password to lock the book records:");
                    String password = scanner.next();
                    if(password.equals(books[0].getPassword())){
                        System.out.println("Password is correct. Locking book records...");
                        for(Book book : books) {
                            book.lock(password);
                        }
                        System.out.println("Book records locked successfully.");
                    } else {
                        System.out.println("Incorrect password. Book records remain unlocked.");
                    }       menu(books);
                    break;
                }
            // checks current password and unlocks records if correct. if incorrect, notifies user and does not unlock records
            case 3:
                {
                    System.out.println("Enter password to unlock the book records:");
                    String password = scanner.next();
                    if(password.equals(books[0].getPassword())){
                        System.out.println("Password is correct. Unlocking book records...");
                        for(Book book : books) {
                            book.unlock(password);
                        }
                        System.out.println("Book records unlocked successfully.");
                    } else {
                        System.out.println("Incorrect password. Book records remain locked.");
                    }       menu(books);
                    break;
                }
            // allows user to change book record details. if record is locked, notifies user and does not allow changes
            case 4:
                {
                    System.out.println("Which book record would you like to change? (1-5)");
                    int bookIndex = scanner.nextInt() - 1;
                    scanner.nextLine(); // Consume newline left-over
                    System.out.println("What would you like to change?");
                    System.out.println("1. Title");
                    System.out.println("2. Price");
                    System.out.println("3. Genre code");
                    System.out.println("4. All of the above");
                    int changeChoice = scanner.nextInt();
                    scanner.nextLine(); // Consume newline left-over
                    if(books[bookIndex].isLocked()){
                        System.out.println("Book record is locked. Cannot change details.");
                    } else {
                        switch (changeChoice) {
                            case 1:
                                {
                                    System.out.println("Enter new title:");
                                    String newTitle = scanner.nextLine();
                                    books[bookIndex].setTitle(newTitle);
                                    break;
                                }
                            case 2:
                                {
                                    System.out.println("Enter new price:");
                                    double newPrice = scanner.nextDouble();
                                    books[bookIndex].setPrice(newPrice);
                                    break;
                                }
                            case 3:
                                {
                                    System.out.println("Enter new genre code \n0 - Unclassified \n1 - Adventure \n2 - Mystery \n3 - Fantasy");
                                    int newGenreCode = scanner.nextInt();
                                    books[bookIndex].setGenreCode(newGenreCode);
                                    break;
                                }
                            case 4:
                                {
                                    System.out.println("Enter new title:");
                                    String newTitle = scanner.nextLine();
                                    books[bookIndex].setTitle(newTitle);
                                    System.out.println("Enter new price:");
                                    double newPrice = scanner.nextDouble();
                                    books[bookIndex].setPrice(newPrice);
                                    System.out.println("Enter new genre code \n0 - Unclassified \n1 - Adventure \n2 - Mystery \n3 - Fantasy");
                                    int newGenreCode = scanner.nextInt();
                                    books[bookIndex].setGenreCode(newGenreCode);
                                    break;
                                }
                            default:
                                break;
                        }
                        System.out.println("Book record updated successfully.");
                        System.out.println(books[bookIndex].toString());
                    }   
                    menu(books);
                    break;
                }
            // displays book records to user. regardles of locked status
            case 5:
                {
                    System.out.println("Which book record would you like to view? (1-5)");
                    int bookIndex = scanner.nextInt() - 1;
                    switch (bookIndex) {
                        case 0:
                            System.out.print("First book record: ");
                            break;
                        case 1:
                            System.out.print("Second book record: ");
                            break;
                        case 2:
                            System.out.print("Third book record: ");
                            break;
                        case 3:
                            System.out.print("Fourth book record: ");
                            break;
                        case 4:
                            System.out.print("Fifth book record: ");
                            break;
                        default:
                            break;
                    }
                    System.out.println(books[bookIndex].toString());
                    menu(books);
                    break;
                }
            // sorts book records by price in descending order and displays sorted records to user
            case 6:
                System.out.println("Books sorted by price:");
                for (int i = 0; i < books.length - 1; i++) {
                    for (int j = 0; j < books.length - i - 1; j++) {
                        if (books[j].compareTo(books[j + 1]) < 0) {
                            Book temp = books[j];
                            books[j] = books[j + 1];
                            books[j + 1] = temp;
                        }
                    }
                }   System.out.println("Sorted books:");
                for (Book book : books) {
                    System.out.println(book.toString());
                }   menu(books);
                break;
            // exits program
            case 7:
                System.out.println("Exiting program.");
                scanner.close();
                return;
            default:
                System.out.println("Invalid choice. Please try again.");
                menu(books);
                break;
        }
}
}

