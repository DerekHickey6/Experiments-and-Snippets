//**************************************************************************************************************************************************************//
//  MyGPA.java                                                                                                                                                  //
//                                                                                                                                                              //
//  Derek Hickey T00777678                                                                                                                                      //
//  May 19, 2025                                                                                                                                                //
//  COMP 1231 Assignment 4 Question 1                                                                                                                           //
//  This class represents a program that organizes a team of students based on their cumulative GPA. It allows the user to add students to the team,            //
//  remove students who do not meet the GPA requirement, and sort the remaining students in descending order of GPA.                                            //  
//  The program also handles cases where no candidates are available for selection.                                                                             //                        
//**************************************************************************************************************************************************************//

import java.util.Iterator;


public class MyGPA {
    public static void main(String[] args) {
        // Create standard student with a GPA of 3.67 to compare against
        Student standard = new Student("Standard", 3.67);

        // Create students with different GPAs
        Student student1 = new Student("Gary", 3.66);
        Student student2 = new Student("Jenny", 4.0);
        Student student3 = new Student("Partick", 3.33);
        Student student4 = new Student("Danny", 3.0);
        Student student5 = new Student("Priscilla", 4.33);
        Student student6 = new Student("Betty", 3.12);
        Student student7 = new Student("June", 3.99);
        Student student8 = new Student("Nick", 3.67);

        // Create a team of students
        Team<Student> studentList = new Team<>();
        studentList.addToTeam(student1);
        studentList.addToTeam(student2);
        studentList.addToTeam(student3);
        studentList.addToTeam(student4);
        studentList.addToTeam(student5);
        studentList.addToTeam(student6);
        studentList.addToTeam(student7);
        studentList.addToTeam(student8);

        System.out.println("The following students want to join the competition team:\n" + studentList);

        // Displays the total number of students in the list
        System.out.println("Total number of students: " + studentList.getNumberOfMembers() + "\n");

        // Iterates through the list of students, displaying and removing those who do not meet the GPA requirement
        System.out.println("Students who do not fulfill cumulative GPA requirement (3.67):");
        Iterator<Student> iterator = studentList.getMembers().iterator();
        while (iterator.hasNext()) {
            Student student = iterator.next();
            if (student.compareTo(standard) != -1) {
                System.out.println(student.toString());
                iterator.remove();
            }
        }

        // Displays the remaining students who meet the GPA requirement
        System.out.println("\nCandidates who have fulfilled cumulative GPA requirement (3.67):");
        if(studentList.hasNoMember()) {
            System.out.println("No candidate has been selected.");
        } else {  
            for (Student student : studentList.getMembers()) {
                System.out.println(student.toString());
            }
        }

        // displays and removes the last student in the list
        System.out.println("\nThe following students who submitted the application last has been removed from the list:");
        if(studentList.hasNoMember()) {
            System.out.println("No candidate in the list.");
        } else {  
            System.out.println("Remove: " + studentList.getMembers().get(studentList.getNumberOfMembers() - 1).toString());
            studentList.removerFromTeam();
        }
        System.out.println();
        

        // Displays the remaining students on the team, sorted in descending order based on their cumulative GPA
        System.out.println("The candidates are sorted in a descending order based on their cumulative GPA:");
        if(studentList.hasNoMember()) {
            System.out.println("No candidate in the list.");
        } else {  
        studentList.getMembers().sort(null);
        for (Student student : studentList.getMembers()) {
            System.out.println(student.toString());
        }
    }

    }
}
