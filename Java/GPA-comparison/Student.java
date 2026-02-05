//**************************************************************************************************************************************************************//
//  Student.java                                                                                                                                                //
//                                                                                                                                                              //
//  Derek Hickey T00777678                                                                                                                                      //
//  May 19, 2025                                                                                                                                                //
//  COMP 1231 Assignment 4 Question 1                                                                                                                           //
//  This class represents a student with a name and cumulative GPA. It implements the Comparable interface to allow for sorting based on GPA.                   //   
//  The class includes methods for getting and setting the student's name and GPA, as well as a toString method for displaying the student's information.       //
//**************************************************************************************************************************************************************//

public class Student implements Comparable<Student> {
    private String name;
    private double cGPA;

    // compareTo method to compare students based on GPA
    @Override
    public int compareTo(Student o) {
        if (this.cGPA >= o.cGPA) {
            return -1;
        } else {
            return 1;
        }
    }

    // parameterized constructor
    public Student(String name, double cGPA) {
        this.name = name;
        this.cGPA = cGPA;
    }

    // accessor for name
    public String getName() {
        return name;
    }

    // mutator for name
    public void setName(String name) {
        this.name = name;
    }

    // accessor for cGPA
    public double getCGPA() {
        return cGPA;
    }

    // mutator for cGPA
    public void setCGPA(double cGPA) {
        this.cGPA = cGPA;
    }

    // toString method to display student information
    @Override
    public String toString(){
        return "Name: " + name +  "\t\tCumulative GPA: " + cGPA;
    }

    
}
