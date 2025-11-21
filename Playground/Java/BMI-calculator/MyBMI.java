//**************************************************************************************************************************************************************//
//  MyBMI.java                                                                                                                                                  //
//                                                                                                                                                              //
//  Derek Hickey T00777678                                                                                                                                      //
//  COMP 1231 Assignment 1 Question 1                                                                                                                           //
//  This class is an abstract class that serves as a base for the MetricUnit and ImperialUnit classes. It contains the instance fields and methods for          //
//  both types of units.                                                                                                                                        //
//**************************************************************************************************************************************************************//

abstract class MyBMI {
    // Attributes
    private String name;
    private double weight;
    
    // abstract method to be implemented in subclasses to calculate BMI
    public abstract double calculateBMI();
    public abstract String getBMIAnalysis();

    //Default Constructor
    public MyBMI() {
        this.name = "Unknown";
        this.weight = 0.0;
    }
    // Parameterized Constructor
    public MyBMI(String name, double weight) {
        if(name == null || name.isEmpty()) {
            this.name = "Unknown";
        } else {
            this.name = name;
        }
        if(weight < 0) {
            this.weight = 0.0;
        } else {
            this.weight = weight;
        }
    }
    // accessors and mutator methods
    public String getName(){
        return name;
    }
    public double getWeight(){
        return weight;
    }
    public void setWeight(double weight){
        this.weight = weight;
    }
    public void setName(String name){
        this.name = name;
    }
    
    // Displays users name and weight
    @Override
    public String toString(){
        return "Name: " + name + " \tWeight: " + String.format("%.2f", weight) + "kg";
    }
    

}
