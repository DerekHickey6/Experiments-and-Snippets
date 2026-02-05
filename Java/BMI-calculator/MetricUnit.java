//**************************************************************************************************************************************************************//
//  MetricUnit.java                                                                                                                                             //
//                                                                                                                                                              //
//  Derek Hickey T00777678                                                                                                                                      //
//  COMP 1231 Assignment 1 Question 1                                                                                                                           //
//  This class is a subclass of MyBMI that calculates the BMI of a person using metric units (centimeters). It includes methods to calculate BMI, analyze the   //
//  BMI, and display the information of the object.                                                                                                             //                          //
//**************************************************************************************************************************************************************//

public class MetricUnit extends MyBMI{
    private  double heightCM;

    // Default constructor for MetricUnit class
    public MetricUnit() {
        super(); 
        this.heightCM = 0.0;
    }

    // Constructor for MetricUnit class
    public MetricUnit(String name, double weight, double heightCM) {
        super(name, weight);
            this.heightCM = heightCM;
    }

    // Accessor and mutator methods
    public double getHeightCM() {
        return heightCM;
    }
    public void setHeightCM(double heightCM) {
        this.heightCM = heightCM;
    }
    public double getHeightM() {
        return this.heightCM / 100; // Convert height from centimeters to meters
    }

    // Calculate BMI using the formula: (weight in kg / (height in meters)^2)
    @Override
    public double calculateBMI() {
        return (getWeight() / (getHeightM() * getHeightM()));
    }

    // method to give analysis of BMI (underweight, normal weight or overweight)
    @Override
    public String getBMIAnalysis(){
        if (calculateBMI() < 18.5) {
            return "Underweight";
        } else if (calculateBMI() >= 18.5 && calculateBMI() < 24.9) {
            return "Normal weight";
        } else {
            return "Overweight";
        } 
    }

    // displays information of the object - including name, weight, height in meters, BMI and analysis of BMI
    @Override
    public String toString(){
        if(getName() == null || getName().isEmpty()) {
            return "Invalid name. Please enter a valid name and try again.";
        } else if(getWeight() <= 0) {
            return "Invalid weight. Please enter a valid weight and try again.";
        } else if(getHeightM() <= 0) {
            return "Invalid height. Please enter a valid height and try again.";
        } else {
        return super.toString() + "\tHeight: " + String.format("%.2f", getHeightM()) + "m\tBMI: " + String.format("%.2f", calculateBMI()) + "\tResults of analysis: " + getBMIAnalysis();
        }
    }

}
