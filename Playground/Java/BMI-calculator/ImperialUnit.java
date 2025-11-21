//**************************************************************************************************************************************************************//
//  ImperialUnit.java                                                                                                                                           //
//                                                                                                                                                              //
//  Derek Hickey T00777678                                                                                                                                      //
//  COMP 1231 Assignment 1 Question 1                                                                                                                           //
//  This class is a subclass of MyBMI that calculates the BMI of a person using imperial units (feet and inches). It includes methods to calculate BMI, analyze //    
//  the BMI, and display information of the object                                                                                                              //
//**************************************************************************************************************************************************************//

public class ImperialUnit extends MyBMI{
    private  double heightFeet;           // included as per assignment instructions but unneeded
    private  double heightInches;         // included as per assignment instructions but unneeded
    

    //Default constructor for ImperialUnit class
    public ImperialUnit() {
        super(); 
        this.heightFeet = 0.0;
        this.heightInches = 0.0;
    }   

    // Parameterized Constructor for ImperialUnit class
    public ImperialUnit(String name, double weight, double heightFeet, double heightInches) {
        super(name, weight);
            this.heightFeet = heightFeet;
            this.heightInches = heightInches;
    }

    //Accessor and mutator methods
    public double getHeightFeet() {
        return heightFeet;
    }
    public void setHeightFeet(double heightFeet) {
        this.heightFeet = heightFeet;
    }
    public double getHeightInches() {
        return heightInches;
    }
    public void setHeightInches(double heightInches) {
        this.heightInches = heightInches;
    }
    public double getHeightM() {
        return ((this.heightFeet * 12) + this.heightInches) * 0.0254; // Convert height from feet and inches to meters
    }

    // Calculate BMI using the formula: (weight in kg / (height in meters)^2)
    @Override
    public double calculateBMI() {
        return (getWeight() / (getHeightM()*getHeightM()));
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
}
