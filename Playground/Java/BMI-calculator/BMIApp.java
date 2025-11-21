//**************************************************************************************************************************************************************//
//  BMIApp.java                                                                                                                                                 //
//                                                                                                                                                              //
//  Derek Hickey T00777678                                                                                                                                      //
//  COMP 1231 Assignment 1 Question 1                                                                                                                           //
//  This application calculates the BMI of a person using either metric or imperial units using BMI = (weight(kg) / height(m)^2). The application determines if //
//  the person is underweight, normal weight or overweight based on the BMI value and displays the results of the analysis for all individuals in the program.  //
//**************************************************************************************************************************************************************//

public class BMIApp {
    public static void main(String[] args) {

        MyBMI[] bmiList = new MyBMI[6];

        //Creating objects of the MyBMI class and storing them in the bmiList array
        
        bmiList[0] = new ImperialUnit("Derek", 77.3, 6, 4);         // Derek has a weight of 77.3kg and is 6 feet 4 inches tall
        bmiList[1] = new ImperialUnit("Dryden", 58.2, 5, 11);       // Dryden has a weight of 58.2kg and is 5 feet 11 inches tall
        bmiList[2] = new ImperialUnit("Jamie", 88.8, 5, 5);         // Jamie has a weight of 88.8kg and is 5 feet 5 inches tall
        bmiList[3] = new MetricUnit("Jen", 66.3, 173);              // Jen has a weight of 66.3kg and is 173cm tall
        bmiList[4] = new MetricUnit("Anna", 72.2, 162);             // Anna has a weight of 72.2kg and is 162cm tall
        bmiList[5] = new MetricUnit("Hailey", 59.9, -188);          // Hailey has a weight of 59.9kg and is 188cm tall

        //Loop traversing that bmiList array and prints results of each object
        for(MyBMI bmi : bmiList){
            System.out.println(bmi);
        }
    }
}
