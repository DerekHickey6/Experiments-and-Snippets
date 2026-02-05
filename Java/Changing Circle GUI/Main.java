//**************************************************************************************************************************************************************//
//  Main.java                                                                                                                                                   //
//                                                                                                                                                              //
//  Derek Hickey T00777678                                                                                                                                      //
//  May 23, 2025                                                                                                                                                //
//  COMP 1231 Assignment 5 Question 1                                                                                                                           //
//  This class is the main class for the Changing Circle application                                                                                            //
//**************************************************************************************************************************************************************//

package application;
	
import javafx.application.Application;
import javafx.stage.Stage;
import javafx.scene.Scene;

// main class for the Changing Circle application
public class Main extends Application {
	@Override
	public void start(Stage primaryStage) {
		try {
			// sets up root and scene with desired background
			ChangingCirclePane root = new ChangingCirclePane();
			root.setStyle("-fx-background-color: lightgoldenrodyellow");
			Scene scene = new Scene(root,400,400);
			
			// sets stage with scene
			primaryStage.setTitle("Changing Circle");
			primaryStage.setScene(scene);
			primaryStage.show();
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	// launches program
	public static void main(String[] args) {
		launch(args);
	}
}
