//**************************************************************************************************************************************************************//
//  ChangingCirclePane.java                                                                                                                                     //
//                                                                                                                                                              //
//  Derek Hickey T00777678                                                                                                                                      //
//  May 23, 2025                                                                                                                                                //
//  COMP 1231 Assignment 5 Question 1                                                                                                                           //
//  This class creates a pane with a circle that changes color and size based on user input. The user can select a color using radio buttons and adjust the     //
//  size using a slider. A warning sound plays if the user clicks outside the allowed controls. The circle's radius is bound to the slider value,               //
//  and the pane is designed with appropriate layout and styling.                                                                                               //                                                                                                 //
//**************************************************************************************************************************************************************//

package application;

import javafx.event.ActionEvent;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.RadioButton;
import javafx.scene.control.Slider;
import javafx.scene.control.ToggleGroup;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.VBox;
import javafx.scene.media.AudioClip;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.text.Text;

public class ChangingCirclePane extends BorderPane {
    private RadioButton rbOrange, rbYellow, rbBlue;
    private Text instruction1, instruction2, warning1, warning2;
    private Slider slider;
    private Circle circle;
    private AudioClip warningSound;

    public ChangingCirclePane() {
    	
        // Load warning sound
    	warningSound = new AudioClip(getClass().getResource("pin.wav").toExternalForm());

        // Play warning sound if user clicks outside allowed controls
        setOnMouseClicked(event -> {
            Node clickedNode = (Node) event.getTarget();
            if (!isClickInsideControl(clickedNode)) {
                warningSound.stop();
                warningSound.play();
            }
        });

        // Top - instructions
        instruction1 = new Text("Change the circle color using the radio buttons.");
        instruction1.setStyle("-fx-font-weight: bold;");
        instruction2 = new Text("Change the scale of the circle between 0-100% using the slider.");
        instruction2.setStyle("-fx-font-weight: bold;");
        VBox instructionsBox = new VBox(5, instruction1, instruction2);
        instructionsBox.setAlignment(Pos.CENTER);
        instructionsBox.setPadding(new Insets(30));
        setTop(instructionsBox);

        // Center - circle
        circle = new Circle(25, Color.ORANGE);
        VBox centerBox = new VBox(circle);
        centerBox.setAlignment(Pos.CENTER);
        centerBox.setPadding(new Insets(0, 60, 30, 0));
        centerBox.setMaxHeight(80);
        setCenter(centerBox);

        // Left - radio buttons
        ToggleGroup group = new ToggleGroup();
        rbOrange = createColorRadioButton("Orange", Color.ORANGE, group);
        rbOrange.setSelected(true);
        rbYellow = createColorRadioButton("Yellow", Color.YELLOW, group);
        rbBlue = createColorRadioButton("Blue", Color.BLUE, group);
        VBox colorOptions = new VBox(10, rbOrange, rbYellow, rbBlue);
        colorOptions.setAlignment(Pos.CENTER_LEFT);
        colorOptions.setPadding(new Insets(0, 20, 30, 80));
        setLeft(colorOptions);

        // Slider setup - controls circle radius
        slider = new Slider(0, 100, 25);
        slider.setShowTickMarks(true);
        slider.setShowTickLabels(true);
        slider.setMaxWidth(340);
        centerBox.heightProperty().addListener((obs, oldVal, newVal) -> {
            circle.radiusProperty().bind(slider.valueProperty().multiply(0.7));
        });

        // Warnings setup
        warning1 = new Text("Select the radio buttons or the slider only.");
        warning2 = new Text("You'll hear a warning sound if the mouse is clicked elsewhere!");
        warning1.setFill(Color.RED);
        warning2.setFill(Color.RED);
        VBox warningBox = new VBox(5, warning1, warning2);
        warningBox.setAlignment(Pos.CENTER);

        // Bottom - slider on top, warnings below
        VBox sliderAndWarningBox = new VBox(10, slider, warningBox);
        sliderAndWarningBox.setAlignment(Pos.CENTER);
        sliderAndWarningBox.setPadding(new Insets(0, 0, 35, 0));
        sliderAndWarningBox.setSpacing(20);
        setBottom(sliderAndWarningBox);
    }
    
    // creates radio button
    private RadioButton createColorRadioButton(String label, Color color, ToggleGroup group) {
        RadioButton rb = new RadioButton(label);
        rb.setToggleGroup(group);
        rb.setUserData(color);
        rb.setOnAction(this::processRadioButton);
        return rb;
    }
    
    // event handler for radio buttons
    private void processRadioButton(ActionEvent event) {
        RadioButton selected = (RadioButton) ((ToggleGroup) ((RadioButton) event.getSource()).getToggleGroup()).getSelectedToggle();
        circle.setFill((Color) selected.getUserData());
    }
    
    // checks to see if clicked buttons or slider was clicked on
    private boolean isClickInsideControl(Node node) {
        while (node != null) {
            if (node == rbOrange || node == rbYellow || node == rbBlue || node == slider) {
                return true;
            }
            node = node.getParent();
        }
        return false;
    }
}