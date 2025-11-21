//**************************************************************************************************************************************************************//
//  Team.java                                                                                                                                                   //
//                                                                                                                                                              //
//  Derek Hickey T00777678                                                                                                                                      //
//  May 19, 2025                                                                                                                                                //
//  COMP 1231 Assignment 4 Question 1                                                                                                                           //
//  This class represents a team of members, allowing for adding, removing, and managing team members. It provides methods to add and remove members,           //
//  check the number of members, and retrieve the list of members. The class uses generics to allow for different types of team members.                        //                                                                          
//**************************************************************************************************************************************************************//

import java.util.ArrayList;

public class Team<T> {
    private ArrayList<T> aList;

    // default constructor
    public Team() {
        aList = new ArrayList<>();
    }

    // method to add a member to the team
    public void addToTeam(T member) {
        aList.add(member);
    }

    // method to remove the last member from the team
    public void removerFromTeam() {
        aList.remove(aList.size() - 1);
    }

    // method to remove a specific member from the team
    public void removeFromTeam(T member) {
        aList.remove(member);
    }

    // method to get the number of members in the team
    public int getNumberOfMembers() {
        return aList.size();
    }

    // method to check if the team has no members
    public boolean hasNoMember() {
        return aList.isEmpty();
    }

    // method to get the list of members
    public ArrayList<T> getMembers() {
        return aList;
    }

    // method to display the team members
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (T member : aList) {
            sb.append(member.toString()).append("\n");
        }
        return sb.toString();
    }
}
