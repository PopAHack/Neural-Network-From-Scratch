package com.company;

import java.util.ArrayList;
import java.util.List;

public class MyMatrix {
    //WARNING: NXM MATRICES ONLY (RECTANGULAR)
    private List<List<Double>> columnsList = new ArrayList<>();

    public static List<MyMatrix> outLayerMatrices;
    public static List<MyMatrix> midLayerMatrices;
    public static List<MyMatrix> inLayerMatrices;
    public static List<MyMatrix> inBiasMatrices;
    public static List<MyMatrix> midBiasMatrices;
    public static List<MyMatrix> outBiasMatrices;

    public void printMatrix()
    {
        for(int i = 0; i < columnsList.size(); i++)//columns
        {
            String row = "";
            for(int j = 0; j < columnsList.get(i).size(); j++)//rows
            {
                row += getElement(i, j) + ", ";
            }
            System.out.println(row + '\n');
        }
        System.out.println("Number of columns: " + columnSize());
        System.out.println("Number of rows: " + rowSize());
    }

    //tested and works
    public void setElement(int column, int row, Double value)
    {
        //if that column doesn't exist, populate all columns up to that column
        if(columnsList.size() < column + 1)
            for(int i = columnsList.size(); i < column + 1; i++)
                columnsList.add(new ArrayList<>());

        //if that column doesn't exist, populate all rows up to that row
        if(columnsList.get(column).size() < row + 1)
            for(int i = columnsList.get(column).size(); i < row + 1; i++)
                columnsList.get(column).add(0.00);

        //add element to correct place
        columnsList.get(column).set(row, value);
    }

    public void addElement(int column, int row, Double value)
    {
        //if it doesn't exist, set it to the value
        if(getElement(column,row) == null)
            setElement(column, row, value);
        else
            setElement(column, row, getElement(column, row) + value);
    }

    //tested and works
    public Double getElement(int column, int row)
    {
        //check if there are enough columns
        if(columnsList.size() < column + 1)
            return null;
        //check if there are enough rows
        if(columnsList.get(column).size() < row +1)
            return null;
        //return value
        return columnsList.get(column).get(row);
    }

    public boolean isEmpty(){
        return columnsList.isEmpty();
    }

    public int columnSize()
    {
        if(isEmpty())
            return 0;
        else
            return columnsList.size();
    }

    public int rowSize()
    {
        if(isEmpty())
            return 0;
        else
            return columnsList.get(columnsList.size() -1).size();
    }
}
