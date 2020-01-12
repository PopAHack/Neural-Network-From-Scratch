package com.company;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Neuron {
    private double activation;
    private double bias;
    private List<Double> weightList;
    private Double maxInitWeight;

    //constructor
    public Neuron(int numOfWeights, int numOfNeurons)
    {
        weightList = new ArrayList<>();
        maxInitWeight = 1.00/numOfNeurons;
        double weightScalar = 5;//scalar for all non-input layers
        if(numOfNeurons == 784)
            weightScalar = 14;//scalar for input layer
        //give default range for weights to all neurons
        for(int i= 0; i < numOfWeights; i++) {
            Double weightTemp;
            //accept no zeros
            while((weightTemp = getGaussian(0, Math.sqrt(maxInitWeight)))==0)
                continue;
            weightList.add(weightTemp * weightScalar);//Get range, convert to full precision int, get rand, convert back + min

        }

        //give a default bias to all neurons
        bias = 0;
    }

    private double getGaussian(double mean, double variance){
        return mean + new Random().nextGaussian() * variance;
    }

    public double getActivation() {
        return activation;
    }

    public void setActivation(double activation) {
        this.activation = activation;
    }

    public List<Double> getWeightList() {
        return weightList;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    private static double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();

        BigDecimal bd = BigDecimal.valueOf(value);
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }
}
