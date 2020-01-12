package com.company;

import java.util.ArrayList;
import java.util.List;

public class Layer {
    private List<Neuron> neuronList = new ArrayList<>();

    public void generateList(int numOfNeurons, int numOfWeights)
    {
        for(int i =0; i < numOfNeurons; i++)
            neuronList.add(new Neuron(numOfWeights, numOfNeurons));
    }

    public void print()
    {
        for(int i = 0; i < neuronList.size(); i++) {
            System.out.println(i + ": " + neuronList.get(i).getActivation() + ", bias: " + neuronList.get(i).getBias());
        }
    }

    public List<Neuron> getNeuronList() {
        return neuronList;
    }

    public void setNeuronList(List<Neuron> neuronList) {
        this.neuronList = neuronList;
    }
}
