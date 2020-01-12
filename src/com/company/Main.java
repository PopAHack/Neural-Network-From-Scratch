package com.company;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;


public class Main {
    private static Layer inputLayer = new Layer();
    private static Layer hidden1Layer = new Layer();
    private static Layer hidden2Layer = new Layer();
    private static Layer outputLayer = new Layer();
    private static int SUB_SET_NUM = 10;
    private static int IMAGE_REPEAT_NUM = 20;
    private static double accuracy = 0;
    private static double accuracy2 = 0;
    private static int numOfTests = 0;
    public static Boolean training = true;

    public static void main(String[] args) {
        //initialise lists
        int  inNum = 784;
        int h1Num = 16;
        int h2Num = 16;
        int outNum = 10;
        inputLayer.generateList(inNum, h1Num);
        hidden1Layer.generateList(h1Num, h2Num);
        hidden2Layer.generateList(h2Num, outNum);
        outputLayer.generateList(outNum, 0);
        if(!training)
            loadFromFile("First Attempt");
        //Initialize ImageReader

        ImageReader imageReader;
        //here we travel through the images
        byte[] image;
        if(!training)
            IMAGE_REPEAT_NUM = 1;
        for(int l = 0; l < IMAGE_REPEAT_NUM; l++) {
            System.out.println("Shuffles: " + l);
            imageReader = new ImageReader();

            while (imageReader.hasNextImage()) {
                //Run a batch of test cases
                double totalCostOverCases = 0;
                accuracy2 = 0;
                MyMatrix.inLayerMatrices = new ArrayList<>();
                MyMatrix.midLayerMatrices = new ArrayList<>();
                MyMatrix.outLayerMatrices = new ArrayList<>();
                MyMatrix.inBiasMatrices = new ArrayList<>();
                MyMatrix.midBiasMatrices = new ArrayList<>();
                MyMatrix.outBiasMatrices = new ArrayList<>();

                for (int i = 0; i < SUB_SET_NUM; i++) {
                    if(!imageReader.hasNextImage())
                        break;
                    image = imageReader.getNextImage();
                    numOfTests++;
                    //set the input layer
                    for (int j = 0; j < inputLayer.getNeuronList().size(); j++) {
                        //need to map 0-255 range to 0-1 range
                        int imageJ = (image[j] & 0xFF);
                        inputLayer.getNeuronList().get(j).setActivation(round((double) imageJ / 255, 2));
                    }
                    int label = image[784];//get the label from the back of the array

                    //Forward propagation:
                    generateActivation(inputLayer, hidden1Layer);
                    generateActivation(hidden1Layer, hidden2Layer);
                    generateActivation(hidden2Layer, outputLayer);

                    //System Output:
//                System.out.println("Input");
//                inputLayer.print();
//                System.out.println("H1");
//                hidden1Layer.print();
//                System.out.println("H2");
//                hidden2Layer.print();
//                System.out.println("Output");
//                outputLayer.print();
//                System.out.println("Label: " + label);

                    //Cost calculation:
                    totalCostOverCases += calcCost(label);
                    totalCostOverCases = round(totalCostOverCases, 2);

                    //Backward propagation:
                    //aims to minimise cost result
                    List<Double> labelList = new ArrayList<>();
                    for (int j = 0; j < 10; j++) {
                        if (j == label)
                            labelList.add(.5 * (outputLayer.getNeuronList().get(j).getActivation() - 1));
                        else
                            labelList.add(.5 * (outputLayer.getNeuronList().get(j).getActivation()));
                    }

                    //Calculate accuracy
                    int guess = -1;
                    double guessActivation = 0;
                    for (int j = 0; j < outputLayer.getNeuronList().size(); j++) {
                        if (outputLayer.getNeuronList().get(j).getActivation() > guessActivation) {
                            guess = j;
                            guessActivation = outputLayer.getNeuronList().get(j).getActivation();
                        }
                    }
                    if (guess == label) {
                        accuracy++;
                        accuracy2++;
                    }
                    backProp(labelList, hidden2Layer, outputLayer);
                }

                //Now calculate avg weight change for each weight, and apply
//            MyMatrix.outLayerMatrices.get(0).printMatrix();
//            MyMatrix.midLayerMatrices.get(0).printMatrix();
//            MyMatrix.inLayerMatrices.get(0).printMatrix();
//            MyMatrix.outBiasMatrices.get(0).printMatrix();

                MyMatrix outWeightMatrix = avgMatrixValues(MyMatrix.outLayerMatrices);
                MyMatrix midWeightMatrix = avgMatrixValues(MyMatrix.midLayerMatrices);
                MyMatrix inWeightMatrix = avgMatrixValues(MyMatrix.inLayerMatrices);
                MyMatrix outBiasMatrix = avgMatrixValues(MyMatrix.outBiasMatrices);
                MyMatrix midBiasMatrix = avgMatrixValues(MyMatrix.midBiasMatrices);
                MyMatrix inBiasMatrix = avgMatrixValues(MyMatrix.inBiasMatrices);

                if (training)
                    applyChanges(inBiasMatrix, midBiasMatrix, outBiasMatrix, inWeightMatrix, midWeightMatrix, outWeightMatrix);
                //System.out.println("Cost of test case: " + totalCostOverCases);
                //System.out.println("Test Case Accuracy: " + (accuracy2 / SUBSETNUM) * 100 + "%");
                System.out.println("Overall Accuracy: " + round(accuracy / numOfTests, 2) * 100 + "%");
                //return;
            }
        }
        System.out.println("Final Accuracy Over All Cases: " + round(accuracy/numOfTests, 2) * 100 + "%");
        if(training)
            saveToFile();
    }

    private static void loadFromFile(String name)
    {
        try {
            File save = new File(name);
            FileReader fileReader = new FileReader(save.getAbsolutePath());
            BufferedReader reader = new BufferedReader(fileReader);
            if(!save.exists())
                return;

            String[] line;

            //read bias out
            line = reader.readLine().split(",");
            for(int i = 0; i < outputLayer.getNeuronList().size(); i++)
            {
                outputLayer.getNeuronList().get(i).setBias(Double.parseDouble(line[i]));
            }
            //read bias mid
            line = reader.readLine().split(",");
            for(int i = 0; i < hidden2Layer.getNeuronList().size(); i++)
            {
                hidden2Layer.getNeuronList().get(i).setBias(Double.parseDouble(line[i]));
            }
            //read bias in
            line = reader.readLine().split(",");
            for(int i = 0; i < hidden1Layer.getNeuronList().size(); i++)
            {
                hidden1Layer.getNeuronList().get(i).setBias(Double.parseDouble(line[i]));
            }
            //read weights out
            line = reader.readLine().split(",");
            for(int j = 0; j < outputLayer.getNeuronList().size(); j++)
            {
                for(int k = 0; k < hidden2Layer.getNeuronList().size(); k++)
                {
                    hidden2Layer.getNeuronList().get(k).getWeightList().set(j, Double.parseDouble(line[j*hidden2Layer.getNeuronList().size() + k]));
                }
            }
            //read weights mid
            line = reader.readLine().split(",");
            for(int j = 0; j < hidden2Layer.getNeuronList().size(); j++)
            {
                for(int k = 0; k < hidden1Layer.getNeuronList().size(); k++)
                {
                    hidden1Layer.getNeuronList().get(k).getWeightList().set(j, Double.parseDouble(line[j*hidden1Layer.getNeuronList().size() + k]));
                }
            }
            //read weights in
            line = reader.readLine().split(",");
            for(int j = 0; j < hidden1Layer.getNeuronList().size(); j++)
            {
                for(int k = 0; k < inputLayer.getNeuronList().size(); k++)
                {
                    inputLayer.getNeuronList().get(k).getWeightList().set(j, Double.parseDouble(line[j*inputLayer.getNeuronList().size() + k]));
                }
            }
            reader.close();
        }catch (Exception ex)
        {
            System.out.println(ex);
        }
    }

    private static void saveToFile()
    {
        try{
            File save = new File("First Attempt");
            save.delete();
            save.createNewFile();
            FileWriter fileWriter = new FileWriter(save.getAbsolutePath());
            //write bias out
            for(int i = 0; i < outputLayer.getNeuronList().size(); i++)
            {
                fileWriter.write(outputLayer.getNeuronList().get(i).getBias() + ",");
            }
            fileWriter.write('\n');
            //write bias mid
            for(int i = 0; i < hidden2Layer.getNeuronList().size(); i++)
            {
                fileWriter.write(hidden2Layer.getNeuronList().get(i).getBias() + ",");
            }
            fileWriter.write('\n');
            //write bias in
            for(int i = 0; i < hidden1Layer.getNeuronList().size(); i++)
            {
                fileWriter.write(hidden1Layer.getNeuronList().get(i).getBias() + ",");
            }
            fileWriter.write('\n');
            //write weights out
            for(int j = 0; j < outputLayer.getNeuronList().size(); j++)
            {
                for(int k = 0; k < hidden2Layer.getNeuronList().size(); k++)
                {
                    fileWriter.write(hidden2Layer.getNeuronList().get(k).getWeightList().get(j) + ",");
                }
            }
            fileWriter.write('\n');
            //write weights mid
            for(int j = 0; j < hidden2Layer.getNeuronList().size(); j++)
            {
                for(int k = 0; k < hidden1Layer.getNeuronList().size(); k++)
                {
                    fileWriter.write(hidden1Layer.getNeuronList().get(k).getWeightList().get(j) + ",");
                }
            }
            fileWriter.write('\n');
            //write weights in
            for(int j = 0; j < hidden1Layer.getNeuronList().size(); j++)
            {
                for(int k = 0; k < inputLayer.getNeuronList().size(); k++)
                {
                    fileWriter.write(inputLayer.getNeuronList().get(k).getWeightList().get(j) + ",");
                }
            }
            fileWriter.close();
        }catch (Exception ex)
        {
            System.out.println(ex);
        }
    }

    //Takes an activated input list and generates the activation of neurons in the output list.
    private static void generateActivation(Layer input, Layer output)
    {
        //calculate the activation of each node in the output layer
        //iterate through each node in the output layer
        for(int i = 0; i < output.getNeuronList().size(); i++)
        {
            //collect the sum to place in the Sigmoid function:
            //sum the multiplication of the weight and activation, adding the bias, of each node towards the i'th node of the output layer.
            Double z = 0.00;
            for(int j = 0; j < input.getNeuronList().size(); j++)
            {
                z+= input.getNeuronList().get(j).getActivation() * input.getNeuronList().get(j).getWeightList().get(i);
            }
            //add bias
            z += output.getNeuronList().get(i).getBias();

            //now set the new activation of this neuron
            output.getNeuronList().get(i).setActivation(round(sigmoidFn(z),2));
        }
    }


    private static void applyChanges(MyMatrix biasIn, MyMatrix biasMid, MyMatrix biasOut, MyMatrix weightIn, MyMatrix weightMid, MyMatrix weightOut)
    {
        double learningRate = 1;

        //apply changes to biases
        for(int i = 0; i < outputLayer.getNeuronList().size(); i++)
        {
            outputLayer.getNeuronList().get(i).setBias(outputLayer.getNeuronList().get(i).getBias() - learningRate * biasOut.getElement(i,0));
        }
        for(int i = 0; i < hidden2Layer.getNeuronList().size(); i++)
        {
            hidden2Layer.getNeuronList().get(i).setBias(hidden2Layer.getNeuronList().get(i).getBias() - learningRate * biasMid.getElement(i,0));
        }
        for(int i = 0; i < hidden1Layer.getNeuronList().size(); i++)
        {
            hidden1Layer.getNeuronList().get(i).setBias(hidden1Layer.getNeuronList().get(i).getBias() - learningRate * biasIn.getElement(i,0));
        }

        //apply changes to weights
        for(int j = 0; j < outputLayer.getNeuronList().size(); j++)//columns
        {
            for(int k = 0; k < hidden2Layer.getNeuronList().size(); k++)//rows
            {
                hidden2Layer.getNeuronList().get(k).getWeightList().set(j, hidden2Layer.getNeuronList().get(k).getWeightList().get(j) - learningRate * weightOut.getElement(j,k));
            }
        }

        //apply changes to weights
        for(int j = 0; j < hidden2Layer.getNeuronList().size(); j++)//columns
        {
            for(int k = 0; k < hidden1Layer.getNeuronList().size(); k++)//rows
            {
                hidden1Layer.getNeuronList().get(k).getWeightList().set(j, hidden1Layer.getNeuronList().get(k).getWeightList().get(j) - learningRate * weightMid.getElement(j,k));
            }
        }

        //apply changes to weights
        for(int j = 0; j < hidden1Layer.getNeuronList().size(); j++)//columns
        {
            for(int k = 0; k< inputLayer.getNeuronList().size(); k++)//rows
            {
                inputLayer.getNeuronList().get(k).getWeightList().set(j, inputLayer.getNeuronList().get(k).getWeightList().get(j) - learningRate * weightIn.getElement(j,k));
            }
        }
    }

    private static MyMatrix avgMatrixValues(List<MyMatrix> inputMatrixList)
    {
        MyMatrix matrixAvg = new MyMatrix();
        for(int i = 0; i < SUB_SET_NUM; i++)
        {
            for(int j = 0; j < inputMatrixList.get(i).columnSize(); j++)//ith matrix, column
            {
                for(int k = 0; k < inputMatrixList.get(i).rowSize(); k++)//ith matrix, row
                {
                    matrixAvg.addElement(j, k, inputMatrixList.get(i).getElement(j,k));
                }
            }
        }
        for(int j = 0; j < matrixAvg.columnSize(); j++)//ith matrix, column
        {
            for(int k = 0; k < matrixAvg.rowSize(); k++)//ith matrix, row
            {
                matrixAvg.setElement(j,k, matrixAvg.getElement(j,k)/ SUB_SET_NUM);
            }
        }
        return matrixAvg;
    }

    private static void backProp(List<Double> labelList, Layer leftLayer, Layer rightLayer)
    {
        MyMatrix weightMatrix = new MyMatrix();
        MyMatrix biasMatrix = new MyMatrix();
        //calculate the required weight adjustment for each weight between the two layers
        for(int j = 0; j < rightLayer.getNeuronList().size(); j++)
        {
            for(int k = 0; k < leftLayer.getNeuronList().size(); k++)
            {
                Double weightChange = leftLayer.getNeuronList().get(k).getActivation();//activation
                weightChange = weightChange * (rightLayer.getNeuronList().get(j).getActivation() * (1.00 - rightLayer.getNeuronList().get(j).getActivation()));//sigmoid derivative
                weightChange = weightChange * labelList.get(j);//cost
                weightMatrix.setElement(j, k, weightChange);
            }
        }

        //calculate the bias adjustment for each neuron.
        for(int j =0; j < rightLayer.getNeuronList().size(); j++)
        {
            Double biasChange = 1.00;//activation
            biasChange = biasChange * (rightLayer.getNeuronList().get(j).getActivation() * (1.00 - rightLayer.getNeuronList().get(j).getActivation()));//sigmoid function derivative
            biasChange = biasChange * labelList.get(j);//cost
            biasMatrix.setElement(j, 0, biasChange);
        }

        //sum cost for each neuron.
        List<Double> newLabelList = new ArrayList<>();
        for(int r = 0; r < leftLayer.getNeuronList().size(); r++)//rows
        {
            newLabelList.add(0.00);
            for(int c = 0; c < rightLayer.getNeuronList().size(); c++)//columns
            {
                newLabelList.set(r, newLabelList.get(r) + weightMatrix.getElement(c,r));
            }
        }

        //if this is the first time its been run (Yuck Recursion)
        if(rightLayer.equals(outputLayer)) {
            MyMatrix.outLayerMatrices.add(weightMatrix);
            MyMatrix.outBiasMatrices.add(biasMatrix);
            backProp(newLabelList, hidden1Layer, hidden2Layer);
        }
        //second time
        if(rightLayer.equals(hidden2Layer)) {
            MyMatrix.midLayerMatrices.add(weightMatrix);
            MyMatrix.midBiasMatrices.add(biasMatrix);
            backProp(newLabelList, inputLayer, hidden1Layer);
        }
        //third time
        if(rightLayer.equals(hidden1Layer)) {
            MyMatrix.inLayerMatrices.add(weightMatrix);
            MyMatrix.inBiasMatrices.add(biasMatrix);
        }
    }

    //returns the output from the Sigmoid function
    private static Double sigmoidFn(double input)
    {
        double value = 1.00 / (1.00 + Math.exp(-input)); //Sigmoid function
        return value;
    }

    private static Double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();

        BigDecimal bd = BigDecimal.valueOf(value);
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }

    //calculates and returns the cost of a single image
    private static Double calcCost(int label){
        Double total = 0.00;
        for(int i = 0; i < 10; i++)
        {
            if(i == label)
                total += Math.pow((outputLayer.getNeuronList().get(i).getActivation() - 1.00), 2);//raise to the power of 2
            else
                total += Math.pow((outputLayer.getNeuronList().get(i).getActivation()), 2);//raise to the power of 2
        }
        return round(total,2);
    }
}
