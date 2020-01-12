package com.company;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ImageReader {
    private static File imageFileIDX3;
    private static File labelFileIDX1;
    private static File imageFilet10k;
    private static File labelFilet10k;
    private static FileInputStream fileInputStream;
    private static FileInputStream fileInputStreamIm;
    private static FileInputStream fileInputStreamLa;
    private static List<Integer> imageNumList = new ArrayList<>();
    private static List<Integer> imageNumFillerList = new ArrayList<>();
    private static final int IMAGECOUNT  = 60000;

    public ImageReader()
    {
        try {
            imageFileIDX3 = new File("train-images.idx3-ubyte");
            labelFileIDX1 = new File("train-labels.idx1-ubyte");
            imageFilet10k = new File("t10k-images.idx3-ubyte");
            labelFilet10k = new File("t10k-labels.idx1-ubyte");

            File label;
            if(Main.training)
                label = labelFileIDX1;
            else
                label = labelFilet10k;
            File image;
            if(Main.training)
                image = imageFileIDX3;
            else
                image = imageFilet10k;

            //init rand list
            //gives us a list of randomly sorted ints, from 0 to IMAGECOUNT
            Random rand = new Random();
            for(int i = 0; i < IMAGECOUNT; i++)
                imageNumFillerList.add(i);
            for(int i = 0; i < IMAGECOUNT; i++)
                imageNumList.add(imageNumFillerList.get(rand.nextInt(IMAGECOUNT - i)));

            File fileImages = new File("fileImages");
            File fileLabels = new File("fileLables");
            fileImages.delete();
            fileLabels.delete();
            fileImages.createNewFile();
            fileLabels.createNewFile();

            FileOutputStream fileImOutputStream = new FileOutputStream(fileImages.getAbsolutePath());
            FileOutputStream fileLaOutputStream = new FileOutputStream(fileLabels.getAbsolutePath());

            //randomly organise the images in a new text file
            System.out.println("Randomising images...");
            for(int i = 0; i < IMAGECOUNT; i++)
            {
                fileInputStream = new FileInputStream(image.getAbsolutePath());
                fileInputStream.skip(16 + 784*imageNumList.get(0));
                fileImOutputStream.write(fileInputStream.readNBytes(784));
                fileInputStream.close();

                fileInputStream = new FileInputStream(label.getAbsolutePath());
                fileInputStream.skip(8 + imageNumList.get(0));
                fileLaOutputStream.write(fileInputStream.readNBytes(1));
                fileInputStream.close();

                imageNumList.remove(0);
            }
            fileImOutputStream.close();
            fileLaOutputStream.close();

            fileInputStreamIm = new FileInputStream(fileImages.getAbsolutePath());
            fileInputStreamLa = new FileInputStream(fileLabels.getAbsolutePath());
            System.out.println("Finished randomising images.");

        }catch (Exception ex)
        {
            System.out.println(ex);
        }
    }

    public static byte[] getNextImage()
    {
        try {
            //if at the end of the file
            if(fileInputStreamIm.available() == 0)
                return null;
            //else send next image
            byte[] image;
            image = fileInputStreamIm.readNBytes(784);//read in the next random image
            byte label = fileInputStreamLa.readNBytes(1)[0];
            //make last byte the label
            byte[] packageImLa = new byte[785];
            for(int i = 0; i < 783; i++)
                packageImLa[i]= image[i];
            packageImLa[784] = label;
            return packageImLa;
        }catch (Exception ex)
        {
            System.out.println(ex);
            return null;
        }
    }

    public Boolean hasNextImage()
    {
        try {
            if (fileInputStreamIm.available() == 0)
                return false;
            else
                return true;
        }catch(Exception ex)
        {
            System.out.println(ex);
            return null;
        }
    }
}
