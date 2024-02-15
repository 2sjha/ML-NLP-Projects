
/***
 * Author: Vibhav Gogate
 * The University of Texas at Dallas
 ***/

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import javax.imageio.ImageIO;

public class KMeans {

    private static int MAX_ITERATIONS = 50;

    public static void main(String[] args) {
        if (args.length < 3) {
            System.out.println(
                    "Usage: Kmeans <input-image> <k> <output-image> <optional-p>");
            return;
        }
        try {
            BufferedImage originalImage = ImageIO.read(new File(args[0]));
            int k = Integer.parseInt(args[1]);
            // p = minkowski distance of order p, by default we use p = 1 (Manhattan
            // distance)
            // p is an optional paramter
            int p = args.length == 4 ? Integer.parseInt(args[3]) : 1;
            BufferedImage kmeansJpg = kmeans_helper(originalImage, k, p);
            ImageIO.write(kmeansJpg, "jpg", new File(args[2]));
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }
    }

    private static BufferedImage kmeans_helper(
            BufferedImage originalImage,
            int k,
            int p) {
        int w = originalImage.getWidth();
        int h = originalImage.getHeight();
        BufferedImage kmeansImage = new BufferedImage(
                w,
                h,
                originalImage.getType());
        Graphics2D g = kmeansImage.createGraphics();
        g.drawImage(originalImage, 0, 0, w, h, null);
        // Read rgb values from the image
        int[] rgb = new int[w * h];
        int count = 0;
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                rgb[count++] = kmeansImage.getRGB(i, j);
            }
        }
        // Call kmeans algorithm: update the rgb values
        kmeans(rgb, k, p);

        // Write the new rgb values to the image
        count = 0;
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                kmeansImage.setRGB(i, j, rgb[count++]);
            }
        }
        return kmeansImage;
    }

    // Your k-means code goes here
    // Update the array rgb by assigning each entry in the rgb array to its cluster
    // center
    private static void kmeans(int[] rgb, int k, int p) {
        int rgbLen = rgb.length;
        // Clusters is a map of cluster center's rgb value and cluster members' indices
        // in rgb array
        Map<Integer, Set<Integer>> clusters = new HashMap<>();
        // cluster centers in the previous iteration to check for convergence
        Set<Integer> prevClusterCenters = new HashSet<>();

        // Initially select k cluster centers randomly
        Random random = new Random();
        for (int i = 0; i < k; ++i) {
            int r = random.nextInt(rgbLen);
            // Make sure that randomly selected k clusters are not duplicates
            while (clusters.containsKey(rgb[r])) {
                r = random.nextInt(rgbLen);
            }
            clusters.put(rgb[r], new HashSet<>());
        }

        // Run k-means clustering for MAX_ITERATIONS or until it converges
        for (int i = 0; i < MAX_ITERATIONS; ++i) {
            // Assign pixels to their closest cluster centers
            for (int j = 0; j < rgbLen; ++j) {
                int closestClusterCenter = getClosestClusterCenter(
                        clusters.keySet(),
                        rgb[j],
                        p);

                if (!clusters.get(closestClusterCenter).contains(j)) {
                    clusters.get(closestClusterCenter).add(j);
                }
            }

            // Average all the pixels in a cluster and find new cluster center
            Map<Integer, Set<Integer>> newClusters = new HashMap<>();
            for (Map.Entry<Integer, Set<Integer>> entry : clusters.entrySet()) {
                int avgRgb = calculateAverageRgbVal(entry.getValue(), rgb);

                if (newClusters.containsKey(avgRgb)) {
                    newClusters.get(avgRgb).addAll(entry.getValue());
                } else {
                    newClusters.put(avgRgb, entry.getValue());
                }
            }
            clusters = newClusters;

            // Check if cluster centers have converged
            if (prevClusterCenters.equals(clusters.keySet())) {
                break;
            } else {
                prevClusterCenters = clusters.keySet();
            }
        }

        // Finally update rgb array with cluster center values
        for (Map.Entry<Integer, Set<Integer>> entry : clusters.entrySet()) {
            for (Integer rgbIndex : entry.getValue()) {
                rgb[rgbIndex] = entry.getKey();
            }
        }
    }

    /*
     * Calculates average of ARGB components from a list of RGB itegers
     */
    private static int calculateAverageRgbVal(Set<Integer> rgbIndices, int[] rgb) {
        int len = rgbIndices.size();
        int sumA = 0;
        int sumR = 0;
        int sumG = 0;
        int sumB = 0;

        for (Integer rgbIdx : rgbIndices) {
            Rgb rgbComps = new Rgb(rgb[rgbIdx]);
            sumA += rgbComps.a;
            sumR += rgbComps.r;
            sumG += rgbComps.g;
            sumB += rgbComps.b;
        }

        short avgA = (short) (sumA / len);
        short avgR = (short) (sumR / len);
        short avgG = (short) (sumG / len);
        short avgB = (short) (sumB / len);

        return Rgb.getRgbIntVal(avgA, avgR, avgG, avgB);
    }

    /*
     * Returns the closest cluster center the list of clusters
     */
    private static int getClosestClusterCenter(
            Set<Integer> clusterCenters,
            int rgbVal,
            int p) {
        double minDist = Integer.MAX_VALUE;
        int res = -1;
        for (Integer clusterCenter : clusterCenters) {
            double dist = getDistance(clusterCenter, rgbVal, p);
            if (dist < minDist) {
                minDist = dist;
                res = clusterCenter;
            }
        }

        return res;
    }

    /*
     * Calculates the Minkowski distance of order p (where p is an integer) between
     * two points
     */
    private static double getDistance(Integer rgb1, int rgb2, int p) {
        Rgb rgbFirst = new Rgb(rgb1);
        Rgb rgbSecond = new Rgb(rgb2);

        return Math.pow(
                Math.pow(Math.abs(rgbFirst.r - rgbSecond.r), p) +
                        Math.pow(Math.abs(rgbFirst.g - rgbSecond.g), p) +
                        Math.pow(Math.abs(rgbFirst.b - rgbSecond.b), p),
                1 / p);
    }
}

class Rgb {

    public short a;
    public short r;
    public short g;
    public short b;

    public Rgb(int rgb) {
        this.a = (short) ((rgb >> 24) & 0xFF);
        this.r = (short) ((rgb >> 16) & 0xFF);
        this.g = (short) ((rgb >> 8) & 0xFF);
        this.b = (short) (rgb & 0xFF);
    }

    static int getRgbIntVal(short a, short r, short g, short b) {
        int ai = a;
        ai = ai << 24;
        int ri = r;
        ri = ri << 16;
        int gi = g;
        gi = gi << 8;
        int bi = b;

        return (int) (ai | ri | gi | bi);
    }
}
