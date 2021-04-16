package com.embian.IDRecognizerSample;

import android.content.Context;
import android.graphics.Bitmap;
import android.media.Image;


import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;


import static org.opencv.android.Utils.bitmapToMat;
import static org.opencv.android.Utils.matToBitmap;
import static org.opencv.core.Core.merge;
import static org.opencv.core.Core.subtract;

public class OpenCVUtil {
    private final YuvToRgbConverter imageConverter;

    public OpenCVUtil(){imageConverter = null;}
    public OpenCVUtil(Context context){
        imageConverter = new YuvToRgbConverter(context);
    }

    public Mat ImageToMat(Image image, double rotationDegrees){
        Bitmap srcBitmap = Bitmap.createBitmap(image.getWidth(), image.getHeight(),
                Bitmap.Config.ARGB_8888);
        Mat src = new Mat();
        imageConverter.yuvToRgb(image, srcBitmap);
        bitmapToMat(srcBitmap, src);

        if (rotationDegrees == 90){
            Core.transpose(src, src);
            Core.flip(src, src, 1);
        }else if(rotationDegrees==180){
            Core.flip(src, src, 0);
        }else if(rotationDegrees==270){
            Core.transpose(src, src);
            Core.flip(src, src, 0);
        }
        return src;
    }

    public Mat CenterCropTransform(Mat src, int target_width, int target_height){

        int adjust_height = src.height() - (src.width()*target_height/target_width);
        int adjust_width = src.width() - (src.height()*target_width/target_height);

        if (adjust_height > 0){
            adjust_height = adjust_height/2;
            src = src.submat(new Rect(0, adjust_height, src.width(), src.height() - adjust_height*2));
        }else if(adjust_width > 0){
            adjust_width = adjust_width/2;
            src = src.submat(new Rect(adjust_width, 0, src.width() - adjust_width*2, src.height()));
        }
        return src;
    }

    public Mat DetectQuad(Mat src){
        Mat edge = new Mat();
        Mat hierarchy = new Mat();
        int srcArea = src.width() * src.height() ;

//        Imgproc.blur(src, src, new Size(3,3));
//        Imgproc.erode(src, src, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3)));
//        Imgproc.dilate(src, src, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(6, 6)));
        Imgproc.Canny(src, edge, 50, 150);

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(edge, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        // Get the 5 largest contours
        Collections.sort(contours, (o1, o2) -> {
            double area1 = Imgproc.contourArea(o1);
            double area2 = Imgproc.contourArea(o2);
            return (int) (area2 - area1);
        });

        if (contours.size() > 5) contours.subList(4, contours.size() - 1).clear();

        MatOfPoint2f largest2f = null;
        MatOfPoint2f secondLargest2f = null;
        MatOfPoint largest = new MatOfPoint();
        MatOfPoint secondLargest = new MatOfPoint();

        for (MatOfPoint contour : contours) {
            MatOfInt hull = new MatOfInt();
            Imgproc.convexHull(contour, hull);
            Point[] contourArray = contour.toArray();
            Point[] hullPoints = new Point[hull.rows()];
            List<Integer> hullContourIdxList = hull.toList();
            for (int i = 0; i < hullContourIdxList.size(); i++) {
                hullPoints[i] = contourArray[hullContourIdxList.get(i)];
            }

            MatOfPoint new_contour = new MatOfPoint(hullPoints);
            MatOfPoint2f approx = new MatOfPoint2f();
            MatOfPoint2f c = new MatOfPoint2f();
            new_contour.convertTo(c, CvType.CV_32FC2);
            Imgproc.approxPolyDP(c, approx, Imgproc.arcLength(c, true) * 0.02, true);


            if (srcArea * 0.01 > Imgproc.contourArea(approx))
                continue;

            if (isRectangle(approx)){
                if (largest2f == null){
                    largest2f = approx;
                }else{
                    secondLargest2f = approx;
                    break;
                }
            }
        }

        Mat drawing = Mat.zeros(edge.size(), CvType.CV_8UC3);
        if (largest2f != null){
            List<MatOfPoint> hullList = new ArrayList<>();
            largest2f.convertTo(largest, CvType.CV_32S);
            hullList.add(largest);

            if (secondLargest2f != null){
                secondLargest2f.convertTo(secondLargest, CvType.CV_32S);
                hullList.add(secondLargest);
            }

            Imgproc.drawContours(drawing, hullList, 0, new Scalar(255, 255, 255), 2,
                    Imgproc.LINE_8, hierarchy, 0, new Point());
            Imgproc.cvtColor(drawing, edge, Imgproc.COLOR_BGR2GRAY);
            drawing.release();
        }else{
            Imgproc.cvtColor(drawing, edge, Imgproc.COLOR_BGR2GRAY);
        }

        return edge;
    }

//    public Mat Resize(Mat src){
//        int width = src.width()/4;
//        int height = src.height()/4;
//        Mat resizeImage = new Mat();
//        Size sz = new Size(width,height);
//        Imgproc.resize( src, resizeImage, sz );
//        return resizeImage;
//    }

//    public Mat EdgeDetection(Mat src){
//        Mat edge = new Mat();
//        Imgproc.Canny(src, edge, 50, 150);
//        return edge;
//    }

    public static double angle(Point p1, Point p2, Point p0) {
        double dx1 = p1.x - p0.x;
        double dy1 = p1.y - p0.y;
        double dx2 = p2.x - p0.x;
        double dy2 = p2.y - p0.y;
        return (dx1 * dx2 + dy1 * dy2) / Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
    }

    private boolean isRectangle(MatOfPoint2f polygon ){
        // Check if the all angles are more than 72.54 degrees (cos 0.3).
        if (polygon.rows() != 4) return false;

        double maxCosine = 0;
        Point[] approxPoints = polygon.toArray();
//        Check if the all angles are more than 72.54 degrees (cos 0.3).
        for (int i = 2; i < 5; i++) {
            double cosine = angle(approxPoints[i % 4], approxPoints[i - 2], approxPoints[i - 1]);
            maxCosine = Math.min(cosine, maxCosine);
        }
        return maxCosine <= 0.3;
    }


    public Mat GrayToGreenRGBA(Mat src){
        Mat channel = new Mat();
        Mat ones = Mat.ones(src.size(), CvType.CV_8U);
        subtract(ones, src, channel);
        Core.multiply(src, new Scalar(128), ones);
        merge(new ArrayList<>(Arrays.asList(channel, ones, channel, src)), src);
        ones.release();
        return src;
    }

    public Bitmap MatToBitmap(Mat src, int width, int height){
        Imgproc.resize(src, src, new Size(width,height));
        Bitmap dstBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        matToBitmap(src, dstBitmap);
        return dstBitmap;
    }
}

