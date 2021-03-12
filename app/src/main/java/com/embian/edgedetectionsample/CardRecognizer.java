package com.embian.edgedetectionsample;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Build;


import androidx.annotation.RequiresApi;
import androidx.camera.core.ImageProxy;
import androidx.camera.view.PreviewView;
import androidx.core.math.MathUtils;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.Random;

import static org.opencv.android.Utils.bitmapToMat;
import static org.opencv.android.Utils.matToBitmap;
import static org.opencv.core.Core.merge;
import static org.opencv.core.Core.subtract;

public class CardRecognizer {
    private final YuvToRgbConverter imageConverter;

    public CardRecognizer(){imageConverter = null;}
    public CardRecognizer(Context context){
        imageConverter = new YuvToRgbConverter(context);
    }

    @SuppressLint("UnsafeExperimentalUsageError")
    public Mat ImageToMat(ImageProxy image){
        double imageRotationDegrees = image.getImageInfo().getRotationDegrees();
        Bitmap srcBitmap = Bitmap.createBitmap(image.getWidth(), image.getHeight(),
                Bitmap.Config.ARGB_8888);
        Mat src = new Mat();
        imageConverter.yuvToRgb(image.getImage(), srcBitmap);
        bitmapToMat(srcBitmap, src);

        if (imageRotationDegrees == 90){
            Core.transpose(src, src);
            Core.flip(src, src, 1);
        }else if(imageRotationDegrees==180){
            Core.flip(src, src, 0);
        }else if(imageRotationDegrees==270){
            Core.transpose(src, src);
            Core.flip(src, src, 0);
        }
        return src;
    }

    public Mat CenterCropTransform(Mat src, PreviewView viewFinder){
        int preview_width = viewFinder.getWidth();
        int preview_height = viewFinder.getHeight();

        int adjust_height = src.height() - (src.width()*preview_height/preview_width);
        int adjust_width = src.width() - (src.height()*preview_width/preview_height);

        if (adjust_height > 0){
            adjust_height = adjust_height/2;
            src = src.submat(new Rect(0, adjust_height, src.width(), src.height() - adjust_height*2));
        }else if(adjust_width > 0){
            adjust_width = adjust_width/2;
            src = src.submat(new Rect(adjust_width, 0, src.width() - adjust_width*2, src.height()));
        }
        return src;
    }

    public Mat EdgeDetection(Mat src){
        Mat edge = new Mat();
        Imgproc.Canny(src, edge, 50, 150);
        return edge;
    }

    public static double angle(Point p1, Point p2, Point p0) {
        double dx1 = p1.x - p0.x;
        double dy1 = p1.y - p0.y;
        double dx2 = p2.x - p0.x;
        double dy2 = p2.y - p0.y;
        return (dx1 * dx2 + dy1 * dy2) / Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
    }

    private boolean isRectangle(MatOfPoint2f polygon, double area){
        // Check if the all angles are more than 72.54 degrees (cos 0.3).
        if (area < 150.0) return false;

        if (polygon.rows() != 4) return false;

        double maxCosine = 0;
        Point[] approxPoints = polygon.toArray();
//        Check if the all angles are more than 72.54 degrees (cos 0.3).
        for (int i = 2; i < 5; i++) {
            double cosine = Math.abs(angle(approxPoints[i % 4], approxPoints[i - 2], approxPoints[i - 1]));
            maxCosine = Math.max(cosine, maxCosine);
        }
        return !(maxCosine >= 0.4);
    }

    public Mat BoundaryDetection2(Mat src){
        Mat edge = new Mat();
        Mat hierarchy = new Mat();
        Imgproc.Canny(src, edge, 50, 150);
        Imgproc.blur(edge, edge, new Size(3,3));

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(edge, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        // Get the 5 largest contours
        Collections.sort(contours, new Comparator<MatOfPoint>() {
            public int compare(MatOfPoint o1, MatOfPoint o2) {
                double area1 = Imgproc.contourArea(o1);
                double area2 = Imgproc.contourArea(o2);
                return (int) (area2 - area1);
            }
        });

        if (contours.size() > 5) contours.subList(4, contours.size() - 1).clear();

        MatOfPoint2f largest2f = null;
        MatOfPoint largest = new MatOfPoint();

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
            if (isRectangle(approx, Imgproc.contourArea(contour))){
                largest2f = approx;
                break;
            }
        }

        Mat drawing = Mat.zeros(edge.size(), CvType.CV_8UC3);
        if (largest2f != null){
            List<MatOfPoint> hullList = new ArrayList<>();
            largest2f.convertTo(largest, CvType.CV_32S);
            hullList.add(largest);
            Imgproc.drawContours(drawing, hullList, 0, new Scalar(255, 255, 255), 2,
                    Imgproc.LINE_8, hierarchy, 0, new Point());
            Imgproc.cvtColor(drawing, edge, Imgproc.COLOR_BGR2GRAY);
            drawing.release();
        }else{
            Imgproc.cvtColor(drawing, edge, Imgproc.COLOR_BGR2GRAY);
        }

        return edge;
    }

    public Mat BoundaryDetection(Mat src){
        Mat edge = new Mat();
        Imgproc.Canny(src, edge, 50, 150);
        Imgproc.blur(edge, edge, new Size(3,3));

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(edge, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        Mat drawing = Mat.zeros(edge.size(), CvType.CV_8UC3);

        double maxArea = 0;
        int maxContourId = 0;
        for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++){
            double contourArea = Imgproc.contourArea(contours.get(contourIdx));
            if (maxArea < contourArea){
                maxArea = contourArea;
                maxContourId = contourIdx;
            }
        }

        List<MatOfPoint> hullList = new ArrayList<>();
        MatOfInt hull = new MatOfInt();
        Imgproc.convexHull(contours.get(maxContourId), hull);
        Point[] contourArray = contours.get(maxContourId).toArray();
        Point[] hullPoints = new Point[hull.rows()];
        List<Integer> hullContourIdxList = hull.toList();
        for (int i = 0; i < hullContourIdxList.size(); i++) {
                hullPoints[i] = contourArray[hullContourIdxList.get(i)];
        }
        hullList.add(new MatOfPoint(hullPoints));

        if (hullPoints.length <= 4){
            RotatedRect rect = Imgproc.minAreaRect(new MatOfPoint2f(hullPoints));
            Mat box = new Mat();
            Imgproc.boxPoints(rect, box);
        }


        Imgproc.drawContours(drawing, hullList, 0, new Scalar(255, 255, 255), 2,
                Imgproc.LINE_8, hierarchy, 0, new Point());
        Imgproc.cvtColor(drawing, edge, Imgproc.COLOR_BGR2GRAY);

        drawing.release();
        hierarchy.release();
        return edge;
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

    public Bitmap MatToBitmap(Mat src, PreviewView viewFinder){
        Imgproc.resize(src, src, new Size(viewFinder.getWidth(),viewFinder.getHeight()));
        Bitmap dstBitmap = Bitmap.createBitmap(viewFinder.getWidth(),
                viewFinder.getHeight(), Bitmap.Config.ARGB_8888);
        matToBitmap(src, dstBitmap);
        return dstBitmap;
    }
}

