package com.embian.edgedetectionsample;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;


import androidx.camera.core.ImageProxy;
import androidx.camera.view.PreviewView;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;

import static org.opencv.android.Utils.bitmapToMat;
import static org.opencv.android.Utils.matToBitmap;
import static org.opencv.core.Core.merge;
import static org.opencv.core.Core.subtract;

public class CardRecognizer {
    private final YuvToRgbConverter imageConverter;

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

    public Mat BoundaryDetection(Mat src){
        Imgproc.blur(src, src, new Size(3,3));
         
        
    }

    public Mat GrayToGreenRGBA(Mat src){
        Mat channel = new Mat();
        Mat ones = Mat.ones(src.size(), CvType.CV_8U);
        subtract(ones, src, channel);
        Core.multiply(src, new Scalar(128), ones);
        merge(new ArrayList<Mat>(Arrays.asList(channel, ones, channel, src)), src);
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

