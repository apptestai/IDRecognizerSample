package com.embian.IDRecognizerSample;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import org.opencv.core.Mat;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import static org.opencv.android.Utils.bitmapToMat;


public class CardRecognizerBySegmentation {
    private final Module module;
    private final OpenCVUtil opencvUtil= new OpenCVUtil();
    private final int SEGMENTATION_INPUT_WIDTH = 270;
    private final int SEGMENTATION_INPUT_HEIGHT = 480;

    public CardRecognizerBySegmentation(Context context) throws IOException {
        module = Module.load(assetFilePath(context, "mobile_real_2_0.torchscript"));
    }


    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

//    @RequiresApi(api = Build.VERSION_CODES.O)//
//    Just use to input tensor test
//    public Bitmap TensorToBitmap(Tensor inputTensor, int width, int height){
//        float R, G, B, A;
//        int color;
//        final float[] inputs = inputTensor.getDataAsFloatArray();
//        int[] srcImage = new int[width*height];
//        for (int j = 0; j < height; j++) {
//            for (int k = 0; k < width; k++) {
//                A = 1.0f;
//                R = inputs[j*width + k]*0.229f + 0.485f;
//                G = inputs[width*height + j*width + k] *0.224f + 0.456f;
//                B = inputs[width*height*2 + j*width + k] * 0.225f + 0.406f;
//                color = Color.argb(A, R, G, B);
//                srcImage[j*width+k] = color;
//            }
//        }
//        return Bitmap.createBitmap(srcImage, width, height, Bitmap.Config.ARGB_8888);
//    }

    public Bitmap SegmentationToBitImage(Tensor outputTensor) {
        int class_number, color;
        int[] intValues = new int[SEGMENTATION_INPUT_WIDTH * SEGMENTATION_INPUT_HEIGHT];
        final int[] scores = outputTensor.getDataAsIntArray();
        for (int j = 0; j < SEGMENTATION_INPUT_HEIGHT; j++) {
            for (int k = 0; k < SEGMENTATION_INPUT_WIDTH; k++) {
                class_number = scores[j * SEGMENTATION_INPUT_WIDTH + k];
                if (class_number == 0)
                    color = 0x00FFFFFF;
                else if (class_number == 1)
                    color = 0x1F000000;
                else if (class_number == 2)
                    color = 0x1F000000;
                else if (class_number == 3)
                    color = 0x1F000000;
                else if (class_number == 4)
                    color = 0x1F000000;
                else if (class_number == 5)
                    color = 0x00FFFFFF;
                else
                    color = 0x00FFFFFF;

                intValues[j * SEGMENTATION_INPUT_WIDTH + k] = color;
            }
        }
        return Bitmap.createBitmap(intValues, SEGMENTATION_INPUT_WIDTH, SEGMENTATION_INPUT_HEIGHT, Bitmap.Config.ARGB_8888);
    }

    public Bitmap ConvertToPolygon(Bitmap srcBitmap) {
        Mat src = new Mat();
        bitmapToMat(srcBitmap, src);
        src = opencvUtil.DetectQuad(src);
        src = opencvUtil.GrayToGreenRGBA(src);
        Bitmap dstBitmap = opencvUtil.MatToBitmap(src, srcBitmap.getWidth(), srcBitmap.getHeight());
        src.release();
        return dstBitmap;
    }

    public Bitmap Detect(Mat srcMat, int target_width, int target_height){
        Bitmap dstBitmap = opencvUtil.MatToBitmap(srcMat, SEGMENTATION_INPUT_WIDTH, SEGMENTATION_INPUT_HEIGHT);
//        todo input Tensor 재활용
//        todo direct converting from Mat to InputTensor or Image to input Tensor by Torch

        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(dstBitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

        final long startTime = SystemClock.elapsedRealtime();
        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        final long inferenceDuration = SystemClock.elapsedRealtime() - startTime;
        Log.e("Analyzer", "Inference: " + inferenceDuration);

        Bitmap segmentBitmap = SegmentationToBitImage(outputTensor);
        Bitmap originalBitmap = Bitmap.createScaledBitmap(segmentBitmap,target_width , target_height, false);
        return ConvertToPolygon(originalBitmap);
    }
}