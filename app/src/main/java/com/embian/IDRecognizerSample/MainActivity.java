package com.embian.IDRecognizerSample;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.Image;
import android.os.Build;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.embian.IDRecognizerSample.databinding.ActivityMainBinding;
import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.core.Mat;

import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class MainActivity extends AppCompatActivity {
    private static final String[] PERMISSIONS = {Manifest.permission.CAMERA};
    private static final int REQUEST_CODE_CAMERA_PERMISSION = 200 ;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private ActivityMainBinding binding = null;


    private CardRecognizerBySegmentation cardRecognizer;
    private OpenCVUtil opencvUtil;


    static {
        System.loadLibrary("opencv_java4");
    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());

        View view = binding.getRoot();
        setContentView(view);

        if (ActivityCompat.checkSelfPermission( this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this, PERMISSIONS,
                    REQUEST_CODE_CAMERA_PERMISSION);
        }else{
            setupCameraX();
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults){
        if (requestCode == REQUEST_CODE_CAMERA_PERMISSION){
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_DENIED){
                Toast.makeText(this,
                        "You can't use this app without granting CAMERA permission",
                        Toast.LENGTH_LONG)
                        .show();
                finish();
            }else{
                setupCameraX();
            }
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    private void setupCameraX(){
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);
//        cardRecognizer = new CardRecognizer(this);
        try{
            cardRecognizer = new CardRecognizerBySegmentation(this);
            opencvUtil = new OpenCVUtil(this);
        }catch (IOException e) {
            e.printStackTrace();
        }
        cameraProviderFuture.addListener(() -> {
            try{
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider);
            } catch(InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    @SuppressLint("UnsafeExperimentalUsageError")
    private void bindPreview(@NonNull ProcessCameraProvider cameraProvider){
        final Preview preview = new Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .build();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        preview.setSurfaceProvider(binding.viewFinder.getSurfaceProvider());

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(executor, image -> {
            Bitmap bitmap  = EdgeAnalyze2(image);
            if (bitmap == null) return;
            runOnUiThread(() -> applyAnalyzeResult(bitmap));
            image.close();
        });

        cameraProvider.bindToLifecycle(this, cameraSelector,
                imageAnalysis, preview);
    }

    private void applyAnalyzeResult(Bitmap bitmap){
        binding.overlayView.setImageBitmap(bitmap);
    }

    // Segmentation network을 사용한 version
    @SuppressLint("UnsafeExperimentalUsageError")
    private Bitmap EdgeAnalyze2(ImageProxy image){
        final long startTime = SystemClock.elapsedRealtime();

        Image srcImage = image.getImage();
        if (srcImage == null)  return null;

        int target_width = binding.viewFinder.getWidth();
        int target_height = binding.viewFinder.getHeight();
        double imageRotationDegrees = image.getImageInfo().getRotationDegrees();

        Mat srcMat = opencvUtil.ImageToMat(srcImage, imageRotationDegrees);
        srcMat = opencvUtil.CenterCropTransform(srcMat, target_width, target_height);

        Bitmap resultBitmap = cardRecognizer.Detect(srcMat,target_width, target_height);
        srcMat.release();

        final long analyzeDuration = SystemClock.elapsedRealtime() - startTime;
        Log.e("Analyzer", "Analyze: " + analyzeDuration);
        return resultBitmap;
    }


    // 순수 openCV Version
    @SuppressLint("UnsafeExperimentalUsageError")
    private Bitmap EdgeAnalyze(ImageProxy image){
        Image srcImage = image.getImage();
        if (srcImage == null)  return null;

        int target_width = binding.viewFinder.getWidth();
        int target_height = binding.viewFinder.getHeight();
        double imageRotationDegrees = image.getImageInfo().getRotationDegrees();

        Mat srcMat = opencvUtil.ImageToMat(srcImage, imageRotationDegrees);
        srcMat = opencvUtil.CenterCropTransform(srcMat, target_width, target_height);

        Mat edge = opencvUtil.DetectQuad(srcMat);
        edge = opencvUtil.GrayToGreenRGBA(edge);
        Bitmap dstBitmap = opencvUtil.MatToBitmap(edge, target_width, target_height);

        srcMat.release();
        edge.release();
        return dstBitmap;
    }
}