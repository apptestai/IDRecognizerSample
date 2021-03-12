package com.embian.edgedetectionsample;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.embian.edgedetectionsample.databinding.ActivityMainBinding;
import com.google.common.util.concurrent.ListenableFuture;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.opencv.core.Mat;

public class MainActivity extends AppCompatActivity {
    private static final String TAG ="CameraSample" ;
    private static final String[] PERMISSIONS = {Manifest.permission.CAMERA};
    private static final int REQUEST_CODE_CAMERA_PERMISSION = 200 ;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private CardRecognizer cardRecognizer;
    private ActivityMainBinding binding = null;

    static {
        System.loadLibrary("opencv_java4");
    }

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

    private void setupCameraX(){
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);
        cardRecognizer = new CardRecognizer(this);
        cameraProviderFuture.addListener(() -> {
            try{
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider);
            } catch(InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

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
            Bitmap bitmap  = EdgeAnalyze(image);
            runOnUiThread(() -> applyAnalyzeResult(bitmap));
            image.close();
        });

        cameraProvider.bindToLifecycle(this, cameraSelector,
                imageAnalysis, preview);
    }

    private void applyAnalyzeResult(Bitmap bitmap){
        binding.overlayView.setImageBitmap(bitmap);
    }

    @SuppressLint("UnsafeExperimentalUsageError")
    private Bitmap EdgeAnalyze(ImageProxy image){
        Log.d(TAG, "Edge Analyze is called");
        Mat src = cardRecognizer.ImageToMat(image);
        src = cardRecognizer.CenterCropTransform(src, binding.viewFinder);
        Mat edge = cardRecognizer.BoundaryDetection2(src);
        edge = cardRecognizer.GrayToGreenRGBA(edge);
        Bitmap dstBitmap = cardRecognizer.MatToBitmap(edge, binding.viewFinder);

        src.release();
        edge.release();
        return dstBitmap;
    }


}