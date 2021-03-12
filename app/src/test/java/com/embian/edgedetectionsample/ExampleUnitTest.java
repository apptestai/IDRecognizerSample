package com.embian.edgedetectionsample;

import org.junit.Test;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import static org.junit.Assert.*;

/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
public class ExampleUnitTest {
    @Test
    public void card_recognizer_test() {
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        System.loadLibrary("libs/opencv_java4");
        System.loadLibrary("libopencv_java4.so");
        CardRecognizer recognizer = new CardRecognizer();
        Mat src = Imgcodecs.imread("d:\\chae.jpg");
        Mat dst = recognizer.BoundaryDetection(src);
        Imgcodecs.imwrite("chae-result.jpg", dst);
        assertNotNull(dst);
    }
}