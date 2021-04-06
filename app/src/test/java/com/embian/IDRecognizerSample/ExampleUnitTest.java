package com.embian.IDRecognizerSample;

import org.junit.Test;
//import org.opencv.core.Core;
//import org.opencv.core.Mat;
//import org.opencv.imgcodecs.Imgcodecs;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.util.Arrays;

/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
public class ExampleUnitTest {
    @Test
    public void card_recognizer_test() {
        // Context of the app under test.
        Module mod = Module.load("asset/mobilenet_model.pt");
        Tensor data =
                Tensor.fromBlob(
                        new int[] {1, 2, 3, 4, 5, 6}, // data
                        new long[] {2, 3} // shape
                );
        IValue result = mod.forward(IValue.from(data), IValue.from(3.0));
        Tensor output = result.toTensor();
        System.out.println("shape: " + Arrays.toString(output.shape()));
        System.out.println("data: " + Arrays.toString(output.getDataAsFloatArray()));

        // Workaround for https://github.com/facebookincubator/fbjni/issues/25
        System.exit(0);
    }
}