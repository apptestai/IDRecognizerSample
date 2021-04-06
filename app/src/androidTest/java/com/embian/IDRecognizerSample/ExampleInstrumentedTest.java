package com.embian.IDRecognizerSample;

import androidx.test.ext.junit.runners.AndroidJUnit4;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import java.util.Arrays;

/**
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class ExampleInstrumentedTest {
    @Test
    public void useAppContext() {
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