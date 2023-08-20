package com.example.multimodule

import android.content.Context
import android.graphics.Bitmap
import android.media.ThumbnailUtils
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.result.launch
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import com.example.multimodule.ml.Finale
import com.example.multimodule.ml.Model
import com.example.multimodule.ml.PlantsModel
import com.example.multimodule.ui.theme.MultiModuleTheme
import org.json.JSONObject
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min


class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MultiModuleTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    CaptureImage()
                }
            }
        }
    }
}

@Composable
fun CaptureImage() {
    val context = LocalContext.current
    var imageBitmap by remember {
        mutableStateOf<ImageBitmap?>(null)
    }
    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicturePreview(),
        onResult = {
            imageBitmap = it?.asImageBitmap()
        }
    )
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Button(onClick = {
            launcher.launch()
        }) {
            Text(text = "Capture Image", color = MaterialTheme.colorScheme.onPrimary)
        }
        if (imageBitmap != null) {
            Button(onClick = {
                val imageSize = 200
                var image = imageBitmap?.asAndroidBitmap()
                if (image != null) {
                    val dimension = min(image.width, image.height)
                    val newImageBitmap =
                        ThumbnailUtils.extractThumbnail(image, dimension, dimension)
                    image = Bitmap.createScaledBitmap(
                        newImageBitmap,
                        imageSize,
                        imageSize,
                        false
                    )
                    classifyPart3(image, context = context)
                }
            }) {
                Text(
                    text = "Hello ML",
                    color = MaterialTheme.colorScheme.onPrimary
                )
            }
        }
    }

}

fun classifyImage(bitmap: Bitmap, context: Context) {
    val model = PlantsModel.newInstance(context.applicationContext)

    val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 200, 200, 3), DataType.FLOAT32)
    val byteBuffer = ByteBuffer.allocateDirect(4 * 200 * 200 * 3)
    byteBuffer.order(ByteOrder.nativeOrder())
    val intValues = IntArray(200 * 200)
    bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
    var pixel = 0
    for (i in 0 until 200) {
        for (j in 0 until 200) {
            val `val` = intValues[pixel++]
            byteBuffer.putFloat((`val` shr 16 and 0xFF) / 255f)
            byteBuffer.putFloat((`val` shr 8 and 0xFF) / 255f)
            byteBuffer.putFloat((`val` and 0xFF) / 255f)
        }
    }
    inputFeature0.loadBuffer(byteBuffer)

    val outputs = model.process(inputFeature0)
    val outputFeature0 = outputs.outputFeature0AsTensorBuffer
    Log.i("TAG", "classifyImage: ${outputFeature0.floatArray.contentToString()}")
    val byteArray = ByteArray(outputFeature0.buffer.remaining())
    val jsonString = byteArray.toString(Charsets.UTF_8)
    Log.i("Raw JSON Data:", "classifyImage: $jsonString")
    println(jsonString)
    model.close()
}

fun classifyPart2(bitmap: Bitmap, context: Context) {
    val model = Model.newInstance(context)

// Creates inputs for reference.
    val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 1), DataType.FLOAT32)
    // Determine the size of the input tensor based on its shape and data type
    val inputSize = 1 * 1 * DataType.FLOAT32.byteSize() // 1 x 1 x 4 (FLOAT32 size)

// Create a ByteBuffer of the required size
    val byteBuffer = ByteBuffer.allocateDirect(inputSize)
    byteBuffer.order(ByteOrder.nativeOrder()) // Set the byte order to native (platform-specific)

// Populate the ByteBuffer with your input data (assuming you have a Float value)
    val inputValue: Float = 0.5f // Replace with your actual input value
    byteBuffer.putFloat(inputValue)

// Reset the ByteBuffer's position to 0 before use
    byteBuffer.rewind()
    inputFeature0.loadBuffer(byteBuffer)

// Runs model inference and gets result.
    val outputs = model.process(inputFeature0)
    val outputFeature0 = outputs.outputFeature0AsTensorBuffer
    Log.i("IMAGES", "classifyImage: $outputFeature0")
    val byteArray = ByteArray(outputFeature0.buffer.remaining())
    outputFeature0.buffer.get(byteArray)
    val jsonString = byteArray.toString(Charsets.UTF_8)

    Log.i("Raw JSON Data:", "classifyImage: $jsonString")
    println(jsonString)

// Releases model resources if no longer used.
    model.close()
}


fun extractPredictions(outputFeature0: TensorBuffer): List<Map<String, Any>> {
    try {
        val obj = JSONObject(outputFeature0.buffer.toString())
        val predictions = obj.getJSONArray("predictions")
        val predictionList = mutableListOf<Map<String, Any>>()
        for (i in 0 until predictions.length()) {
            val prediction = predictions.getJSONObject(i)
            val label = prediction.getString("label")
            val score = prediction.getDouble("score")
            predictionList.add(mapOf("label" to label, "score" to score))
        }
        return predictionList

    } catch (e: IOException) {
        e.printStackTrace()
        return emptyList()
    }
}

fun classifyPart3(bitmap: Bitmap, context: Context) {
    val model = Finale.newInstance(context)

// Creates inputs for reference.
    val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 3, 200, 200), DataType.FLOAT32)
    val byteBuffer = ByteBuffer.allocateDirect(4 * 200 * 200 * 3)
    byteBuffer.order(ByteOrder.nativeOrder())
    val intValues = IntArray(200 * 200)
    bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
    var pixel = 0
    for (i in 0 until 200) {
        for (j in 0 until 200) {
            val `val` = intValues[pixel++]
            byteBuffer.putFloat((`val` shr 16 and 0xFF) / 255f)
            byteBuffer.putFloat((`val` shr 8 and 0xFF) / 255f)
            byteBuffer.putFloat((`val` and 0xFF) / 255f)
        }
    }
    inputFeature0.loadBuffer(byteBuffer)

// Runs model inference and gets result.
    val outputs = model.process(inputFeature0)
    val outputFeature0 = outputs.outputFeature0AsTensorBuffer
    Log.i("TAG", "classifyImage: ${outputFeature0.floatArray.contentToString()}")
    val confidences = outputFeature0.floatArray
    // find the index of the class with the biggest confidence.
    // find the index of the class with the biggest confidence.
    // Post-processor which dequantize the result
    // Post-processor which dequantize the result
    val labelsList = listOf(
        "Tomato with Bacterial Spot", "Grape with Esca (Black Measles)", "Grape with Black Rot",
        "Strawberry with Leaf Scorch", "Tomato with Late Blight"
    )
    Log.i("CMON", "Ehlo ${convertToLabelScorePairs(outputFeature0.floatArray, labelsList)}")
// Releases model resources if no longer used.
    model.close()

}

fun convertToLabelScorePairs(
    outputArray: FloatArray,
    labels: List<String>
): List<Pair<String, Float>> {
    val labelScorePairs = mutableListOf<Pair<String, Float>>()

    for (i in outputArray.indices) {
        try {
            val pair = Pair(labels[i], outputArray[i])
            Log.i("CMON", "paiiiir $pair")
            labelScorePairs.add(Pair(labels[i - 1], outputArray[i]))
        } catch (e: Exception) {
            continue
        }
    }

    return labelScorePairs
}









