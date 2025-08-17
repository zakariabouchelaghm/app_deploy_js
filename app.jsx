import express from "express";
import bodyParser from "body-parser";
import tflite from "tflite-node";

const app = express();
app.use(bodyParser.json());


const interpreter = new tflite.Interpreter("handwriting_model.tflite");
interpreter.allocateTensors();


const inputDetails = interpreter.getInputTensorInfo(0);
const outputDetails = interpreter.getOutputTensorInfo(0);


app.post("/predict", (req, res) => {
  try {
    const { data } = req.body; // Expecting { data: [...] }

    // Create input tensor
    const inputTensor = new tflite.Tensor(data, inputDetails);
    interpreter.setInputTensor(0, inputTensor);

    // Run inference
    interpreter.invoke();

    // Get output
    const outputTensor = interpreter.getOutputTensor(0);
    const predictionArray = outputTensor.data;

    // Get class with highest probability
    const predictedClass = predictionArray.indexOf(Math.max(...predictionArray));

    res.json({ predicted_class: predictedClass,
      probabilities: predictionArray });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Prediction failed" });
  }
});