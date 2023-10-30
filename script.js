const session = new onnx.InferenceSession();
const url = 'mnist.onnx'; // Replace with the actual path to your ONNX model
session.loadModel(url).then(() => {
    console.log('Model loaded successfully.');
    // You can perform inference or other operations here
}).catch((err) => {
    console.error('Error loading the model:', err);
});