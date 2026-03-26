import {
  layers,
  sequential,
  tensor2d,
  type Sequential,
  type Tensor2D,
} from "@tensorflow/tfjs-node";

const peopleTensors = [
  [0.33, 1, 0, 0, 1, 0, 0],
  [0, 0, 1, 0, 0, 1, 0],
  [1, 0, 0, 1, 0, 0, 1],
];

const labels = ["premium", "medium", "basic"];
const tensorLabels = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

async function trainModel(inputXs: Tensor2D, outputYs: Tensor2D) {
  const model = sequential();

  // First net layer:
  // 7 positions entry (normalized age + 3 colors + 3 locations)

  // 80 neurons = big cause the train database is week and little
  // More neurons == more net learning complexity, and more processing cost.

  // The ReLU acts like a filter:
  // It makes only the relevant data flow through the net
  // If a positive data reach a neuron it goes through, if zero or negative, it dumps it.
  model.add(layers.dense({ inputShape: [7], units: 80, activation: "relu" }));

  // Output: 3 neurons - premium, medium, basic
  // Activation softmax, it normalizes the output in probabilities
  model.add(layers.dense({ units: 3, activation: "softmax" }));

  // Compile model
  // Optimizer Adam (Adaptive Moment Estimation)
  // Adam is a modern personal trainer for neural network:
  // it adjust the weights smart and efficiently, learn from mistakes and successes

  // loss: categoricalCrossentropy
  // Compares what the model "thinks" (category scores) with the correct answer.

  // metrics
  // The further the model's prediction is from the correct answer,
  // bigger loss number
  // Example: Image classification, recommendation, user categorization
  // Anything that answer is deterministic or "one-between-many"
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // Training
  await model.fit(inputXs, outputYs, {
    // Remove internal logs
    verbose: 0,
    // Times that runs the dataset
    epochs: 100,
    // The database on each epoch to prevent BIAS
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, log) =>
        console.log(`Epoch: ${epoch}: loss = ${log ? log.loss : 0}`),
    },
  });

  return model;
}

async function predict(model: Sequential, person: number[][]) {
  // First, transform JS array to tensor
  const tfInput = tensor2d(person);

  // Predict (output will be an 3 probability vector)
  const pred = model.predict(tfInput);
  const predArray = await pred.array();
  return predArray[0].map((prob, index) => ({ prob, index }));
}

const inputXs = tensor2d(peopleTensors);
const outputYs = tensor2d(tensorLabels);

// More data === better model understanding
const model = await trainModel(inputXs, outputYs);

// Test data
// Data that doesn't exist on the train data to test the model understanding.
// To normalize
const new_person = {
  name: "Agatha",
  age: 28,
  color: "green",
  city: "New York",
};
// Normalizing
// min_age = 25, max_age = 40, so (28-25) / (40-25) = 0.2
const normalized_person = [[0.2, 0, 0, 1, 0, 0, 1]];

const predictions = await predict(model, normalized_person);
const results = predictions
  .sort((a, b) => b.prob - a.prob)
  .map((p) => `${labels[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
  .join("\n");
console.log(results);
