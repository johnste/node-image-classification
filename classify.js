process.env["TF_CPP_MIN_LOG_LEVEL"] = "2";
//TensorFlow.js is an open-source hardware-accelerated JavaScript library
//for training and deploying machine learning models.
const tf = require("@tensorflow/tfjs");
//MobileNet : pre-trained model for TensorFlow.js
const mobilenet = require("@tensorflow-models/mobilenet");
//The module provides native TensorFlow execution
//in backend JavaScript applications under the Node.js runtime.
const tfnode = require("@tensorflow/tfjs-node");
//The fs module provides an API for interacting with the file system.
const fs = require("fs");
const readline = require("readline");
const chalk = require("chalk");
const ora = require("ora");
const prompt = require("prompt");
const imgcat = require("imgcat");
const path = require("path");

const imagePaths = process.argv.slice(2);
let numberLeft = imagePaths.length;

if (process.argv.length < 3)
  throw new Error("Incorrect arguments: node classify.js <IMAGE_FILE>");

const spinner = ora(`Analyzing ${numberLeft} images`).start();

const imagePredictions = Promise.all(
  imagePaths.map((image) => imageClassification(image, spinner))
);

rename(imagePredictions);

async function rename(imagePredictions) {
  const images = await imagePredictions;
  spinner.stop();

  for await (file of images) {
    const { predictions, path } = file;
    const suggestedNames = getNames(predictions, path);

    let option = undefined;

    suggestedNames.forEach(({ name, rating }, i) => {
      let stars = "⭐️";
      if (rating > 0.3) {
        stars += "⭐️";
      }

      if (rating > 0.9) {
        stars += "⭐️";
      }

      console.log(`${i + 1}: ${name} ${stars}`);
    });

    console.log(await imgcat(path, { height: 5 }));
    console.log(`Do you want to rename ${chalk.magenta(path)} to:`);

    const result = await prompt.get(["option"]);

    const option = +result.option;

    if (option >= 1 && option <= suggestedNames.length) {
      const selectedName = suggestedNames[option - 1].name;
      fs.renameSync(path, selectedName);
      console.log(
        `${chalk.magenta(path)} renamed to ${chalk.green(selectedName)}`
      );
    }
  }
}

function readImage(path) {
  //reads the entire contents of a file.
  //readFileSync() is synchronous and blocks execution until finished.
  const imageBuffer = fs.readFileSync(path);
  //Given the encoded bytes of an image,
  //it returns a 3D or 4D tensor of the decoded image. Supports BMP, GIF, JPEG and PNG formats.
  const tfimage = tfnode.node.decodeImage(imageBuffer);
  return tfimage;
}

function getNames(predictions, filepath) {
  const suffix = filepath.slice(filepath.lastIndexOf("."));
  const filename = path.basename(filepath);
  const dirname = path.dirname(filepath);

  const allNames = predictions.flatMap((prediction) => {
    const suggestedPaths = prediction.className
      .split(",")
      .map((name) => name.trim())
      .filter((name) => name + suffix !== filename)
      .map((name) => {
        let suggestedPath = path.resolve(dirname, name + suffix);
        let attempt = 0;
        while (fs.existsSync(suggestedPath)) {
          suggestedPath = path.resolve(
            dirname,
            `${name} (${++attempt})${suffix}`
          );
        }

        return suggestedPath;
      })
      .map((suggestedPath) => path.relative(process.cwd(), suggestedPath));

    return suggestedPaths.map((suggestedPath) => ({
      name: suggestedPath,
      rating: prediction.probability,
    }));
  });

  return allNames.slice(0, 6);
}

async function imageClassification(path, spinner) {
  const image = readImage(path);
  // Load the model.
  const mobilenetModel = await mobilenet.load({
    version: 2,
    alpha: 1.0,
  });

  // Classify the image.
  const result = { predictions: await mobilenetModel.classify(image), path };

  spinner.text = `Analyzing ${numberLeft} images`;

  return result;
}
