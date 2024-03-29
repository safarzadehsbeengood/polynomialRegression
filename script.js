var cnv;
let x_vals = [0.5];
let y_vals = [0.5];
let a, b, c, d, loss_val;
let lr;
let opt = tf.train.adam(lr);
var clearBtn, lrSlider;

function setup() {
  cnv = createCanvas(windowWidth - 100, windowHeight - 100);
  cnv.position((windowWidth - width) / 2, (windowHeight - height) / 2);
  a = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
  c = tf.variable(tf.scalar(random(1)));
  d = tf.variable(tf.scalar(random(1)));
  clearBtn = createButton("clear");
  clearBtn.mousePressed(() => {
    x_vals = [0.5];
    y_vals = [0.5];
  });
  lrSlider = createSlider(0.001, 0.5, 0.05, 0.001)
  lr = lrSlider.value();
}

function loss(pred, labels) {
  // take predictions, sub difference from true values, square them, and take the mean
  const loss = pred.sub(labels).square().mean();
  loss_val = loss.dataSync()[0];
  return loss;
}

// using x vals, make a tensor of x vals, and return their respective y vals in a tensor
function predict(x) {
  const tx = tf.tensor1d(x);
  // const ty = tx.square().mul(a).add(tx.mul(b)).add(c); // 2nd degree
  const ty = tx.mul(tx.mul(tx)).mul(a).add(tx.square()).mul(b).add(tx.mul(c)).add(d);
  return ty;
}

function draw() {
  if (lr != lrSlider.value()) {
    lr = lrSlider.value();
    opt = tf.train.adam(lr);
  }
  tf.tidy(() => {
    if (x_vals.length > 0) {
      // optimize
      const ty = tf.tensor1d(y_vals);
      opt.minimize(() => loss(predict(x_vals), ty));
    }
  });
  background(0);
  stroke(255);
  strokeWeight(6);
  for (let i = 0; i < x_vals.length; i++) {
    point(map(x_vals[i], -1, 1, 0, width), map(y_vals[i], -1, 1, height, 0));
  }
  strokeWeight(4);
  stroke(color(70, 230, 70));
  xd = [];
  for (let x = -1; x < 1; x+=0.01) {
    xd.push(x);
  }
  tf.tidy(() => {
    const yd = predict(xd);
    const yvals = yd.dataSync();
    beginShape();
    noFill();
    for (let i = 0; i < xd.length; i++) {
      const xVal = map(xd[i], -1, 1, 0, width);
      const yVal = map(yvals[i], -1, 1, height, 0);
      vertex(xVal, yVal);
    }
    endShape();
  });
  console.log(tf.memory().numTensors);
  fill(255);
  noStroke();
  text("Loss", 30, 25);
  text(loss_val.toFixed(6), 30, 40);
  text(`lr: ${lr.toFixed(3)}`, width - 60, 40);
  text(`points: ${x_vals.length}`, width/2-20, 25);
  stroke(255);
  noFill();
  rect(0, 0, width, height);
  fill(100);
  point(mouseX, mouseY, 20);
}

function mouseDragged() {
  if (mouseX > 0 && mouseX < width && mouseY > 0 && mouseY < height) {
    x_vals.push(map(mouseX, 0, width, -1, 1));
    y_vals.push(map(mouseY, 0, height, 1, -1));
  }
}

window.onresize = () => {
  resizeCanvas(windowWidth - 100, windowHeight - 100);
};
