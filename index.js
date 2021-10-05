import { RedeNeural } from "./RedeNeural.js";

async function run(accept_error = 0.01) {
    var nn = new RedeNeural(2, 50, 1, 0.1, "sigmoid", false);
    await nn.loadWeigths();

    var dataset = {
        inputs:
            [[1, 1],
            [1, 0],
            [0, 1],
            [0, 0]],
        outputs:
            [[0],
            [1],
            [1],
            [0]]
    }
    var train = true;
    const now = new Date();
    while (train) {
        for (var i = 0; i < 10000; i++) {
            var index = Math.floor(Math.random() * dataset.inputs.length);
            nn.train(dataset.inputs[index], dataset.outputs[index]);
        }
        console.log('saida', nn.predict(dataset.inputs[0]), 'index', index);
        if (nn.predict(dataset.inputs[0])[0] < dataset.outputs[0] + accept_error && nn.predict(dataset.inputs[dataset.inputs.length - 1])[0] > dataset.outputs[dataset.inputs.length - 1] - accept_error) {
            train = false;
            console.log("terminou");
            nn.saveWeigths();
        }
    }
    console.log(((new Date()).getTime() - now.getTime()) / 1000)
}
run(0.01)

