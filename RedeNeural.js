import { Matrix } from "./matrix.js";
import fs from "fs";
//https://www.youtube.com/watch?v=d8U7ygZ48Sc
//https://www.youtube.com/watch?v=zVlMVanp-tA&t=0s
//https://www.youtube.com/watch?v=9KfelZhls2Q
class RedeNeural {
    //i_nodes numero de nós das entradas
    //h_nodes número de nós do oculto
    //o_nodes número de nós da saída
    constructor(i_nodes, h_nodes, o_nodes, learning_rate, activation_function = "sigmoid", bias_1 = false) {
        this.i_nodes = i_nodes;
        this.h_nodes = h_nodes;
        this.o_nodes = o_nodes;
        this.bias_ih = new Matrix(this.h_nodes, 1);
        this.bias_ho = new Matrix(this.o_nodes, 1);
        this.weigths_ih = new Matrix(this.h_nodes, this.i_nodes);
        this.weigths_ho = new Matrix(this.o_nodes, this.h_nodes);
        this.weigths_ho.randomize();
        this.weigths_ih.randomize();

        if (bias_1) {
            this.bias_ho.map(bias1)
            this.bias_ih.map(bias1)
        }
        else {
            this.bias_ho.randomize();
            this.bias_ih.randomize();
        }
        this.learning_rate = learning_rate;
        this.activation_function = activation_function;

    }
    async train(arr, target) {
        var activation_function = sigmoid;
        var d_activation_function = sigmoid;
        switch (this.activation_function) {
            case "sigmoid":
                activation_function = sigmoid;
                d_activation_function = dsigmoid;
                break;
            case "tanh":
                activation_function = tanh;
                d_activation_function = dtanh;
                break;
            default:
                activation_function = sigmoid;
                d_activation_function = dsigmoid;
                break;
        }
        //feedfoward
        //oculta =  FuncaoAtivação (pesos_entradas_oculta * entrada)
        //output =  FuncaoAtivação (pesos_oculta_saida * saida_oculta)
        // INPUT -> HIDDEN
        let input = Matrix.arrayToMatrix(arr);
        let hidden = Matrix.multiply(this.weigths_ih, input);
        hidden = Matrix.add(hidden, this.bias_ih);
        hidden.map(activation_function)

        // HIDDEN -> OUTPUT
        // d(Sigmoid) = Output * (1- Output)
        let output = Matrix.multiply(this.weigths_ho, hidden);
        output = Matrix.add(output, this.bias_ho);
        output.map(activation_function);

        //backpropagation
        //delta_pesos_saida_oculta= Erro x d(S)*lr*oculta_transposta
        //output_error = Esperado-Saida
        // OUTPUT -> HIDDEN
        let expected = Matrix.arrayToMatrix(target);
        let output_error = Matrix.subtract(expected, output);

        let d_output = Matrix.map(output, d_activation_function);
        let hidden_T = Matrix.transpose(hidden);

        let gradient = Matrix.hadamard(d_output, output_error);
        gradient = Matrix.escalar_multiply(gradient, this.learning_rate);

        // Adjust Bias O->H
        this.bias_ho = Matrix.add(this.bias_ho, gradient);
        // Adjust Weigths O->H
        let weigths_ho_deltas = Matrix.multiply(gradient, hidden_T);
        this.weigths_ho = Matrix.add(this.weigths_ho, weigths_ho_deltas);

        //erro_oculta = pesos_oculta_saida_transposto*erro_saida
        //delta_pesos_oculta_entrada= Erro_oculta x d(O)*lr*entrada_transposta
        // HIDDEN -> INPUT
        let weigths_ho_T = Matrix.transpose(this.weigths_ho);
        let hidden_error = Matrix.multiply(weigths_ho_T, output_error);
        let d_hidden = Matrix.map(hidden, d_activation_function);
        let input_T = Matrix.transpose(input);

        let gradient_H = Matrix.hadamard(d_hidden, hidden_error);
        gradient_H = Matrix.escalar_multiply(gradient_H, this.learning_rate);

        // Adjust Bias O->H
        this.bias_ih = Matrix.add(this.bias_ih, gradient_H);
        // Adjust Weigths H->I
        let weigths_ih_deltas = Matrix.multiply(gradient_H, input_T);
        this.weigths_ih = Matrix.add(this.weigths_ih, weigths_ih_deltas);

    }
    async saveWeigths() {
        const data = {
            weigths_ih: this.weigths_ih,
            weigths_ho: this.weigths_ho,
            bias_ih: this.bias_ih,
            bias_ho: this.bias_ho,
            activation_function: this.activation_function
        }
        await fs.promises.writeFile("weigths.json", JSON.stringify(data));
    }
    async loadWeigths() {
        try {
            const data = await fs.promises.readFile("weigths.json");
            const data_json = JSON.parse(data);
            console.log("pesos carregados: ", data_json)
            if (data_json.weigths_ho.rows != this.weigths_ho.rows
                || data_json.weigths_ho.cols != this.weigths_ho.cols ||
                data_json.weigths_ih.rows != this.weigths_ih.rows
                || data_json.weigths_ih.cols != this.weigths_ih.cols ||
                data_json.bias_ho.rows != this.bias_ho.rows
                || data_json.bias_ho.cols != this.bias_ho.cols ||
                data_json.bias_ih.rows != this.bias_ih.rows
                || data_json.bias_ih.cols != this.bias_ih.cols ||
                data_json.activation_function != this.activation_function
            ) {
                console.log("pesos na base não considerados")
                return;
            }
            if (data_json.weigths_ho)
                Object.assign(this.weigths_ho, data_json.weigths_ho)
            if (data_json.weigths_ih)
                Object.assign(this.weigths_ih, data_json.weigths_ih)
            if (data_json.bias_ho)
                Object.assign(this.bias_ho, data_json.bias_ho)
            if (data_json.bias_ih)
                Object.assign(this.bias_ih, data_json.bias_ih)
        } catch (error) {
            console.log("pesos na base não considerados")
            return;
        }

    }
    predict(arr) {
        // INPUT -> HIDDEN
        let input = Matrix.arrayToMatrix(arr);

        let hidden = Matrix.multiply(this.weigths_ih, input);
        hidden = Matrix.add(hidden, this.bias_ih);

        hidden.map(sigmoid)

        // HIDDEN -> OUTPUT
        let output = Matrix.multiply(this.weigths_ho, hidden);
        output = Matrix.add(output, this.bias_ho);
        output.map(sigmoid);
        output = Matrix.MatrixToArray(output);

        return output;
    }

}
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x))
}

function dsigmoid(x) {
    return x * (1 - x)

}

function tanh(x) {
    return Math.tanh(x)
}

function dtanh(x) {
    return (1 - x * x)

}

function bias1(x) {
    return x - x + 1
}




export { RedeNeural }