import java.util.Arrays;
import java.io.*;

/**
 * Classe que representa a rede neuronal com dois neuronios
 * @version 2.0
 * @author Daniel Goncalves, Diogo Damasio, Goncalo Rodrigues
 */
public class NeuralNetwork {

    private Neuron h, o;


    /**
     * Construtor da Rede Neuronal
     *
     * @param p1 Perceptron de entrada
     * @param p2 Perceptron de saida
     */
    public NeuralNetwork(Neuron p1, Neuron p2) {

        this.h = p1;
        this.o = p2;

    }


    /**
     * Fornece a previsao da rede neuronal de acordo com o input
     * @param inputs entradas da rede
     * @return Previsao da rede neuronal
     */
    public double predict(double[] inputs) {


        double y1 = h.predict(inputs);          //a saida do neuronio h e a terceira entrada do segundo neuronio

        double[] inputsSaida = Arrays.copyOf(inputs, inputs.length + 1);
        inputsSaida[inputsSaida.length - 1] = y1;

        return o.predict(inputsSaida);     //saida do neuronio o
    }

}



















