import java.util.Arrays;
import java.io.*;

/**
 * Classe que representa a rede neuronal com dois neuronios
 * @version 3.0
 * @author Daniel Goncalves, Diogo Damasio, Goncalo Rodrigues
 */
public class NeuralNetwork {

    private Neuron h,o;



    /**
     * Construtor da Rede Neuronal
     * @param p1 Perceptron de entrada
     * @param p2 Perceptron de saida
     */
    public NeuralNetwork(Neuron p1, Neuron p2) {

        this.h = p1;
        this.o = p2;

    }

    /**
     * Calcula a precisao da rede para um determinado conjunto de entradas
     * @param inputsTeste conjunto de entradas para teste
     * @param outputsTeste conjunto de saidas para teste
     * @return precisao da rede
     */
    public double precision(double[][] inputsTeste, double[] outputsTeste) {

        double corretos = 0;

        for (int i = 0; i < inputsTeste.length; i++) {
            double saida = Math.round(this.predict(inputsTeste[i]));
            if (Math.abs(saida - outputsTeste[i]) < 10e-9)
                corretos++;
        }

        return corretos / inputsTeste.length;

    }







    /**
     * Fornece a previsao da rede neuronal de acordo com o input
     * @param inputs valores de entrada
     * @return Previsao da rede neuronal
     */
    public double predict(double[] inputs) {


        double y1 = h.predict(inputs);          //a saida do neuronio h e a terceira entrada do segundo neuronio

        double[] inputsSaida = Arrays.copyOf(inputs,inputs.length+1);
        inputsSaida[inputsSaida.length-1] = y1;

        return o.predict(inputsSaida);     //saida do neuronio o
    }

















    /**
     * Treina a rede neuronal atraves do algoritmo de backpropagation na sua versao exata
     * @param inputsTreino Conjunto de entradas de treino
     * @param outputsTreino Saidas esperadas do conjunto de treino
     * @param inputsTeste Conjunto de entradas de teste
     * @param outputsTeste Saidas esperadas do conjunto de teste
     * @throws IOException erro no ficheiro
     */
    public void train(double[][] inputsTreino, double[] outputsTreino, double[][] inputsTeste, double[] outputsTeste) throws IOException {


        double learningRate = 0.3;      //taxa de aprendizagem

        int epoch = 0; //iteracao
        double mseTreino; //mean squared error treino

        double mseTeste = 0; //mse do teste
        double minMSETeste = Double.MAX_VALUE;          //valor minimo do mse de teste
        double mseTesteAnterior = Double.MAX_VALUE;     //valor do mse da epoca anterior


        double[] melhoresPesosH = new double[h.getWeights().length];      //pesos para a melhor epoca (mse de treino minimo)
        double[] melhoresPesosO = new double[o.getWeights().length];

        double melhorBiasH = 0;
        double melhorBiasO = 0;

        int tolerancia = 0;


        BufferedWriter writer = new BufferedWriter(new FileWriter("resultados.csv"));

        //ciclo infinito ate atingir condicao de paragem
        while(true) {

            epoch++;
            double erroTotal = 0;


            for (int i = 0; i < inputsTreino.length; i++) {


                double[] entrada = inputsTreino[i];
                double objetivo = outputsTreino[i];

                //calcula a saida da rede
                double saida = this.predict(entrada);



                //calcula erro^2 e incrementa o erro total
                erroTotal += calculaErro(saida,objetivo);


                //calcula delta do neuronio de saida
                double deltaO = (saida-objetivo) * dSigmoide(saida);

                //calcula h (saida do neuronio oculto)
                double saidaH = h.predict(entrada);

                //calcula delta do neuronio oculto
                double deltaH = deltaO * o.getWeights()[400] * dSigmoide(saidaH);           //peso 400 corresponde ao peso associado a saida do neuronio oculto


                for (int j = 0; j < h.getWeights().length; j++) {
                    h.getWeights()[j] -= learningRate * deltaH * entrada[j];
                }
                h.setBias(h.getBias() - learningRate * deltaH);



                for (int k = 0; k < o.getWeights().length; k++) {
                    if (k == 400) {//se for o do neuronio oculto para saida
                        o.getWeights()[400] -= learningRate * deltaO * saidaH;
                    } else {
                        o.getWeights()[k] -= learningRate * deltaO * entrada[k];
                    }
                }
                o.setBias(o.getBias() - learningRate * deltaO);
            }


            //calcula o erro quadratico medio
            mseTreino = erroTotal / inputsTreino.length;












            //aqui comeca o early stopping
            double erroTotalTeste = 0;


            for (int i = 0; i < inputsTeste.length; i++) {      //percorre todos os elementos do conjunto de teste


                double[] entrada = inputsTeste[i];
                double objetivo = outputsTeste[i];

                //calcula a saida da rede
                double saida = this.predict(entrada);


                //calcula erro^2 e incrementa o erro total
                erroTotalTeste += calculaErro(saida, objetivo);

            }


            mseTeste = erroTotalTeste / inputsTeste.length;


            if (mseTeste > mseTesteAnterior) {          //se mse for maior que anterior incrementa tolerancia
                tolerancia++;
            } else {                                    //se nao for, tolerancia volta a 0
                tolerancia = 0;

                if (mseTeste < minMSETeste) {            //caso seja o menor mse ja obtido

                    System.arraycopy(h.getWeights(), 0, melhoresPesosH, 0, h.getWeights().length); //guarda melhores pesos
                    System.arraycopy(o.getWeights(), 0, melhoresPesosO, 0, o.getWeights().length);
                    melhorBiasH = h.getBias();
                    melhorBiasO = o.getBias();

                    minMSETeste = mseTeste;             //atualiza o minimo
                }
            }

            mseTesteAnterior = mseTeste;













            // Exibe progresso a cada 100 épocas
            if (epoch % 100 == 0) {
                System.out.println("Epoca: " + epoch + " | MSE (treino): " + mseTreino + " | MSE (teste): " + mseTeste);
            }


            //escreve para ficheiro
            writer.write(epoch + ";" + mseTreino + ";" + mseTeste);
            writer.newLine();
            writer.flush();



            //condicao de paragem,se o erro quadratico medio for menor que 0.00001 ou mseT comecar a subir quebra o ciclo
            if (mseTreino <= 0.00001 || tolerancia == 10) {

                h.setWeights(melhoresPesosH);
                o.setWeights(melhoresPesosO);
                h.setBias(melhorBiasH);
                o.setBias(melhorBiasO);



                System.out.print("Treino concluído na época ");
                if(tolerancia == 10) System.out.println(epoch-10 + " (tolerancia)");
                else System.out.println(epoch);


                System.out.println("MSE (treino): " + mseTreino + ", MSE (teste): " + minMSETeste);
                break;
            }
        }
    }


    /**
     * Calcula o erro quadratico medio para um input
     * @param saida Saida obtida pela rede
     * @param objetivo Saida esperada
     * @return MSE
     */
    private double calculaErro(double saida, double objetivo) {
        return 0.5 * Math.pow(saida - objetivo, 2);
    }




    /**
     * Funcao derivada da funcao sigmoide
     * @param x Objeto da funcao
     * @return sig'(x)
     */
    private double dSigmoide(double x) {
        return x * (1-x);
    }

}
