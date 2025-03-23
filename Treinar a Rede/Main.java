import java.io.*;
import java.io.IOException;
import java.util.Arrays;


/**
 * Classe responsavel pela interacao com o utilizador
 * @version 1.0
 * @author Daniel Goncalves, Diogo Damasio, Goncalo Rodrigues
 */
public class Main {
    /**
     * Fluxo principal do programa
     * @param args Argumentos da linha de comandos
     * @throws IOException erro no ficheiro
     */
    public static void main(String[] args) throws IOException {

        double[][] inputsTreino = null;
        double[] outputsTreino = null;
        double[][] inputsTeste= null;
        double[] outputsTeste = null;

        try {

            inputsTreino = loadInputs("src/dataset.csv", 640,"treino"); // Arquivo CSV com os pixeis das imagens
            outputsTreino = loadOutputs("src/labels.csv",640,"treino"); // Arquivo CSV com os rotulos (0 ou 1)
            inputsTeste = loadInputs("src/dataset.csv", 160,"teste");
            outputsTeste = loadOutputs("src/labels.csv", 160, "teste");

        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        // Aplica data augmentation ao conjunto de treino
        double[][] augmentedInputsTreino = augmentData(inputsTreino);
        double[] augmentedOutputsTreino = new double[outputsTreino.length * 2];

        // Duplica os outputs
        System.arraycopy(outputsTreino, 0, augmentedOutputsTreino, 0, outputsTreino.length);
        System.arraycopy(outputsTreino, 0, augmentedOutputsTreino, outputsTreino.length, outputsTreino.length);

        // Usa os dados aumentados no treino
        inputsTreino = augmentedInputsTreino;
        outputsTreino = augmentedOutputsTreino;


        double[] weights1 = new double[400];

        //inicializa pesos e biases entre 0 e 0.01
        for(int i=0;i<400;i++)
        {
            weights1[i]=Math.random() * 0.01;
        }
        double bias1 = Math.random() * 0.01;



        double[] weights2= new double[401];

        for(int i=0;i<401;i++)
        {
            weights2[i]= Math.random() * 0.01;
        }
        double bias2 = Math.random() * 0.01;


        Neuron h = new Neuron(weights1, bias1);
        Neuron o = new Neuron(weights2, bias2);


        NeuralNetwork nn = new NeuralNetwork(h,o);

        nn.train(inputsTreino, outputsTreino, inputsTeste, outputsTeste);   //treina a rede

        System.out.printf("Precisão: %.2f%%\n", nn.precision(inputsTeste,outputsTeste) * 100);  //calcula precisao

        //imprime pesos e biases
        System.out.println("Pesos H\n");

        for(double p : h.getWeights()) {
            System.out.print(p + " , ");
        }


        System.out.println("\n\nBias H\n");
        System.out.println(h.getBias());



        System.out.println("\nPesos O\n");

        for(double p : o.getWeights()) {
            System.out.print(p + " , ");
        }

        System.out.println("\n\nBias O\n");
        System.out.println(o.getBias());




    }


    /**
     * Permite ler os pixeis das imagens
     * @param filePath Ficheiro onde estao as imagens
     * @param numeroLinhas Numero de linhas a ler (imagens)
     * @param modo treino ou teste, dependendo a que se destinam as imagens
     * @return array de imagens
     * @throws IOException erro no ficheiro
     */
    private static double[][] loadInputs(String filePath, int numeroLinhas, String modo) throws IOException {

        BufferedReader br = new BufferedReader(new FileReader(filePath));
        String line;

        double[][] inputs = new double[numeroLinhas][400];
        int linhaInicial = 0;



        //se for dados para teste
        if (modo.equals("teste")) {

            // Contar o número total de linhas no arquivo
            int totalLinhas = 0;
            while (br.readLine() != null) {
                totalLinhas++;
            }


            // Fechar e reabrir o BufferedReader para reseta-lo
            br.close();
            br = new BufferedReader(new FileReader(filePath));



            // Definir o ponto de início para leitura no modo "teste". se for treino linha inicial = 0
            linhaInicial = totalLinhas - numeroLinhas;
        }





        // Pular ate a linha inicial (se teste) se treino linha inicial = 0 nao faz nada
        for (int i = 0; i < linhaInicial; i++) {
            br.readLine(); // Ignora as linhas até chegar a inicial
        }





        // Ler as proximas linhas e processar
        int sampleIndex = 0;
        while (sampleIndex < numeroLinhas) {
            line = br.readLine();

            String[] pixelValues = line.split(",");

            for (int i = 0; i < 400; i++) {
                double numero = normalizeInput(pixelValues[i]);
                inputs[sampleIndex][i] = numero;
            }

            sampleIndex++;
        }

        br.close();
        return inputs;
    }


    /**
     * Permite ler as labels das imagens
     * @param filePath Ficheiro onde estao as labels
     * @param numeroLinhas Numero de linhas a ler
     * @param modo treino ou teste, dependendo a que se destinam as labels
     * @return array de labels
     * @throws IOException erro no ficheiro
     */
    private static double[] loadOutputs(String filePath, int numeroLinhas, String modo) throws IOException {

        BufferedReader br = new BufferedReader(new FileReader(filePath));
        String line;

        double[] outputs = new double[numeroLinhas];
        int linhaInicial = 0;



        if (modo.equals("teste")) {

            // Contar o número total de linhas no arquivo
            int totalLinhas = 0;
            while (br.readLine() != null) {
                totalLinhas++;
            }


            // Fechar e reabrir o BufferedReader para resetá-lo
            br.close();
            br = new BufferedReader(new FileReader(filePath));



            // Definir o ponto de inicio para leitura no modo "teste". se for treino linha inicial = 0
            linhaInicial = totalLinhas - numeroLinhas;
        }





        // Pular ate a linha inicial (se teste) se treino linha inicial = 0 nao faz nada
        for (int i = 0; i < linhaInicial; i++) {
            br.readLine(); // Ignora as linhas até chegar na inicial
        }



        // Ler as proximas linhas e processar
        int sampleIndex = 0;
        while (sampleIndex < numeroLinhas) {
            line = br.readLine();
            outputs[sampleIndex++] = Double.parseDouble(line);
        }

        br.close();
        return outputs;
    }


    /**
     * Normaliza um input entre 0 e 1
     * @param input valor do pixel
     * @return [0,1]
     */
    private static double normalizeInput(String input) {
        double numero = Double.parseDouble(input);

        if (numero < 0) return 0;
        else if (numero > 1) return 1;
        else return numero;
    }

    /**
     * Funcao de data augmentation
     * @param inputs dataset
     * @return novo dataset apos augmentation
     */
    private static double[][] augmentData(double[][] inputs) {
        int originalSize = inputs.length;
        int augmentedSize = originalSize * 2;
        double[][] augmentedInputs = new double[augmentedSize][400];

        // Copia os originais
        System.arraycopy(inputs, 0, augmentedInputs, 0, originalSize);

        // Cria amostras
        for (int i = 0; i < originalSize; i++) {
            double[] original = inputs[i];
            double[] augmented = augmentSample(original);
            augmentedInputs[originalSize + i] = augmented;
        }

        return augmentedInputs;
    }




    /**
     * Aplica data augmentation a uma amostra.
     * @param sample imagem original
     * @return nova imagem
     */
    private static double[] augmentSample(double[] sample) {
        double[] augmented = Arrays.copyOf(sample, sample.length);

        augmented = rotateImage90Degrees(augmented, 20, 20); // Rotaciona 90 graus

        return augmented;
    }


    /**
     * Roda uma imagem
     * @param original imagem original
     * @param width largura
     * @param height altura
     * @return imagem rodada
     */
    private static double[] rotateImage90Degrees(double[] original, int width, int height) {
        // Converte o array 1D em 2D
        double[][] matrix = new double[height][width];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                matrix[i][j] = original[i * width + j];
            }
        }

        // Cria uma nova matriz para armazenar a imagem rodada
        double[][] rotated = new double[width][height];

        // Rotação 90 graus sentido horário:
        // pixel (i,j) -> (j, height - 1 - i)
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                rotated[j][height - 1 - i] = matrix[i][j];
            }
        }

        // Converte de volta para array
        double[] rotatedArray = new double[width * height];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                rotatedArray[i * height + j] = rotated[i][j];
            }
        }

        return rotatedArray;
    }




}
