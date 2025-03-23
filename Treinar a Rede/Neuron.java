/**
 * Classe que representa um neuronio
 * @version 2.1
 * @author Daniel Goncalves, Diogo Damasio, Goncalo Rodrigues
 */
public class Neuron {

    private double[] weights;
    private double bias;


    /**
     * Construtor de Neuron
     * @param weights pesos
     * @param bias valor do bias
     */
    public Neuron(double[] weights, double bias) {

        this.weights = weights;
        this.bias = bias;

    }


    /**
     * Getter para pesos
     * @return weights
     */
    public double[] getWeights() {
        return weights;
    }


    /**
     * Setter para pesos
     * @param weights Novos pesos
     */
    public void setWeights(double[] weights) {
        this.weights = weights;
    }


    /**
     * Define o valor de bias
     * @param bias Novo bias
     */
    public void setBias(double bias) {
        this.bias = bias;
    }


    /**
     * Getter para bias
     * @return bias
     */
    public double getBias() {
        return bias;
    }


    /**
     * Calcula a previsao do neuronio. A soma dos produtos dos inputs pelos pesos, adiciona o bias e transforma esta
     * soma atraves de uma funcao sigmoide
     * @param inputs valores de entrada
     * @return previsao do neuronio
     */
    public double predict(double[] inputs) {


        double sum = bias;                  //inicializa a soma com o bias (o mesmo que iniciar a 0 e depois somar bias)

        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];          //soma w*x
        }



        return sigmoide(sum);               //passa por sigmoide

    }


    /**
     * Calcula o valor de z na funcao sigmoide
     * @param z objeto da funcao
     * @return Valor double entre 0 e 1
     */
    private double sigmoide(double z) {

        return 1 / (1 + Math.exp(-z));

    }


}
