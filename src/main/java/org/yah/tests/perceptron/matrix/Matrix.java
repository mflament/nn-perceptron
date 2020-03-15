package org.yah.tests.perceptron.matrix;

public interface Matrix<M extends Matrix<M>> {

    @FunctionalInterface
    public interface MatrixFunction {
        double apply(int row, int column, double value);
    }

    M self();

    int rows();

    int columns();

    double get(int row, int col);

    void apply(MatrixFunction func);

    void zero();

    default M sub(M b) {
        return sub(b, self());
    }

    M sub(M b, M target);

    default M mul(M b) {
        return mul(b, self());
    }

    M mul(M b, M target);

    default M mul(double s) {
        return mul(s, self());
    }

    M mul(double s, M target);

    M addColumnVector(M vector, M target);

    default M addColumnVector(M vector) {
        return addColumnVector(vector, self());
    }

    M dot(M b);

    /**
     * this . b
     * 
     * @param b
     * @param target
     */
    M dot(M b, M target);

    M transpose_dot(M b);

    /**
     * T(this) . b
     * 
     * @param b
     * @param target
     */
    M transpose_dot(M b, M target);

    /**
     * this . T(b)
     * 
     * @param b
     * @param target
     */
    M dot_transpose(M b, M target);

    M dot_transpose(M b);
    
    M sigmoid(M target);

    default M sigmoid() {
        return sigmoid(self());
    }

    M sigmoid_prime(M target);

    default M sigmoid_prime() {
        return sigmoid_prime(self());
    }

    int maxRowIndex(int column);
    

    int slide(int offset, int columns);

    M createView();

    M sumRows(M target);

    static String toString(Matrix<?> matrix) {
        StringBuilder sb = new StringBuilder();
        int rows = matrix.rows();
        int columns = matrix.columns();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                sb.append(String.format("%7.3f ", matrix.get(r, c)));
            }
            sb.append(System.lineSeparator());
        }
        return sb.toString();
    }

}
