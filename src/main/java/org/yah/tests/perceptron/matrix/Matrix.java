package org.yah.tests.perceptron.matrix;

public interface Matrix<M extends Matrix<M>> {

    M self();

    int rows();

    int columns();

    double get(int row, int col);

    void set(int row, int col, double value);

    void apply(MatrixFunction func);

    default void sub(M b) {
        sub(b, self());
    }

    void sub(M b, M target);

    default void mul(M b) {
        mul(b, self());
    }

    void mul(M b, M target);

    default M mul(double s) {
        return mul(s, self());
    }

    M mul(double s, M target);

    void addColumnVector(M vector, M target);

    default void addColumnVector(M vector) {
        addColumnVector(vector, self());
    }

    M dot(M b);

    /**
     * this . b
     * 
     * @param b
     * @param target
     * @noinspection JavaDoc
     */
    M dot(M b, M target);

    M transpose_dot(M b);

    /**
     * T(this) . b
     * 
     * @param b
     * @param target
     * @noinspection JavaDoc
     */
    M transpose_dot(M b, M target);

    /**
     * this . T(b)
     * 
     * @param b
     * @param target
     * @noinspection JavaDoc
     */
    M dot_transpose(M b, M target);

    M dot_transpose(M b);
    
    M sigmoid(M target);

    default M sigmoid() {
        return sigmoid(self());
    }

    void sigmoid_prime(M target);

    default void sigmoid_prime() {
        sigmoid_prime(self());
    }

    int maxRowIndex(int column);

    int slide(int offset, int columns);

    M createView();

    void sumRows(M target);

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
