package org.yah.tests.perceptron.matrix.nd;

import java.lang.reflect.Array;
import java.util.Arrays;

import static java.lang.System.arraycopy;
import static java.lang.reflect.Array.getLength;
import static java.util.Objects.requireNonNull;

public class NDMatrix {

    private final int[] lengths;
    private final int[] defaultStrides;
    private final int[] defaultOffsets;

    private Object data;

    public static NDMatrix withDimensions(int dimensions) {
        return new NDMatrix(new int[dimensions]);
    }

    public static NDMatrix withLengths(int... lengths) {
        return new NDMatrix(lengths);
    }

    public static NDMatrix withCapacities(int... capacities) {
        return new NDMatrix(capacities, Array.newInstance(Double.TYPE, capacities));
    }

    private NDMatrix(int[] lengths) {
        this(lengths, null);
    }

    private NDMatrix(int[] lengths, Object data) {
        this.lengths = new int[lengths.length];
        arraycopy(lengths, 0, this.lengths, 0, lengths.length);

        this.defaultStrides = new int[dimensions()];
        Arrays.fill(defaultStrides, 1);
        this.defaultOffsets = new int[dimensions()];

        this.data = data;
    }

    public int dimensions() {
        return lengths.length;
    }

    public int length(int index) {
        return lengths[index];
    }

    public int getTotalElements() {
        int res = 1;
        for (int length : lengths) {
            res *= length;
        }
        return res;
    }

    public int getTotalCapacity() {
        return getTotalCapacity(data);
    }

    private int getTotalCapacity(Object array) {
        if (array == null)
            return 0;
        Class<?> componentType = array.getClass().getComponentType();
        if (componentType == Double.TYPE)
            return getLength(array);

        int lenght = getLength(array);
        int res = 0;
        for (int i = 0; i < lenght; i++) {
            res += getTotalCapacity(Array.get(array, i));
        }
        return res;
    }

    public double get(int... indices) {
        Object array = data;
        for (int d = 0; d < indices.length; d++) {
            int index = indices[d];
            if (index < lengths[d]) {
                if (array != null && index < getLength(array)) {
                    if (d < dimensions() - 1)
                        array = Array.get(array, index);
                } else {
                    array = null;
                }
            } else {
                throw new ArrayIndexOutOfBoundsException(String.format("index %d is out of dimension %d length (%d)", index, d, length(d)));
            }
        }
        if (array == null) return 0;
        return Array.getDouble(array, indices[indices.length - 1]);
    }

    public void set(double v, int... indices) {
        Object vector = getOrCreateVector(indices);
        Array.setDouble(vector, indices[indices.length - 1], v);
    }

    public void visit(Visitor visitor) {
        visit(defaultStrides, defaultOffsets, visitor);
    }

    public void visit(int[] strides, int[] offsets, Visitor visitor) {
        if (requireNonNull(strides).length != dimensions())
            throw new IllegalArgumentException("Invalid strides length");
        if (requireNonNull(offsets).length != dimensions())
            throw new IllegalArgumentException("Invalid offsets length");
        int dim = 0;
        int[] indices = new int[dimensions()];
        indices[0] = offsets[0];
        Object array = data;
        do {
            int index = indices[dim];
            if (index < lengths[dim]) {
                if (dim < dimensions() - 1) {
                    array = array != null && index < getLength(array) ? Array.get(array, index) : null;
                    indices[++dim] = offsets[dim];
                } else {
                    double v = array != null && index < getLength(array) ? Array.getDouble(array, index) : 0;
                    visitor.visit(indices, v);
                    indices[dim] += strides[dim] == 0 ? lengths[dim] : strides[dim];
                }
            } else if (dim == 0) {
                break;
            } else {
                indices[--dim] += strides[dim] == 0 ? lengths[dim] : strides[dim];
                array = getArray(indices, dim);
            }
        } while (true);
    }

    @SuppressWarnings("SuspiciousSystemArraycopy")
    private Object getOrCreateVector(int[] indices) {
        if (indices.length != dimensions())
            throw new IllegalArgumentException("Indices length " + indices.length + " does not match dimensions " + dimensions());
        Object parent = null, array = data;
        for (int d = 0; d < indices.length; d++) {
            int index = indices[d];
            int size = index + 1;
            if (array == null || getLength(array) < size) {
                Object newArray = Array.newInstance(d == dimensions() - 1 ? Double.TYPE : Object.class, size);
                if (array != null)
                    System.arraycopy(array, 0, newArray, 0, getLength(array));
                if (parent == null)
                    data = newArray;
                else
                    Array.set(parent, indices[d - 1], newArray);
                array = newArray;
                lengths[d] = Math.max(size, lengths[d]);
            }
            if (d < indices.length - 1) {
                parent = array;
                array = Array.get(parent, index);
            }
        }
        return array;
    }

    private Object getArray(int[] indices, int dim) {
        Object res = data;
        for (int d = 0; d < dim; d++) {
            int index = indices[d];
            if (res != null && index < getLength(res))
                res = Array.get(res, index);
            else
                return null;
        }
        return res;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("NDMatrix");
        for (int length : lengths) {
            sb.append("[").append(length).append("]");
        }
        return sb.toString();
    }

    public interface Visitor {
        void visit(int[] indices, double value);
    }

//    @Override
//    public String toString() {
//        StringBuilder sb = new StringBuilder();
//        int dim = 0;
//        int[] indices = new int[dimensions()];
//        int headerDim = Math.max(0, dimensions() - 2);
//        do {
//            if (indices[dim] < lengths[dim]) {
//                if (dim == headerDim && indices[dim] == 0) {
//                    sb.append(name);
//                    for (int d = 0; d < dim; d++) {
//                        sb.append("[").append(indices[d]).append("]");
//                    }
//                    sb.append(System.lineSeparator());
//                }
//                if (dim < dimensions() - 1) {
//                    indices[++dim] = 0;
//                } else {
//                    sb.append(String.format(Locale.ENGLISH, "%7.3f ", data[offset(indices)]));
//                    indices[dim]++;
//                }
//            } else if (dim == 0) {
//                break;
//            } else {
//                if (dim == dimensions() - 1)
//                    sb.append(System.lineSeparator());
//                indices[--dim]++;
//            }
//        } while (true);
//        return sb.toString();
//    }

//    public void add(Matrix mx) {
//        if (mx.dimensions() != dimensions())
//            throw new IllegalArgumentException("Dimensions mismatch");
//        for (int i = 0; i < dimensions(); i++) {
//            lengths[i] = Math.max(lengths[i], mx.lengths[i]);
//        }
//        System.arraycopy(mx.data, 0, data, 0, data.length);
//    }


}
