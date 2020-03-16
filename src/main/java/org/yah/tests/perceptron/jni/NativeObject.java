package org.yah.tests.perceptron.jni;

abstract class NativeObject implements AutoCloseable {

    protected long reference;

    @Override
    public void close() {
        if (reference != 0) {
            delete(reference);
            reference = 0;
        }
    }

    protected abstract void delete(long reference);

    protected void checkReference() {
        if (reference == 0)
            throw new IllegalStateException("Object " + this + " has been deleted");
    }
}
