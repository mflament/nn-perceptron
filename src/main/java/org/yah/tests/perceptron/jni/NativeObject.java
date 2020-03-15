package org.yah.tests.perceptron.jni;

interface NativeObject {

    long reference();

    void delete();

    default void checkReference() {
        if (reference() == 0) 
            throw new IllegalStateException("Object " + this + " has been deleted");
    }
}
