package org.yah.tests.perceptron.matrix.nd;

import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;
import static org.yah.tests.perceptron.matrix.nd.NDMatrix.*;

@SuppressWarnings("PointlessArithmeticExpression")
public class NDMatrixTest {

    @Test
    public void dimensions() {
        NDMatrix matrix = withDimensions(3);
        assertEquals(3, matrix.dimensions());

        matrix = withLengths(10, 2);
        assertEquals(2, matrix.dimensions());
    }

    @Test
    public void getTotalCapacity() {
        NDMatrix matrix = withDimensions(3);
        assertEquals(0, matrix.getTotalCapacity());

        matrix = withLengths(2, 3, 4);
        assertEquals(0, matrix.getTotalCapacity());

        matrix = withCapacities(2, 2, 2);
        assertEquals(2 * 2 * 2, matrix.getTotalCapacity());

        matrix = withLengths(2, 3, 4);
        matrix.set(1.0, 1, 0, 1);
        assertEquals(1 * 1 * 2, matrix.getTotalCapacity());
    }

    @Test
    public void getTotalElements() {
        NDMatrix matrix = withDimensions(3);
        assertEquals(0, matrix.getTotalElements());

        matrix = withLengths(2, 3, 4);
        assertEquals(2 * 3 * 4, matrix.getTotalElements());
    }

    @Test
    public void length() {
        NDMatrix matrix = withDimensions(3);
        assertEquals(0, matrix.length(0));
        assertEquals(0, matrix.length(1));
        assertEquals(0, matrix.length(2));

        matrix = withLengths(10, 2);
        assertEquals(10, matrix.length(0));
        assertEquals(2, matrix.length(1));
    }

    @Test
    public void get() {
        NDMatrix matrix = withLengths(10, 2);
        assertEquals(0, matrix.get(0, 0), 0);
        assertEquals(0, matrix.get(9, 1), 0);
    }

    @Test(expected = ArrayIndexOutOfBoundsException.class)
    public void get_OutOfBound() {
        NDMatrix matrix = withDimensions(3);
        matrix.get(0, 0, 0);
    }

    @Test
    public void set() {
        NDMatrix matrix = withDimensions(3);
        matrix.set(1, 0, 0, 0);
        assertEquals(1, matrix.length(0));
        assertEquals(1, matrix.length(1));
        assertEquals(1, matrix.length(2));
        assertEquals(1, matrix.get(0, 0, 0), 0);

        matrix.set(2, 1, 0, 4);
        assertEquals(2, matrix.length(0));
        assertEquals(1, matrix.length(1));
        assertEquals(5, matrix.length(2));
        assertEquals(1, matrix.get(0, 0, 0), 0);
        assertEquals(2, matrix.get(1, 0, 4), 0);
        assertEquals(0, matrix.get(0, 0, 4), 0);
        assertEquals(0, matrix.get(0, 0, 3), 0);
        assertEquals(0, matrix.get(1, 0, 0), 0);
        try {
            matrix.get(0, 1, 0);
            fail("Shoudl be out of bound");
        } catch (ArrayIndexOutOfBoundsException ignored) {
        }
    }

    @Test
    public void visit() {
        NDMatrix matrix = withDimensions(1);
        MockVisitor visitor = new MockVisitor(0);
        matrix.visit(visitor);

        matrix.set(1, 0);
        visitor = new MockVisitor(1);
        visitor.expect(1, 0);
        matrix.visit(visitor);
        visitor.assertCount();

        matrix = withLengths(4, 2, 3);
        matrix.set(5, 3, 1, 2);
        visitor = new MockVisitor(4 * 3 * 2);
        visitor.expect(0, 0, 0, 0)
                .expect(0, 0, 0, 2)
                .expect(0, 1, 1, 0)
                .expect(0, 1, 1, 1)
                .expect(5, 3, 1, 2);
        matrix.visit(visitor);
        visitor.assertCount();
    }

    @Test
    public void visit_vector() {
        NDMatrix matrix = withLengths(2, 2);
        matrix.set(1, 0, 0);
        matrix.set(2, 0, 1);
        matrix.set(3, 1, 0);
        matrix.set(4, 1, 1);

        MockVisitor visitor = new MockVisitor(2);
        visitor.expect(1, 0, 0);
        visitor.expect(2, 0, 0);
        matrix.visit(new int[]{0, 1}, new int[]{0, 0}, visitor);
        visitor.assertCount();

        visitor = new MockVisitor(2);
        visitor.expect(3, 1, 0);
        visitor.expect(4, 1, 0);
        matrix.visit(new int[]{0, 1}, new int[]{1, 0}, visitor);
        visitor.assertCount();

        visitor = new MockVisitor(2);
        visitor.expect(1, 0, 0);
        visitor.expect(3, 1, 0);
        matrix.visit(new int[]{1, 0}, new int[]{0, 0}, visitor);
        visitor.assertCount();

        visitor = new MockVisitor(2);
        visitor.expect(2, 0, 1);
        visitor.expect(4, 1, 1);
        matrix.visit(new int[]{1, 0}, new int[]{0, 1}, visitor);
        visitor.assertCount();
    }

    private static class MockVisitor implements NDMatrix.Visitor {
        private static class Expected {
            final int[] indices;
            final double value;

            public Expected(int[] indices, double value) {
                this.indices = indices;
                this.value = value;
            }
        }

        private final List<Expected> expecteds = new ArrayList<>();

        private final int expectedCount;

        private int visitedCount;

        public MockVisitor(int expectedCount) {
            this.expectedCount = expectedCount;
        }

        public void assertCount() {
            assertEquals(visitedCount, expectedCount);
        }

        public MockVisitor expect(double value, int... indices) {
            expecteds.add(new Expected(indices, value));
            return this;
        }

        private Optional<Expected> findExpected(int[] indices) {
            return expecteds.stream()
                    .filter(e -> Arrays.equals(e.indices, indices))
                    .findFirst();
        }

        @Override
        public void visit(int[] indices, double value) {
            if (visitedCount < expectedCount) {
                findExpected(indices).ifPresent(e -> assertEquals("Mismatched value at " + Arrays.toString(indices), e.value, value, 0));
                visitedCount++;
            } else fail("Visiting " + (visitedCount + 1) + " but " + expectedCount + " were expected");
        }
    }
}