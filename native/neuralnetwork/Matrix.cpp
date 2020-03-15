#include "Matrix.h"

void createMatrix(int rows, int columns, Matrix& matrix) {
	matrix.rows = rows;
	matrix.columns = columns;
	matrix.managed = true;
	matrix.data = new double[(size_t)columns * rows];
}

void createMatrix(void* javaBuffer, Matrix& matrix) {
	matrix.rows = ((int*)javaBuffer)[0];
	matrix.columns = ((int*)javaBuffer)[1];
	matrix.managed = false;
	matrix.data = (double*)((int*)javaBuffer + 2);
}