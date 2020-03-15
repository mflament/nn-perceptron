#pragma once

struct Matrix {
	int rows;
	int columns;
	bool managed;
	double* data;
};

void createMatrix(int rows, int columns, Matrix& matrix);
void createMatrix(void* javaBuffer, Matrix& matrix);