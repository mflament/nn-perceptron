#include "Matrix.h"
#include <algorithm>

void Matrix::create(int _rows, int _columns) {
	rows = _rows;
	columns = capacity = _columns;
	data = new double[(size_t)_columns * _rows];
	managed = true;
	zero();
}

int Matrix::slide(int _offset, int _columns) {
	offset = _offset;
	columns = std::min(_columns, capacity - offset);
	return columns;
}

void Matrix::zero() {
	memset(data + offset, 0, sizeof(double) * columns * rows);
}

void Matrix::sub(Matrix& m) {
	for (int c = 0; c < columns; c++)
	{
		double* col = column(c);
		double* mcol = m.column(c);
		for (int r = 0; r < rows; r++)
		{
			col[r] -= mcol[r];
		}
	}
}

void Matrix::mul(double s) {
	for (int c = 0; c < columns; c++)
	{
		double* col = column(c);
		for (int r = 0; r < rows; r++)
		{
			col[r] *= s;
		}
	}
}

int Matrix::maxRowIndex(int c) {
	int res = -1;
	double max = -DBL_MAX;
	double* col = column(c);
	for (int r = 0; r < rows; r++) {
		if (col[r] > max) {
			max = col[r];
			res = r;
		}
	}
	return res;
}

void Matrix::free() {
	rows = columns = capacity = 0;
	if (managed) {
		delete[]data;
		managed = false;
	}
	data = 0;
}