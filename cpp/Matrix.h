#pragma once

struct Matrix {
	int rows = 0;
	int columns = 0;

	int offset = 0;
	int maxColumns = 0;

	bool managed = false;
	double* data = 0;

	void create(int _rows, int _columns);
	void set(int _rows, int _columns, double* _data);

	int slide(int offset, int _columns);

	inline double get(int row, int col) {
		return *(column(col) + row);
	}

	inline void set(int row, int col, double value) {
		column(col)[row] = value;
	}

	inline double* column(int col) {
		return data + (((size_t)offset + col) * rows);
	}

	void sub(Matrix& m);
	void mul(double s);
	void zero();

	int maxRowIndex(int col);

	void free();
};