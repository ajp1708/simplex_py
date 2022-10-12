from __future__ import annotations
from fractions import Fraction
from typing import List, Tuple
from numpy import ndarray
import numpy


class Tableau:
	def __init__(self, matrix: ndarray):
		# the left matrix, which was passed into this function
		self.left = matrix
		# the right-side matrix, which is an identity matrix
		# vectorize is needed to fill the matrix with fractions
		right = numpy.identity(matrix.shape[0])
		self.right = numpy.vectorize(Fraction)(right)
		# the bottom row, which is initialized with -1's and zeros
		negs = numpy.array([Fraction(-1)] * matrix.shape[1])
		zeros = numpy.array([Fraction(0)] * (matrix.shape[0] + 1))
		self.bottom = numpy.concatenate([negs, zeros])
		# The right-side of the tableau, filled with ones
		self.side = numpy.array([Fraction(1)] * matrix.shape[0])


	def __str__(self) -> str:
		pass


	def _get_row(self, row: int) -> List[Fraction]:
		#construct a row from the tableau
		if row < self.left.shape[0]:
			lst = []
			for i in range(self.left.shape[1]):
				lst.append(self.left[row, i])
			for i in range(self.right.shape[0]):
				lst.append(self.right[row, i])
			lst.append(self.side[row])
			return lst
		elif row == self.left.shape[0]:
			return list(self.bottom)


	def _get_col(self, column: int) -> List[Fraction]:
		# construct a column from the tableau
		if column < self.left.shape[1]:
			lst = []
			for i in range(self.left.shape[0]):
				lst.append(self.left[i, column])
			lst.append(self.bottom[column])
			return lst
		elif column < self.bottom.size - 1:
			lst = []
			for i in range(self.right.shape[0]):
				lst.append(self.right[i, column - self.left.shape[0]])
			lst.append(self.bottom[column])
			return lst
		elif column == self.bottom.size - 1:
			lst = list(self.side)
			lst.append(self.bottom[self.bottom.size - 1])
			return lst


	def pivot(self) -> Tuple[int, int]:
		# return the location of the pivot in the table
		# first we get the column number of the pivot
		min_value = Fraction(0)
		min_index = 0
		for i in range(self.bottom.size):
			if self.bottom[i] < min_value:
				min_value = self.bottom[i]
				min_index = i

		# then we get the row number of the pivot
		col = self._get_col(min_index)
		max_value = Fraction(10_000_000)
		max_index = -1
		for i in range(len(col) - 1):
			print(max_value, max_index, self.side[i], col[i])
			if col[i] <= 0:
				continue
			elif self.side[i] / col[i] < max_value:
				max_value = self.side[i] / col[i]
				max_index = i
		
		return (min_index, max_index)


	def next_tableau(self, pivot: Tuple[int, int]) -> Tableau:
		pass


	def done(self) -> bool:
		# checks to see if we're done simplexing yet
		return self.bottom.min() >= 0


	def row_strategy(self) -> List[Fraction]:
		# get the row player's optimal mixed strategy
		v = self.value(0)
		f = lambda x: x / v
		return f(self.side)


	def column_strategy(self) -> List[Fraction]:
		# get the column player's optimal mixed strategy
		v = self.value(0)
		f = lambda x: x / v
		return f(self.bottom[self.left.shape[0]:])


	def value(self, k: int) -> Fraction:
		# the value of the game, assuming we're done
		return self.bottom[self.bottom.size - 1] - k

