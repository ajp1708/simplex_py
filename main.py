from __future__ import annotations
from fractions import Fraction
from typing import List, Tuple
from numpy import ndarray
import numpy


class Tableau:
	def __init__(self, matrix: ndarray):
		self.left = matrix
		right = numpy.identity(matrix.shape[0])
		self.right = numpy.vectorize(Fraction)(right)
		negs = numpy.array([Fraction(-1)] * matrix.shape[1])
		zeros = numpy.array([Fraction(0)] * matrix.shape[0])
		self.bottom = numpy.concatenate([negs, zeros])
		self.side = numpy.array([Fraction(1)] * matrix.shape[0])


	def __str__(self) -> str:
		pass


	def pivot(self) -> Tuple[int, int]:
		# return the location of the pivot in the table
		pass


	def next_tableau(self, pivot: Tuple[int, int]) -> Tableau:
		pass


	def done(self) -> bool:
		return self.bottom.min() >= 0


	def row_strategy(self) -> List[Fraction]:
		v = self.value(0)
		f = lambda x: x / v
		return f(self.side)


	def column_strategy(self) -> List[Fraction]:
		v = self.value(0)
		f = lambda x: x / v
		return f(self.bottom[self.left.shape[0]:])


	def value(self, k: int) -> Fraction:
		return self.bottom[self.bottom.size - 1] - k
