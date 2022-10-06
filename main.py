from __future__ import annotations
from fractions import Fraction
from typing import List, Tuple
from numpy import matrix


def identity_matrix(size: int) -> matrix:
	pass


class Tableau:
	def __init__(self, mat: matrix):
		pass


	def __eq__(self, other: Tableau) -> bool:
		pass


	def __hash__(self) -> int:
		pass


	def __str__(self) -> str:
		pass


	def pivot(self) -> Tuple[int, int]:
		# return the location of the pivot in the table
		pass


	def next_tableau(self, pivot: Tuple[int, int]) -> Tableau:
		pass


	def done(self) -> bool:
		pass


	def row_strategy(self) -> List[Fraction]:
		pass


	def column_strategy(self) -> List[Fraction]:
		pass


	def value(self, k: int) -> Fraction:
		pass

