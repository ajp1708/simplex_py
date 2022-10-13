from __future__ import annotations
from fractions import Fraction
from os import read
from typing import List, Tuple
from numpy import ndarray
from prompt_toolkit import prompt
from tabulate import tabulate
import numpy
import math


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
		# The row numbers for x1, x2, etc
		self.row_names = []


	def __str__(self) -> str:
		# we'll build a table to use the tabulate module
		table = []
		# we'll first make the header row
		line = [""]
		for i in range(self.left.shape[1]):
			line.append(f"x{i}")
		for i in range(self.left.shape[0]):
			line.append(f"s{i}")
		line.append("")
		table.append(line)
		# then we make all the other rows
		for row in range(self.left.shape[0] + 1):
			line = []
			line.append(self._row_name(row))
			row = self._get_row(row)
			for value in row:
				line.append(value)
			table.append(line)
		return tabulate(table, headers="firstrow")


	def _row_name(self, row: int) -> str:
		# check if it's an x variable
		for i in range(len(self.row_names)):
			if self.row_names[i] == row:
				return f"x{i}"
		# check if it's the bottom row
		if row == self.left.shape[0]: return ""
		# otherwise it's slack
		return f"s{row}"


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

def main():
	print("Welcome to the Super Awesome and Amazing Simplex Method Program\n")
	
	row = 0
	col = 0

	while(row < 1 or col < 1):
		try:
			# might cause the vals of row and col to be prematurely changed
			rows = int(input("How many rows are in the matrix:"))
			cols = int(input("How many columns are in the matrix"))
			if(row < 1 or col < 1):
				print("Invalid dimensions entered, the dimensions need to be an integer greater than 1")

		except:
			print("Invalid dimensions entered, the dimensions need to be an integer greater than 1")

	matrix = numpy.ndarray(shape=(row,col), dtype=Fraction)
	rowsRead = 0

	while(rowsRead < rows):
		rowVals = prompt("Enter the values in row" + str(x+1) + " (values must be space separated")
		splitVals = rowVals.split(" ")
		readErrorFlag = False

		# checks if enough values were read in to ensure rows and cols of the matrix are properly represented
		if(len(splitVals) < col):
			print("Not enough values entered for the row")
			continue
		
		row = numpy.array()

		# trys to add all the values entered into an array, which will eventually be put into the matrix
		for val in rowVals:
			try:
				numpy.append(row, int(val))
			except:
				print("Failed reading the vals in the row. Make sure all the values are space separated and integers")
				readErrorFlag = True
				break
		if not readErrorFlag:
			matrix[rowsRead] = row
			rowsread += 1

	print(Tableau(numpy.array([[1, 2], [3, 4]])))

main()