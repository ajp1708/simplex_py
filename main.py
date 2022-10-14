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
				line.append(str(value))
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


	def pivot(self) -> Tuple[int, int, int]:
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
			# print(max_value, max_index, self.side[i], col[i])
			if col[i] <= 0:
				continue
			elif self.side[i] / col[i] < max_value:
				max_value = self.side[i] / col[i]
				max_index = i

		return (min_index, max_index, col[max_index])


	def next_tableau(self, pivot: Tuple[int, int, int]) -> Tableau:
		self.row_names.append(pivot[1])
		pivot_row_new = self._get_row(pivot[1])
		# calculate the new values for every value in the pivot row
		for x in range(len(pivot_row_new)):
			pivot_row_new[x] = pivot_row_new[x] / pivot[2]
		# set the new values for every entry in the pivot row
		for i in range(self.left.shape[1]):
			self.left[pivot[1], i] = pivot_row_new[i]
		for i in range(self.right.shape[0]):
			self.right[pivot[1], i] = pivot_row_new[i + self.left.shape[1]]
		self.side[pivot[1]] = pivot_row_new[len(pivot_row_new) - 1]

		# loops through all the other rows and changes their values
		for x in range(self.left.shape[0] + 1):
			if not x == pivot[1]:
				new_row = self._get_row(x)
				pivot_row_val = self._get_col(pivot[0])[x]
				pivot_row_subtract = pivot_row_new.copy()
				for i in range(len(pivot_row_new)):
					pivot_row_subtract[i] = pivot_row_subtract[i] * pivot_row_val
				new_row = numpy.subtract(new_row, pivot_row_subtract)
				if x == self.left.shape[0]:
					self.bottom = new_row
				else:
					for i in range(self.left.shape[1]):
						self.left[x, i] = new_row[i]
					for i in range(self.right.shape[0]):
						self.right[x, i] = new_row[i + 1 + self.left.shape[0]]
					self.side[x] = new_row[len(new_row) - 1]

		return self
		


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

def print_fraction_array(fraction_array) -> str:
	temp = ""
	for x in fraction_array:
		temp += str(x) + " "
	return temp

def print_fraction_matrix(fraction_matrix):
	for x in fraction_matrix:
		print_fraction_array(x)

# need to figure out a way to print fractions as actual fractions not Fraction()
def main():
	print("Welcome to the Super Awesome and Amazing Simplex Method Program\n")
	
	rows = 0
	cols = 0

	# checks if the dimensions entered are valid
	while rows < 1 or cols < 1:
		try:
			rows = int(input("How many rows are in the matrix: "))
			cols = int(input("How many columns are in the matrix: "))
			if(rows < 1 or cols < 1):
				print("Invalid dimensions entered, the dimensions need to be an integer greater than 1")

		except:
			print("Invalid dimensions entered, the dimensions need to be an integer greater than 1")

	matrix = numpy.empty((rows,cols),dtype=Fraction)
	rows_read = 0

	# goes until all the values for each row are read in
	while rows_read < rows:
		row_vals = prompt("Enter the values in row " + str(rows_read+1) + " (values must be space separated): ")
		split_vals = row_vals.split(" ")
		read_error_flag = False

		# checks if enough values were read in to ensure rows and cols of the matrix are properly represented
		if not len(split_vals) == cols:
			print("Incorrect number of values entered for the row")
			continue
		
		row = []

		# trys to add all the values entered into an array, which will eventually be put into the matrix
		for val in split_vals:
			# if a value entered can't be converted to a fraction will error prompt again
			try:
				row.append(Fraction(val))
			except:
				print("Failed reading the vals in the row. Make sure all the values are space separated and numbers")
				read_error_flag = True
				break
		if not read_error_flag:
			numpy.array(row)
			matrix[rows_read]= row
			rows_read += 1

	print("Original Matrix\n")
	print_fraction_matrix(matrix)
	print()

	min = numpy.amin(matrix)
	k = 0

	if min < 1:
		k = Fraction(abs(Fraction(1)-min))

	k_matrix = matrix + k

	print("k = " + str(k))
	print("Matrix + k\n")
	print_fraction_matrix(k_matrix)
	print()

	simplex_tableau = Tableau(k_matrix)

	print("INITIAL TABLEAU\n")
	print(str(simplex_tableau) + "\n")

	# for skipping the whole process
	print_steps = input("If you want to only see the tableaus enter Y: ")
	print()

	tab_num = 1
	# keeps going until final tableau reached
	while not simplex_tableau.done():
		pivot_coords = simplex_tableau.pivot()
		if not print_steps.upper() == "Y":
			print("\nFinding Tableau " + str(tab_num))
			pivot_col_num = pivot_coords[0]
			pivot_row_num = pivot_coords[1]
			pivot_val = pivot_coords[2]

			print("Pivot col: " + str(pivot_col_num) + ", row: "
			+ str(pivot_row_num) + ", value: " + str(pivot_val) + "\n")

			pivot_row_new = simplex_tableau._get_row(pivot_row_num)
			for x in range(len(pivot_row_new)):
				pivot_row_new[x] = pivot_row_new[x] / pivot_val
			print("The new pivot row is: ")
			print(print_fraction_array(pivot_row_new))
			print()

			while(True):
				t = input("Continue to new row calculations? (Y for yes N for no): ")
				if t.upper() == "Y":
					break
				elif t.upper() == "N":
					return

			for x in range(rows + 1):
				if not x == pivot_row_num:
					new_row = simplex_tableau._get_row(x)
					pivot_row_val = simplex_tableau._get_col(pivot_col_num)[x]
					pivot_row_subtract = pivot_row_new.copy()
					for i in range(len(pivot_row_new)):
						pivot_row_subtract[i] = pivot_row_subtract[i] * pivot_row_val
					new_row = numpy.subtract(new_row, pivot_row_subtract)
					print("\nThe new row " + str(x+1) + " is: ")
					print(print_fraction_array(new_row))
					print()
		
		simplex_tableau = simplex_tableau.next_tableau(pivot_coords)

		if(simplex_tableau.done()):
			break
		
		print("\nTABLEAU " + str(tab_num))
		print(str(simplex_tableau))
		tab_num += 1

		while(True):
			t = input("Continue to next tableau (Y for yes N for no): ")
			if t.upper() == "Y":
				break
			elif t.upper() == "N":
				return
		
	print("\nFINAL TABLEAU\n")
	print(str(simplex_tableau) + "\n")
	print("Row player's optimial strategy: " + print_fraction_array(simplex_tableau.row_strategy()))
	print("Column player's optimial strategy: " + print_fraction_array(simplex_tableau.column_strategy()))
	print("The value of the game is: " + str(simplex_tableau.value(k)))

main()