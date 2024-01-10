import pandas as pd
import copy
import string

num_of_planes = int(input("Введите кол-во типов самолётов: "))
num_of_avialines = int(input("Введите кол-во авилиний: "))
num_of_vars = num_of_avialines * num_of_planes
num_of_restr = num_of_avialines + num_of_planes
function_goal = (str(input("Введите тип задачи: ")) == 'max')
coefficients = [0 for j in range(num_of_vars + 1)]
simplex_table = [[0 for j in range(num_of_vars + 1)] for i in range(num_of_restr)]
ineq_signs = []
bow = []
bow1 = []
basis = []
negatives_exist = True
show_steps = input("Отображать шаги решения? (да/нет): ").lower()
file_path = "План перевозок, симплекс метод.txt"

def write_to_file(file_path, text):
    with open(file_path, 'a') as file:
        file.write(text + '\n')

def Input(num_of_restr, num_of_vars):
    global coefficients, simplex_table, ineq_signs, bow, bow1

    objective_coeffs = input("Введите эксплутационные расходы на один самолёт(I II I II I II): ").split()
    for i in range(1, len(objective_coeffs) + 1):
        coefficients[i] = float(objective_coeffs[i - 1])

    for i in range(num_of_restr):
        constraint_input = input(f"Введите месячный объём перевозок, затем число самолётов: ").split()
        constraint_coeffs = constraint_input[:-2]
        constraint_sign = constraint_input[-2]
        constraint_free_term = constraint_input[-1]

        for j in range(1, len(constraint_coeffs) + 1):
            simplex_table[i][j] = float(constraint_coeffs[j - 1])
        simplex_table[i][0] = float(constraint_free_term)
        ineq_signs.append(constraint_sign)
        bow = ineq_signs
        bow1 = constraint_coeffs

def Filling(num_of_restr):
    for i in range(len(coefficients)):
        for j in range(num_of_restr):
            index_string[i] += (simplex_table[j][i] * coefficients[basis[j]])
        index_string[i] -= coefficients[i]

def Add_vars(num_of_restr, function_goal):
    global coefficients
    global basis
    extended_form_of_vars = 0
    artificial_vars = 0
    for i in ineq_signs:
        if i != '=': extended_form_of_vars += 1
        if i == '>=': artificial_vars += 1

    new_num_of_vars = extended_form_of_vars + artificial_vars
    single_matrix = [0 for j in range(new_num_of_vars)]

    for i in range(num_of_restr):
        simplex_table[i] = simplex_table[i] + single_matrix
        if ineq_signs[i] == '<=':
            print(ineq_signs)
            simplex_table[i][-new_num_of_vars + i] = 1.0
        if ineq_signs[i] == '>=':
            simplex_table[i][-new_num_of_vars + i] = -1.0
            simplex_table[i][-artificial_vars + i] = 1.0
        if simplex_table[i][0] < 0:
            basis.append(-1)
            simplex_table[i][-new_num_of_vars + i] = -1.0
            simplex_table[i][1 + i] = 1.0

    M = 10 ** 10
    if function_goal:
        M = -10 ** 10
    coefficients += ([0 for i in range(extended_form_of_vars)] + [M for i in range(artificial_vars)])
    for i in range(num_of_restr):
        for j in range(len(simplex_table[i]) - new_num_of_vars, len(simplex_table[i])):
            if simplex_table[i][j] == 1.0:
                basis.append(j)

def Find_pivot_column(function_goal) -> int:
    global negatives_exist
    element_line = 0
    index_element = 0
    for i in range(1, len(index_string)):
        if function_goal:
            if index_string[i] < element_line:
                element_line = index_string[i]
                index_element = i
        else:
            if index_string[i] > element_line:
                element_line = index_string[i]
                index_element = i
    if index_element == 0:
        for i in basis:
            for j in range(len(basis)):
                if num_of_vars+num_of_restr+j+1 in basis:
                    print('Задача не имеет решений')
        negatives_exist = False
    return index_element


def Find_pivot_line(num_of_restr) -> int:
    min_element_column = 10 ** 10
    index_min_element = 0
    for i in range(num_of_restr):
        if simplex_table[i][pivot_column] > 0:
            elem = simplex_table[i][0] / simplex_table[i][pivot_column]
            if elem >= 0 and elem < min_element_column:
                min_element_column = elem
                index_min_element = i
    return index_min_element

def Output(simplex_table, num_of_restr, deltas, header=None):
    df = pd.DataFrame(simplex_table)
    if header is not None:
        df.columns = header
    df.insert(0, 'Bx', [f'x{basis[i]}' for i in range(len(simplex_table))])
    pd.set_option("display.precision", 4)
    coefficients_row = ['Ci'] + coefficients
    df.loc[-1] = coefficients_row
    df.index = df.index + 1
    df = df.sort_index()
    df.loc[len(df.index)] = ['F'] + deltas
    print(df.to_string(index=False))
    write_to_file(file_path, df.to_string(index=False))
    function_row = ['Ci'] + index_string


Input(num_of_restr, num_of_vars)
initial_simplex_table = copy.deepcopy(simplex_table)
dual_simplex_table = [list(i) for i in zip(*simplex_table)]
dual_simplex_table[0] = coefficients[1:]
coefficients_dual = [row[0] for row in simplex_table]
counter = bow.count('>=')
counter1 = 0
for i in range(len(bow1)):
    if bow1[i] != 0:
        counter1 += 1

Add_vars(num_of_restr, function_goal)
index_string = [0 for i in range(len(coefficients))]
Filling(num_of_restr)
pivot_column = Find_pivot_column(function_goal)
pivot_line = Find_pivot_line(num_of_restr)
if show_steps == 'да':
    header = ["A{}".format(i) for i in range(len(coefficients))]
    print("Первая таблица значений:")
    write_to_file(file_path, "Первая таблица значений:")
    Output(simplex_table, num_of_restr, index_string, header)
m = 1

while negatives_exist:
    pivot_line = Find_pivot_line(num_of_restr)
    pivot_elem = simplex_table[pivot_line][pivot_column]

    basis[pivot_line] = pivot_column

    new_simplex_table = [0 for x in range(num_of_restr)]
    for i in range(num_of_restr):
        new_simplex_table[i] = [0 for x in range(len(simplex_table[i]))]
        if i == pivot_line:
            new_simplex_table[pivot_line] = [x/pivot_elem for x in simplex_table[pivot_line]]
    for i in range(num_of_restr):
        if i != pivot_line:
            for j in range(len(simplex_table[i])):
                new_simplex_table[i][j] = simplex_table[i][j] - (new_simplex_table[pivot_line][j] * simplex_table[i][pivot_column])
    simplex_table = new_simplex_table
    index_string = [0 for i in range(len(coefficients))]
    Filling(num_of_restr)
    if show_steps == 'да':
        print(f"Итерация: {m}")
        write_to_file(file_path, f"Итерация: {m}")
        Output(simplex_table, num_of_restr, index_string, header)
    pivot_column = Find_pivot_column(function_goal)
    m += 1

def economic_interpretation(solution, num_of_avialines):
    interpretation = "Результат распределения самолётов:\n"
    types = ["Тип 1", "Тип 2", "Тип 3"]
    airlines = ["Авиалиния 1", "Авиалиния 2"]

    for i in range(len(solution)):
        type_idx = i // num_of_avialines
        airline_idx = i % num_of_avialines
        interpretation += f"{types[type_idx]}, {airlines[airline_idx]}: {solution[i]} самолётов\n"

    return interpretation

product_name = []
for i in range(counter1):
    product_name.append(str(i + 1))

resource_name = []
for i in range(counter):
    resource_name.append(str(i + 1))

F = 0
for i in range(num_of_restr):
    F += (simplex_table[i][0] * coefficients[basis[i]])
print(f"Суммарные расходы: {F}.")
write_to_file(file_path, f"Суммарные расходы: {F}.")


solution = []

for i in range(1, num_of_vars + 1):
    if i in basis:
        solution.append(simplex_table[basis.index(i)][0])
    else:
        solution.append(0)

print(economic_interpretation(solution, num_of_avialines))
write_to_file(file_path, economic_interpretation(solution, num_of_avialines))
write_to_file(file_path,'------------------------------------------------------------------------------------------------------------------------------------')

function_goal_dual = not function_goal
function_goal_str = "max" if function_goal else "min"
function_goal_dual_str = "min" if function_goal else "max"
header_dual = ["A{}".format(i) for i in range(len(coefficients_dual))]