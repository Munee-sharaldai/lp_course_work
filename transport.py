import copy
import pandas as pd
import os

def findDiff(costs):
    rowDiff = []
    colDiff = []
    for i in range(len(costs)):
        arr = costs[i][:]
        arr.sort()
        rowDiff.append(arr[1] - arr[0])
    col = 0
    while col < len(costs[0]):
        arr = []
        for i in range(len(costs)):
            arr.append(costs[i][col])
        arr.sort()
        colDiff.append(arr[1] - arr[0])
        col += 1
    return rowDiff, colDiff

def voguels_approximation(costs, supply, demand, iteration, n, m):
    costs_new = [[0 for _ in range(m)] for _ in range(n)]
    INF = 10**3

    while max(supply) != 0 and max(demand) != 0:
        row_diff, col_diff = findDiff(costs)
        max_row_diff = max(row_diff)
        max_col_diff = max(col_diff)

        if max_row_diff >= max_col_diff:
            row_index = row_diff.index(max_row_diff)
            min_cost = min([c for c in costs[row_index] if c != INF])
            col_index = costs[row_index].index(min_cost)

            allocated = min(supply[row_index], demand[col_index])
            costs_new[row_index][col_index] += allocated
            supply[row_index] -= allocated
            demand[col_index] -= allocated

            if demand[col_index] == 0:
                for i in range(n):
                    costs[i][col_index] = INF
            if supply[row_index] == 0:
                costs[row_index] = [INF] * m
        else:
            col_index = col_diff.index(max_col_diff)
            min_cost = min([costs[i][col_index] for i in range(n) if costs[i][col_index] != INF])
            row_index = [i for i in range(n) if costs[i][col_index] == min_cost][0]

            allocated = min(supply[row_index], demand[col_index])
            costs_new[row_index][col_index] += allocated
            supply[row_index] -= allocated
            demand[col_index] -= allocated

            if demand[col_index] == 0:
                for i in range(n):
                    costs[i][col_index] = INF
            if supply[row_index] == 0:
                costs[row_index] = [INF] * m

    return costs_new


def fake_points():
    total_supply = sum(supply)
    total_demand = sum(demand)

    if total_demand < total_supply:
        deficit = total_supply - total_demand
        demand.append(deficit)

        for supply_row in costs:
            supply_row.extend([0])

    elif total_demand > total_supply:
        surplus = total_demand - total_supply
        supply.append(surplus)

        costs.append([0] * len(demand))

def check_degeneracy(plan_matrix):
    count_positive_elements = sum(plan_matrix[i][j] > 0 for i in range(len(supply)) for j in range(len(demand)))

    non_degeneracy_condition = len(supply) + len(demand) - 1
    return count_positive_elements != non_degeneracy_condition


def remove_columns_with_single_nonzero(matrix, column_indices, row_indices):
    removable_columns = []
    for col_index in column_indices:
        nonzero_count = 0
        for row_index in row_indices:
            if matrix[row_index][col_index] != 0:
                nonzero_count += 1
                if nonzero_count > 1:
                    break
        if nonzero_count <= 1:
            removable_columns.append(col_index)
    return removable_columns


def remove_rows_with_single_nonzero(matrix, column_indices, row_indices):
    removable_rows = []
    for row_index in row_indices:
        nonzero_count = 0
        for col_index in column_indices:
            if matrix[row_index][col_index] != 0:
                nonzero_count += 1
                if nonzero_count > 1:
                    break
        if nonzero_count <= 1:
            removable_rows.append(row_index)
    return removable_rows


def cycle_search(matrix):
    col_indices = list(range(len(matrix[0])))
    row_indices = list(range(len(matrix)))
    found_cycle = False

    while not found_cycle and col_indices and row_indices:
        columns_to_remove = remove_columns_with_single_nonzero(matrix, col_indices, row_indices)
        found_cycle = not columns_to_remove
        for col in columns_to_remove:
            col_indices.remove(col)

        rows_to_remove = remove_rows_with_single_nonzero(matrix, col_indices, row_indices)
        if rows_to_remove:
            found_cycle = False
        for row in rows_to_remove:
            row_indices.remove(row)

    if found_cycle:
        return trace_cycle(matrix, row_indices, col_indices)
    else:
        return []

def trace_cycle(matrix, row_indices, col_indices):
    cycle_coords = []
    for row in row_indices:
        for col in col_indices:
            if matrix[row][col] != 0:
                cycle_coords.append([row, col])
                break
        if cycle_coords:
            break

    while cycle_coords and (cycle_coords[0] != cycle_coords[-1] or len(cycle_coords) < 3):
        last_row, last_col = cycle_coords[-1]
        if len(cycle_coords) % 2 != 0:
            next_col = next((col for col in col_indices if col != last_col and matrix[last_row][col] != 0), None)
            if next_col is not None:
                cycle_coords.append([last_row, next_col])
        else:
            next_row = next((row for row in row_indices if row != last_row and matrix[row][last_col] != 0), None)
            if next_row is not None:
                cycle_coords.append([next_row, last_col])
    if cycle_coords and cycle_coords[0] == cycle_coords[-1]:
        cycle_coords.pop()
    return cycle_coords


def mitigate_degeneracy_effect(plan_matrix):
    for row_index in range(len(plan_matrix)):
        for col_index in range(len(plan_matrix[row_index])):
            if plan_matrix[row_index][col_index] == 0:
                original_value = plan_matrix[row_index][col_index]
                plan_matrix[row_index][col_index] = 0.001

                if not check_degeneracy(plan_matrix) and not cycle_search(plan_matrix):
                    return plan_matrix
                else:
                    plan_matrix[row_index][col_index] = original_value

    return plan_matrix


def potential_calculation(plan):
    array_u = [0] + [10000 for _ in range(1, len(plan))]
    array_v = [10000 for _ in range(len(plan[0]))]
    while 10000 in array_u or 10000 in array_v:
        for u in range(len(array_u)):
            if array_u[u] != 10000:
                for v in range(len(array_v)):
                    if plan[u][v] != 0 and array_v[v] == 10000:
                        array_v[v] = array_u[u] + costs[u][v]
        for v in range(len(array_v)):
            if array_v[v] != 10000:
                for u in range(len(array_u)):
                    if plan[u][v] != 0 and array_u[u] == 10000:
                        array_u[u] = array_v[v] - costs[u][v]
    return array_u, array_v


def find_smallest_element_in_matrix(cost_matrix):
    min_val, min_position = float('inf'), (0, 0)
    for row_idx, row in enumerate(cost_matrix):
        for col_idx, element in enumerate(row):
            if element < min_val:
                min_val, min_position = element, (row_idx, col_idx)
    return [min_val] + list(min_position)

def evaluation_matrix_completion(plan):
    evaluation_matrix = [[n for n in costs[m]] for m in range(len(costs))]
    array_u, array_v = potential_calculation(plan)
    for m in range(len(plan)):
        for n in range(len(plan[m])):
            evaluation_matrix[m][n] = evaluation_matrix[m][n] + array_u[m] - array_v[n]
    return evaluation_matrix

def adjust_plan_based_on_cycle(initial_plan, target_x, target_y):
    initial_plan[target_x][target_y] = 1
    found_cycle = cycle_search(initial_plan)
    adjusted_plan = [list(row) for row in initial_plan]
    initial_plan[target_x][target_y] = 0

    adjust_start_index = next((idx for idx, (x, y) in enumerate(found_cycle) if x == target_x and y == target_y), 0)
    adjust_start_index = (adjust_start_index - 1) % 2

    min_adjustment = min(initial_plan[x][y] for idx, (x, y) in enumerate(found_cycle) if idx % 2 == adjust_start_index)

    for idx, (x, y) in enumerate(found_cycle):
        if idx % 2 == adjust_start_index:
            adjusted_plan[x][y] -= min_adjustment
        else:
            adjusted_plan[x][y] += min_adjustment

    return adjusted_plan


def goal_calc(tmp_cost, costs_new):
    goal_function = 0
    m, n = len(tmp_cost), len(tmp_cost[0])

    for i in range(m):
        for j in range(n):
            goal_function += costs_new[i][j] * tmp_cost[i][j]

    return int(goal_function)

def write_to_file(file_path, text):
    with open(file_path, 'a') as file:
        file.write(text + '\n')

def print_economic_interpretation(costs_new, supply, demand):
    total_cost = goal_calc(costs, costs_new)
    write_to_file(file_path, f"Общая стоимость перевозок составит {total_cost} д.е.")

    is_fake_consumer = len(demand) > len(costs[0])

    for i, row in enumerate(costs_new):
        for j, val in enumerate(row):
            if val > 3:
                if is_fake_consumer and j == len(demand) - 1:
                    print(f"У поставщика А{i+1} остался невостребованным груз в количестве {int(round(val))} ед.")
                    write_to_file(file_path, f"У поставщика А{i+1} остался невостребованным груз в количестве {int(round(val))} ед.")
                else:
                    print(f"Груз поставщика А{i+1} в количестве {int(round(val))} ед. следует направить потребителю В{j+1}.")
                    write_to_file(file_path, f"Груз поставщика А{i+1} в количестве {int(round(val))} ед. следует направить потребителю В{j+1}.")

    is_fake_supplier = len(supply) > len(costs)

    if is_fake_supplier:
        fake_supplier_index = len(supply) - 1
        for j in range(len(demand)):
            val = costs_new[fake_supplier_index][j]
            if val > 0:
                unmet_demand = demand[j] - sum(row[j] for row in costs_new[:-1])
                if unmet_demand > 0:
                    print(f"Потребитель В{j+1} останется неудовлетворенным на {unmet_demand} ед. в связи с нехваткой груза.")
                    write_to_file(file_path, f"Потребитель В{j+1} останется неудовлетворенным на {unmet_demand} ед. в связи с нехваткой груза.")



def plan_optimization(show_steps):
    global costs_new
    iteration = 0
    if check_degeneracy(costs_new):
        mitigate_degeneracy_effect(costs_new)
    evaluation_matrix = evaluation_matrix_completion(costs_new)
    if find_smallest_element_in_matrix(evaluation_matrix)[0] == 0:
        df = pd.DataFrame(costs_new, index=[f'A{i + 1}' for i in range(len(costs_new))],
                          columns=[f'B{i + 1}' for i in range(len(costs_new[0]))])
        df = df.apply(lambda row: row.map(lambda x: 0 if x == 0.001 else x), axis=1)
        print(f'Iteration {0}:\n{df.to_string(header=True, index=True)}')
        write_to_file(file_path, f'Iteration {iteration}:\n{df.to_string(header=True, index=True)}')
    while find_smallest_element_in_matrix(evaluation_matrix)[0] < 0:
        if show_steps == 'да':
            df = pd.DataFrame(costs_new, index = [f'A{i+1}' for i in range(len(costs_new))], columns = [f'B{i+1}' for i in range(len(costs_new[0]))])
            df = df.apply(lambda row: row.map(lambda x: 0 if x == 0.001 else x), axis=1)
            print(f'Iteration {iteration}:\n{df.to_string(header=True, index=True)}')
            write_to_file(file_path, f'Iteration {iteration}:\n{df.to_string(header=True, index=True)}')
        coord_x, coord_y = find_smallest_element_in_matrix(evaluation_matrix)[1], find_smallest_element_in_matrix(evaluation_matrix)[2]
        costs_new = adjust_plan_based_on_cycle(costs_new, coord_x, coord_y)
        print(f'new{costs_new}')
        iteration += 1
        if check_degeneracy(costs_new):
            mitigate_degeneracy_effect(costs_new)
        evaluation_matrix = evaluation_matrix_completion(costs_new)

supply = list(map(int, input("Введите количество товара у производителя: ").split()))
demand = list(map(int, input("Введите количество потребностей товара у потребителя: ").split()))
costs = [list(map(int, input("Введите стоимости перевозок: ").split())) for i in range(len(supply))]
tmp_cost = copy.deepcopy(costs)
fake_points()
costs_temp = copy.deepcopy(costs)

fake_points()
iteration = 0
n = len(costs)
m = len(costs[0])
reference_plan = [[0 for n in range(len(demand))] for m in range(len(supply))]
costs_new = voguels_approximation(costs, supply, demand, iteration, n, m)
show_steps = input("Отображать шаги решения? (да/нет): ").lower()
file_path = "План перевозок.txt"
if show_steps == 'да':
    print('Опорный план: ')
    write_to_file(file_path, f'Опорный план')
    print(f'В ЦИКЛЕ {costs_new}')
    for i in costs_new:
        print(i)
        write_to_file(file_path, f'{i}')
plan_optimization(show_steps)
print("Минимальные затраты на перевозку:", goal_calc(tmp_cost, costs_new))
print_economic_interpretation(costs_new, supply, demand)
