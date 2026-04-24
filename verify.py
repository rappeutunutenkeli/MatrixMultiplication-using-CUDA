import numpy as np
import re
import os

def extract_result_matrix(filename):
    if not os.path.exists(filename):
        print(f"Файл {filename} не найден!")
        return None
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    # Поиск блока матрицы (между "Результирующая матрица:" и пустой строкой)
    matrix_block = re.search(r'Результирующая матрица:\n(.*?)\n\nВремя выполнения', content, re.DOTALL)
    if not matrix_block:
        print("Не удалось найти блок матрицы в файле")
        return None
    lines = matrix_block.group(1).strip().split('\n')
    matrix = []
    for line in lines:
        numbers = line.strip().split()
        row = [int(x) for x in numbers if x]
        if row:
            matrix.append(row)
    return np.array(matrix)

def verify():
    base = "C:/Users/gayvo/cuda/"
    matrix_a_path = os.path.join(base, "matrix_a.txt")
    matrix_b_path = os.path.join(base, "matrix_b.txt")
    result_path = os.path.join(base, "result_cuda.txt")

    if not os.path.exists(matrix_a_path):
        print(f"Ошибка: файл {matrix_a_path} не найден!")
        return
    with open(matrix_a_path, 'r') as f:
        first_line = f.readline().strip()
        size = len(first_line.split())
    print(f"Размер матриц: {size}x{size}")

    # Читаем матрицы с помощью numpy 
    A = np.loadtxt(matrix_a_path, dtype=int)
    B = np.loadtxt(matrix_b_path, dtype=int)

    print("\nИсходные матрицы прочитаны:")
    print(f"Матрица A:\n{A}")
    print(f"\nМатрица B:\n{B}")

    # Вычисляем произведение через NumPy
    C_numpy = np.dot(A, B)
    print(f"\nNumPy результат:\n{C_numpy}")

    C_prog = extract_result_matrix(result_path)
    if C_prog is None:
        print("Не удалось прочитать результат программы.")
        return

    print(f"\nРезультат программы:\n{C_prog}")

    # Сравниваем
    if np.array_equal(C_numpy, C_prog):
        print("\nРезультаты совпадают.")
    else:
        print("\nРезультаты не совпадают!")
        diff = C_numpy - C_prog
        print(f"Разница:\n{diff}")

if __name__ == "__main__":
    verify()